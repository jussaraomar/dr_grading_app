# experiments/train_resnet.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os
import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models import ResNetDR
from utils.data_loader import create_data_loaders, get_class_weights
from utils.training import train_epoch, validate_epoch, calculate_kappa
from config.paths import paths
from config.params import TrainingParams
from config.params import ModelParams


def main():
    # ------------------------------------------------------------
    # DEVICE + PATHS
    # ------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Make sure important directories exist
    os.makedirs(paths.saved_models, exist_ok=True)
    os.makedirs(paths.results, exist_ok=True)
    training_logs_dir = os.path.join(paths.results, "training_logs")
    os.makedirs(training_logs_dir, exist_ok=True)

    # ------------------------------------------------------------
    # DATA LOADING & SPLIT
    # ------------------------------------------------------------
    full_df = pd.read_csv(paths.main_csv)
    train_df, temp_df = train_test_split(
        full_df,
        test_size=0.3,
        random_state=42,
        stratify=full_df['diagnosis']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['diagnosis']
    )

    print(f"Data split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    class_weights_tensor = get_class_weights(train_df).to(device)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df,
        images_dir=paths.images_dir,
        batch_size=TrainingParams.BATCH_SIZE_STAGE1,
        image_size=ModelParams.IMAGE_SIZE,
        use_cache=True
    )

    # ------------------------------------------------------------
    # MODEL SETUP
    # ------------------------------------------------------------
    model = ResNetDR(num_classes=TrainingParams.NUM_CLASSES).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------
    # METRIC STORAGE FOR CURVES
    # ------------------------------------------------------------
    logs = {
        "stage1": {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_kappa": [],
            "lr": []
        },
        "stage2": {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_kappa": [],
            "lr": []
        }
    }

    # ============================================================
    # STAGE 1 TRAINING 
    # ============================================================
    print("\n=== STAGE 1 TRAINING ===")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=0.1
    )
    optimizer_stage1 = AdamW(
        model.parameters(),
        lr=TrainingParams.LEARNING_RATE_STAGE1,
        weight_decay=1e-4
    )
    scheduler_stage1 = CosineAnnealingLR(
        optimizer_stage1,
        T_max=TrainingParams.NUM_EPOCHS_STAGE1
    )

    best_kappa_stage1 = 0.0

    for epoch in range(TrainingParams.NUM_EPOCHS_STAGE1):
        epoch_idx = epoch + 1
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer_stage1, device
        )

        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # Kappa on validation
        kappa = calculate_kappa(model, val_loader, device)

        # Step scheduler
        scheduler_stage1.step()
        current_lr = scheduler_stage1.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        # Log metrics
        logs["stage1"]["train_loss"].append(train_loss)
        logs["stage1"]["val_loss"].append(val_loss)
        logs["stage1"]["train_acc"].append(train_acc)
        logs["stage1"]["val_acc"].append(val_acc)
        logs["stage1"]["val_kappa"].append(kappa)
        logs["stage1"]["lr"].append(current_lr)

        # Save best Stage 1 model by kappa
        if kappa > best_kappa_stage1:
            best_kappa_stage1 = kappa
            torch.save(
                model.state_dict(),
                os.path.join(paths.saved_models, "resnet_stage1_best.pth")
            )
            improved_str = " (improved, model saved)"
        else:
            improved_str = ""

        print(
            f"Epoch {epoch_idx}/{TrainingParams.NUM_EPOCHS_STAGE1} "
            f"- Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
            f"Kappa={kappa:.4f}, LR={current_lr:.2e}, "
            f"Time={epoch_time:.1f}s{improved_str}"
        )

    print(f"\nBest Stage 1 Kappa: {best_kappa_stage1:.4f}")

    # ============================================================
    # STAGE 2 TRAINING 
    # ============================================================
    print("\n=== STAGE 2 FINE-TUNING ===")

    # Load best Stage 1 weights
    stage1_best_path = os.path.join(paths.saved_models, "resnet_stage1_best.pth")
    model.load_state_dict(torch.load(stage1_best_path, weights_only=True))

    # Unfreeze backbone
    model.unfreeze_backbone()

    # New DataLoader with Stage 2 batch size
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=TrainingParams.BATCH_SIZE_STAGE2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer_stage2 = AdamW(
        model.parameters(),
        lr=TrainingParams.LEARNING_RATE_STAGE2,
        weight_decay=1e-4
    )
    scheduler_stage2 = CosineAnnealingLR(
        optimizer_stage2,
        T_max=TrainingParams.NUM_EPOCHS_STAGE2
    )

    best_kappa_stage2 = best_kappa_stage1
    patience_counter = 0
    early_stopping_patience = 5

    for epoch in range(TrainingParams.NUM_EPOCHS_STAGE2):
        epoch_idx = epoch + 1
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer_stage2, device
        )

        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # Kappa on validation
        kappa = calculate_kappa(model, val_loader, device)

        # Step scheduler
        scheduler_stage2.step()
        current_lr = scheduler_stage2.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        # Log metrics
        logs["stage2"]["train_loss"].append(train_loss)
        logs["stage2"]["val_loss"].append(val_loss)
        logs["stage2"]["train_acc"].append(train_acc)
        logs["stage2"]["val_acc"].append(val_acc)
        logs["stage2"]["val_kappa"].append(kappa)
        logs["stage2"]["lr"].append(current_lr)

        # Early stopping on kappa
        if kappa > best_kappa_stage2:
            best_kappa_stage2 = kappa
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(paths.saved_models, "resnet_finetuned_best.pth")
            )
            improved_str = " (improved, model saved)"
        else:
            patience_counter += 1
            improved_str = f" (no improvement, patience={patience_counter})"

        print(
            f"FT Epoch {epoch_idx}/{TrainingParams.NUM_EPOCHS_STAGE2} "
            f"- Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
            f"Kappa={kappa:.4f}, LR={current_lr:.2e}, "
            f"Time={epoch_time:.1f}s{improved_str}"
        )

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    print(f"\nBest Stage 2 Kappa: {best_kappa_stage2:.4f}")

    # ============================================================
    # SAVE TRAINING CURVES LOGS
    # ============================================================
    curves_path = os.path.join(training_logs_dir, "resnet_training_curves.pkl")
    with open(curves_path, "wb") as f:
        pickle.dump(logs, f)

    print(f"\nSaved training curve logs → {curves_path}")

    # ============================================================
    # FINAL TEST EVALUATION
    # ============================================================
    # Choose best overall model
    if best_kappa_stage2 > best_kappa_stage1:
        best_model_path = os.path.join(paths.saved_models, "resnet_finetuned_best.pth")
        print("Using best fine-tuned (Stage 2) model for final test evaluation.")
    else:
        best_model_path = stage1_best_path
        print("Stage 2 did not beat Stage 1; using best Stage 1 model for final test evaluation.")

    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    from utils.evaluation import evaluate_model
    test_results = evaluate_model(model, test_loader, device)

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Kappa:    {test_results['kappa']:.4f}")

    # ------------------------------------------------------------
    # SAVE DETAILED TEST RESULTS (FOR THESIS FIGURES/TABLES)
    # ------------------------------------------------------------
    # 1) Save entire test_results dict
    test_results_path = os.path.join(paths.results, "resnet_test_results.pkl")
    with open(test_results_path, "wb") as f:
        pickle.dump(test_results, f)


    # 3) Classification report as TXT (if present)
    if "classification_report" in test_results:
        cr_txt_path = os.path.join(paths.results, "resnet_classification_report.txt")
        with open(cr_txt_path, "w") as f:
            f.write(str(test_results["classification_report"]))
        print(f"Classification report saved → {cr_txt_path}")

    print(f"\nTest results dict saved → {test_results_path}")
    print("\nResNet training + evaluation completed.")

if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     TARGET_KAPPA = 0.90
#     MAX_RUNS = 10  
#     run_number = 1

#     while run_number <= MAX_RUNS:
#         print("\n" + "=" * 70)
#         print(f" ResNet Training run {run_number}")
#         print("=" * 70)

#         test_kappa = main()  

#         print(f"\nRun {run_number} finished with Test Kappa = {test_kappa:.4f}")

#         if test_kappa >= TARGET_KAPPA:
#             print(f"\n SUCCESS: Achieved target Test Kappa ≥ {TARGET_KAPPA:.2f}")
#             break

#         print(f" Test Kappa below target ({TARGET_KAPPA:.2f}). Retrying...\n")
#         run_number += 1

#     if run_number > MAX_RUNS:
#         print(
#             f"\n Stopped after {MAX_RUNS} runs without reaching "
#             f"Test Kappa ≥ {TARGET_KAPPA:.2f}."
#         )
