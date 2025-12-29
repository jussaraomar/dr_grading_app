# experiments/train_efficientnet.py

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

from dr_app.model_defs.efficientnet_dr import EfficientNetDR
from dr_app.utils.data_loader import create_data_loaders, get_class_weights
from dr_app.utils.training import train_epoch, validate_epoch, calculate_kappa
from dr_app.config.paths import paths
from dr_app.config.params import TrainingParams, ModelParams


def main():
    # ------------------------------------------------------------
    # DEVICE + PATHS
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(paths.saved_models, exist_ok=True)
    os.makedirs(paths.results, exist_ok=True)
    training_logs_dir = os.path.join(paths.results, "training_logs")
    os.makedirs(training_logs_dir, exist_ok=True)

    # ------------------------------------------------------------
    # DATA LOADING & SPLIT
    # ------------------------------------------------------------
    main_csv = paths.main_csv
    if not os.path.exists(main_csv):
        print(f"CSV file not found: {main_csv}")
        print("Please update the path in train_efficientnet.py")
        return

    print("Loading and splitting data...")
    full_df = pd.read_csv(main_csv)
    print(f"Full dataset: {len(full_df)} samples")

    train_df, temp_df = train_test_split(
        full_df,
        test_size=0.3,
        random_state=42,
        stratify=full_df["diagnosis"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["diagnosis"]
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
    model = EfficientNetDR(num_classes=TrainingParams.NUM_CLASSES).to(device)
    print(f"Model parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params:   {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
    print("\n" + "=" * 60)
    print("EFFICIENTNET-B2: STAGE 1 - FROZEN BACKBONE TRAINING")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

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
        print(f"\nEpoch {epoch_idx}/{TrainingParams.NUM_EPOCHS_STAGE1}:")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_stage1, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        kappa = calculate_kappa(model, val_loader, device)

        scheduler_stage1.step()
        current_lr = scheduler_stage1.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        logs["stage1"]["train_loss"].append(train_loss)
        logs["stage1"]["val_loss"].append(val_loss)
        logs["stage1"]["train_acc"].append(train_acc)
        logs["stage1"]["val_acc"].append(val_acc)
        logs["stage1"]["val_kappa"].append(kappa)
        logs["stage1"]["lr"].append(current_lr)

        if kappa > best_kappa_stage1:
            best_kappa_stage1 = kappa
            torch.save(
                model.state_dict(),
                os.path.join(paths.saved_models, "efficientnet_stage1_best.pth")
            )
            improved_str = " (improved, model saved)"
        else:
            improved_str = ""

        print(
            f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}\n"
            f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n"
            f"  Kappa: {kappa:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.1f}s{improved_str}"
        )

    print(f"\nEfficientNet Stage 1 completed! Best Kappa: {best_kappa_stage1:.4f}")

    # ============================================================
    # STAGE 2 TRAINING
    # ============================================================
    print("\n" + "=" * 60)
    print("EFFICIENTNET-B2: STAGE 2 - FINE-TUNING WITH UNFROZEN BACKBONE")
    print("=" * 60)

    stage1_best_path = os.path.join(paths.saved_models, "efficientnet_stage1_best.pth")
    model.load_state_dict(torch.load(stage1_best_path, weights_only=True))

    print("Unfreezing backbone layers...")
    model.unfreeze_backbone()

    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=TrainingParams.BATCH_SIZE_STAGE2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"Stage 2 DataLoader created with batch size = {TrainingParams.BATCH_SIZE_STAGE2}")

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
        print(f"\nFine-tuning Epoch {epoch_idx}/{TrainingParams.NUM_EPOCHS_STAGE2}:")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_stage2, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        kappa = calculate_kappa(model, val_loader, device)

        scheduler_stage2.step()
        current_lr = scheduler_stage2.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        logs["stage2"]["train_loss"].append(train_loss)
        logs["stage2"]["val_loss"].append(val_loss)
        logs["stage2"]["train_acc"].append(train_acc)
        logs["stage2"]["val_acc"].append(val_acc)
        logs["stage2"]["val_kappa"].append(kappa)
        logs["stage2"]["lr"].append(current_lr)

        if kappa > best_kappa_stage2:
            best_kappa_stage2 = kappa
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(paths.saved_models, "efficientnet_finetuned_best.pth")
            )
            improved_str = " (improved, model saved)"
        else:
            patience_counter += 1
            improved_str = f" (no improvement, patience={patience_counter})"

        print(
            f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}\n"
            f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n"
            f"  Kappa: {kappa:.4f}, LR: {current_lr:.8f}, Time: {epoch_time:.1f}s{improved_str}"
        )

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered!")
            break

    # ------------------------------------------------------------
    # SAVE TRAINING CURVES LOGS
    # ------------------------------------------------------------
    curves_path = os.path.join(training_logs_dir, "efficientnet_training_curves.pkl")
    with open(curves_path, "wb") as f:
        pickle.dump(logs, f)
    print(f"\nSaved training curve logs → {curves_path}")

    # ============================================================
    # FINAL TRAINING SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("EFFICIENTNET-B2 FINAL TRAINING RESULTS")
    print("=" * 60)
    print(f"Stage 1 - Best Kappa: {best_kappa_stage1:.4f}")
    print(f"Stage 2 - Best Kappa: {best_kappa_stage2:.4f}")
    print(f"Improvement: {best_kappa_stage2 - best_kappa_stage1:+.4f}")

    # Choose best checkpoint for test evaluation
    if best_kappa_stage2 > best_kappa_stage1:
        best_model_path = os.path.join(paths.saved_models, "efficientnet_finetuned_best.pth")
        print("Using best fine-tuned (Stage 2) EfficientNet model for final evaluation.")
    else:
        best_model_path = stage1_best_path
        print("Stage 2 did not improve Kappa; using best Stage 1 EfficientNet model.")

    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # ============================================================
    # FINAL TEST EVALUATION
    # ============================================================
    print("\nFINAL TEST SET EVALUATION")
    from utils.evaluation import evaluate_model
    test_results = evaluate_model(model, test_loader, device)

    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Kappa:    {test_results['kappa']:.4f}")

    # ------------------------------------------------------------
    # SAVE DETAILED TEST RESULTS
    # ------------------------------------------------------------
    test_results_pkl = os.path.join(paths.results, "efficientnet_test_results.pkl")
    with open(test_results_pkl, "wb") as f:
        pickle.dump(test_results, f)

    if "confusion_matrix" in test_results:
        cm_df = pd.DataFrame(test_results["confusion_matrix"])
        cm_csv_path = os.path.join(paths.results, "efficientnet_confusion_matrix.csv")
        cm_df.to_csv(cm_csv_path, index=False)
        print(f"Confusion matrix saved → {cm_csv_path}")

    if "classification_report" in test_results:
        cr_txt_path = os.path.join(paths.results, "efficientnet_classification_report.txt")
        with open(cr_txt_path, "w", encoding="utf-8") as f:
            f.write(str(test_results["classification_report"]))
        print(f"Classification report saved → {cr_txt_path}")

    # Save summary results
    summary = {
        "model": "EfficientNet-B2",
        "stage1_best_kappa": best_kappa_stage1,
        "stage2_best_kappa": best_kappa_stage2,
        "test_accuracy": float(test_results["accuracy"]),
        "test_kappa": float(test_results["kappa"]),
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_model_path_used_for_test": best_model_path
    }

    results_file = os.path.join(paths.results, "efficientnet_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(summary, f)

    print(f"\nEfficientNet-B2 training completed!")
    print(f"Summary saved as: {results_file}")
    print(f"Full test results saved as: {test_results_pkl}")


if __name__ == "__main__":
    main()
