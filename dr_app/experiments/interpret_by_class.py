#experiments/interpret_by_class.py

import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
from collections import defaultdict
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import ResNetDR, EfficientNetDR, ViTDR
from dr_app.utils.data_loader import create_data_loaders
from dr_app.utils.grad_cam import ResNetGradCAM, EfficientNetGradCAM
from utils.vit_attention import (
    ViTAttentionExtractor,
    cls_attention_to_grid,
    upsample_attention_to_image,
)
from dr_app.config.paths import paths


def create_retinal_mask_from_image(original_image):
    original_np = np.array(original_image)  # [H, W, 3]
    gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
    mask = gray > 5
    return mask


def setup_models(device):
    """Load all trained models."""
    models = {}

    # ResNet
    try:
        resnet = ResNetDR(num_classes=5).to(device)
        resnet.load_state_dict(
            torch.load(f"{paths.saved_models}/resnet_finetuned_best.pth",
                       weights_only=True)
        )
        resnet.eval()
        models["resnet"] = resnet
        print(" ResNet-50 loaded successfully")
    except Exception as e:
        print(f" Failed to load ResNet: {e}")

    # EfficientNet
    try:
        efficientnet = EfficientNetDR(num_classes=5).to(device)
        efficientnet.load_state_dict(
            torch.load(f"{paths.saved_models}/efficientnet_finetuned_best.pth",
                       weights_only=True)
        )
        efficientnet.eval()
        models["efficientnet"] = efficientnet
        print(" EfficientNet-B2 loaded successfully")
    except Exception as e:
        print(f" Failed to load EfficientNet: {e}")

    # ViT
    try:
        vit = ViTDR(num_classes=5).to(device)
        vit.load_state_dict(
            torch.load(f"{paths.saved_models}/vit_finetuned_best.pth",
                       weights_only=True)
        )
        vit.eval()
        models["vit"] = vit
        print(" Vision Transformer loaded successfully")
    except Exception as e:
        print(f" Failed to load ViT: {e}")

    return models


def get_samples_by_class(test_df, test_loader, samples_per_class=10):
    """
    Collect up to `samples_per_class` examples per DR grade from the test set.
    """
    import random

    all_class_samples = defaultdict(list)

    print(f"\n Scanning test set for class distribution...")

    for batch_idx, (images, labels) in enumerate(test_loader):
        true_class = labels.item()
        image_id = test_df.iloc[batch_idx]["id_code"]

        all_class_samples[true_class].append(
            {
                "batch_idx": batch_idx,
                "image_tensor": images[0],
                "image_id": image_id,
                "true_class": true_class,
            }
        )

    print(" Available samples per class:")
    total_samples = 0
    for class_idx in sorted(all_class_samples.keys()):
        count = len(all_class_samples[class_idx])
        print(f"  Class {class_idx}: {count} samples")
        total_samples += count

    print(f" Total test samples: {total_samples}")

    # Randomly select samples from each class
    class_samples = defaultdict(list)
    random.seed(42)

    for class_idx, samples in all_class_samples.items():
        if len(samples) >= samples_per_class:
            shuffled = samples.copy()
            random.shuffle(shuffled)
            selected = shuffled[:samples_per_class]
            class_samples[class_idx] = selected
            print(
                f"✅ Class {class_idx}: Randomly selected {samples_per_class} samples"
            )
        else:
            class_samples[class_idx] = samples
            print(
                f" Class {class_idx}: Only {len(samples)} samples available "
                f"(requested {samples_per_class})"
            )

    return class_samples


def interpret_class_samples(models, class_samples, outputs_dir, device):
    """Generate interpretability visualisations for all models."""
    all_results = {}

    # Create a single ViT attention extractor (reused for all samples)
    vit_att_extractor = None
    if "vit" in models:
        vit_att_extractor = ViTAttentionExtractor(models["vit"], device)

    for class_idx in sorted(class_samples.keys()):
        print(f"\n Processing Class {class_idx}")
        print("=" * 50)

        class_dir = os.path.join(outputs_dir, f"class_{class_idx}")
        os.makedirs(class_dir, exist_ok=True)

        class_results = {}

        for sample_idx, sample_data in enumerate(class_samples[class_idx]):
            print(f"\n  Sample {sample_idx + 1}/{len(class_samples[class_idx])}")
            print(f"  Image ID: {sample_data['image_id']}")

            # Original full-resolution image
            original_img_path = os.path.join(
                paths.images_dir, sample_data["image_id"] + ".png"
            )
            original_image = Image.open(original_img_path).convert("RGB")

            sample_results = interpret_single_sample(
                models,
                sample_data["image_tensor"],
                original_image,
                sample_data["true_class"],
                sample_idx + 1,
                class_dir,
                device,
                sample_data["image_id"],
                vit_att_extractor,
            )

            class_results[sample_idx + 1] = sample_results

        all_results[class_idx] = class_results

    # Clean up ViT hooks
    if vit_att_extractor is not None:
        vit_att_extractor.remove_hooks()

    return all_results


def interpret_single_sample(
    models,
    image_tensor,
    original_image,
    true_class,
    sample_idx,
    outputs_dir,
    device,
    image_id=None,
    vit_att_extractor=None,
):
    """
    Generate interpretability visualisations for all models on one sample.
    """
    # Create sample directory
    sample_dir = os.path.join(outputs_dir, f"sample_{sample_idx}")
    if image_id:
        sample_dir = os.path.join(outputs_dir, f"sample_{sample_idx}_{image_id}")
    os.makedirs(sample_dir, exist_ok=True)

    print(f"    Analyzing (True Class: {true_class})")

    results = {}

    # -------------------------------------------------
    # ResNet Grad-CAM
    # -------------------------------------------------
    if "resnet" in models:
        try:
            resnet_cam = ResNetGradCAM(models["resnet"], device)
            save_path = os.path.join(sample_dir, "resnet_gradcam.png")
            result = resnet_cam.visualize_cam(
                image_tensor,
                original_image,
                true_class=true_class,
                class_idx=None,
                save_path=save_path,
            )
            results["resnet"] = result
            del resnet_cam
            print("    ResNet Grad-CAM completed")
        except Exception as e:
            print(f"    ResNet Grad-CAM failed: {e}")

    # -------------------------------------------------
    # EfficientNet Grad-CAM
    # -------------------------------------------------
    if "efficientnet" in models:
        try:
            efficientnet_cam = EfficientNetGradCAM(models["efficientnet"], device)
            save_path = os.path.join(sample_dir, "efficientnet_gradcam.png")
            result = efficientnet_cam.visualize_cam(
                image_tensor,
                original_image,
                true_class=true_class,
                class_idx=None,
                save_path=save_path,
            )
            results["efficientnet"] = result
            del efficientnet_cam
            print("    EfficientNet Grad-CAM completed")
        except Exception as e:
            print(f"    EfficientNet Grad-CAM failed: {e}")

    # -------------------------------------------------
    # ViT Attention Rollout
    # -------------------------------------------------
    if "vit" in models and vit_att_extractor is not None:
        try:
            # 1) Capture attention maps from encoder blocks
            att_maps = vit_att_extractor.forward_and_capture(image_tensor)
            if not att_maps:
                print("    ViT attention: no maps captured, skipping.")
            else:
                # 2) Rollout -> [N, N]
                rollout = vit_att_extractor.compute_rollout(discard_ratio=0.0)

                # 3) CLS-to-patch attention (drop CLS token itself)
                cls_to_all = rollout[0]        # [N]
                cls_to_patches = cls_to_all[1:]  # [P]

                # 4) Vector -> patch grid -> upsample to model input size
                attn_grid = cls_attention_to_grid(cls_to_patches)
                _, H_in, W_in = image_tensor.shape
                heatmap_small = upsample_attention_to_image(attn_grid, image_size=H_in)

                # 5) Resize to original image resolution & mask background
                orig_w, orig_h = original_image.size
                heatmap_resized = cv2.resize(
                    heatmap_small, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
                )

                mask = create_retinal_mask_from_image(original_image)
                heatmap_masked = np.zeros_like(heatmap_resized)
                heatmap_masked[mask] = heatmap_resized[mask]

                # Normalize to [0,1]
                if heatmap_masked.max() > heatmap_masked.min():
                    vit_cam = (heatmap_masked - heatmap_masked.min()) / (
                        heatmap_masked.max() - heatmap_masked.min() + 1e-8
                    )
                else:
                    vit_cam = np.zeros_like(heatmap_masked)

                # 6) Colour overlay for saving & comparison
                original_np = np.array(original_image)
                heatmap_uint8 = np.uint8(255 * vit_cam)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(original_np, 0.5, heatmap_color, 0.5, 0)

                # 7) Get ViT prediction & confidence
                with torch.no_grad():
                    logits = models["vit"](image_tensor.unsqueeze(0).to(device))
                    pred_class = logits.argmax(dim=1).item()
                    confidence = torch.softmax(logits, dim=1)[0, pred_class].item()

                                # 8) Save standalone ViT attention figure (with separate, narrow colorbar axis)
                fig, axes = plt.subplots(
                    1,
                    4,
                    figsize=(18, 5),
                    gridspec_kw={"width_ratios": [4, 4, 4, 0.2]},
                )

                # Original
                axes[0].imshow(original_np)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                # Heatmap
                im = axes[1].imshow(vit_cam, cmap="jet")
                axes[1].set_title(f"ViT Attention Heatmap\nTrue: {true_class}")
                axes[1].axis("off")

                # Overlay
                axes[2].imshow(overlay)
                axes[2].set_title(
                    f"ViT Attention Overlay\nTrue: {true_class}, Pred: {pred_class}"
                )
                axes[2].axis("off")

                # Colorbar panel
                axes[3].axis("off")
                cbar = fig.colorbar(im, ax=axes[3], fraction=0.8, pad=0.05)
                cbar.ax.tick_params(labelsize=8)
                axes[3].set_title("Attention\nIntensity", fontsize=10)

                plt.tight_layout()




                vit_save_path = os.path.join(sample_dir, "vit_attention.png")
                plt.savefig(vit_save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print("    ViT attention completed")

                results["vit"] = {
                    "cam": vit_cam,
                    "overlay": overlay,
                    "pred_class": pred_class,
                    "true_class": true_class,
                    "confidence": confidence,
                }

        except Exception as e:
            print(f"    ViT attention failed: {e}")

    # -------------------------------------------------
    # Combined comparison figure
    # -------------------------------------------------
    if results:
        create_comparison_figure(results, original_image, true_class, sample_dir, sample_idx)

    return results


def create_comparison_figure(results, original_image, true_class, sample_dir, sample_idx):
    """
    Create a comparison figure showing original image + each model's heatmap & overlay.
    Has:
      - row 0: original + per-model heatmaps
      - row 1: per-model overlays
      - last column: shared colorbar
    """
    if len(results) < 2:
        return

    num_models = len(results)

    # columns: 1 (original) + num_models (heatmaps) + 1 (colorbar)
    fig, axes = plt.subplots(
        2,
        num_models + 2,
        figsize=(5 * (num_models + 2), 10),
        gridspec_kw={
            "width_ratios": [4] + [4] * num_models + [0.25]  # narrow last column
        },
    )

    original_np = np.array(original_image)
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title(f"Original Image\nTrue Class: {true_class}", fontsize=12)
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    # Retina mask (same for all models)
    orig_w, orig_h = original_image.size
    mask = create_retinal_mask_from_image(original_image)

    im_for_cbar = None  # store one heatmap handle for the shared colorbar

    # Model results
    for col, (model_name, result) in enumerate(results.items(), start=1):
        if "cam" in result:
            base_map = result["cam"]
        elif "saliency_map" in result:
            base_map = result["saliency_map"]
        else:
            continue

        # Resize & mask CAM
        cam_resized = cv2.resize(base_map, (orig_w, orig_h))
        cam_resized_masked = np.zeros_like(cam_resized)
        cam_resized_masked[mask] = cam_resized[mask]

        # Heatmap
        im = axes[0, col].imshow(cam_resized_masked, cmap="jet")
        axes[0, col].set_title(f"{model_name.upper()}\nHeatmap", fontsize=12)
        axes[0, col].axis("off")

        if im_for_cbar is None:
            im_for_cbar = im

        # Overlay
        axes[1, col].imshow(result["overlay"])
        pred_class = result["pred_class"]
        confidence = result["confidence"]
        result_true_class = result.get("true_class", true_class)
        title = (
            f"True: {result_true_class}, Pred: {pred_class}\nConf: {confidence:.3f}"
        )
        axes[1, col].set_title(title, fontsize=12)
        axes[1, col].axis("off")

    # Last column: shared colorbar
    axes[0, -1].axis("off")
    axes[1, -1].axis("off")
    if im_for_cbar is not None:
        cbar = fig.colorbar(im_for_cbar, ax=axes[0, -1], fraction=0.8, pad=0.05)
        cbar.ax.tick_params(labelsize=8)
        axes[0, -1].set_title("Heatmap\nIntensity", fontsize=10)

    plt.tight_layout()
    comparison_path = os.path.join(sample_dir, "model_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("    Model comparison saved")


def create_class_summary(all_results, outputs_dir):
    """
    Create a CSV and console summary of correctness per class & per model.
    """
    print(f"\n CREATING CLASS SUMMARY")
    print("=" * 60)

    summary_data = []

    for class_idx, class_results in all_results.items():
        print(f"\nClass {class_idx}:")
        print("-" * 30)

        for sample_idx, sample_results in class_results.items():
            if sample_results:
                for model_name, result in sample_results.items():
                    summary_data.append(
                        {
                            "class": class_idx,
                            "sample": sample_idx,
                            "model": model_name,
                            "true_class": result.get("true_class", class_idx),
                            "pred_class": result["pred_class"],
                            "confidence": result["confidence"],
                            "correct": result.get("true_class", class_idx)
                            == result["pred_class"],
                        }
                    )

                    status = (
                        " CORRECT"
                        if result.get("true_class", class_idx)
                        == result["pred_class"]
                        else " WRONG"
                    )
                    print(
                        f"  Sample {sample_idx} - {model_name.upper()}: {status} "
                        f"(True: {result.get('true_class', class_idx)}, "
                        f"Pred: {result['pred_class']}, "
                        f"Conf: {result['confidence']:.3f})"
                    )

    summary_df = pd.DataFrame(summary_data)

    summary_csv_path = os.path.join(outputs_dir, "class_interpretation_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n Summary saved to: {summary_csv_path}")

    print(f"\n OVERALL STATISTICS")
    print("=" * 40)
    for model_name in summary_df["model"].unique():
        model_data = summary_df[summary_df["model"] == model_name]
        accuracy = model_data["correct"].mean() * 100
        total_samples = len(model_data)
        print(
            f"{model_name.upper()}: {accuracy:.1f}% accuracy ({total_samples} samples)"
        )

    print(f"\n ACCURACY BY CLASS")
    print("=" * 30)
    for class_idx in sorted(summary_df["class"].unique()):
        class_data = summary_df[summary_df["class"] == class_idx]
        for model_name in class_data["model"].unique():
            model_class_data = class_data[class_data["model"] == model_name]
            accuracy = model_class_data["correct"].mean() * 100
            total = len(model_class_data)
            correct = model_class_data["correct"].sum()
            print(
                f"Class {class_idx} - {model_name.upper()}: "
                f"{correct}/{total} correct ({accuracy:.1f}%)"
            )

    return summary_df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load all models
    models = setup_models(device)
    if not models:
        print(" No models loaded successfully!")
        return

    main_csv = paths.main_csv
    full_df = pd.read_csv(main_csv)

    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        full_df,
        test_size=0.3,
        random_state=42,
        stratify=full_df["diagnosis"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["diagnosis"],
    )

    images_dir = paths.images_dir
    _, _, test_loader = create_data_loaders(
        train_df,
        val_df,
        test_df,
        images_dir=images_dir,
        batch_size=1,
        image_size=(224, 224),
        use_cache=True,
    )

    outputs_dir = os.path.join(paths.outputs, "class_based_interpretability")
    os.makedirs(outputs_dir, exist_ok=True)

    print(f"\n Generating class-based interpretability visualizations")
    print(f" Results will be saved to: {outputs_dir}")

    samples_per_class = 15
    class_samples = get_samples_by_class(test_df, test_loader, samples_per_class)

    all_results = interpret_class_samples(
        models, class_samples, outputs_dir, device
    )

    summary_df = create_class_summary(all_results, outputs_dir)

    print(f"\n CLASS-BASED INTERPRETABILITY ANALYSIS COMPLETE!")
    print(f" All results saved to: {outputs_dir}")
    print(f" Summary statistics saved to CSV")
    print(
        f" Analyzed {samples_per_class} samples from each of {len(class_samples)} classes"
    )


if __name__ == "__main__":
    main()
