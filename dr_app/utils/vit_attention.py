# utils/vit_attention.py
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional


# EncoderBlock imported from vit
from torchvision.models.vision_transformer import EncoderBlock


class ViTAttentionExtractor:
    """
    Extracts attention maps from a ViT model
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attn_maps = []
        self.hooks = []
        self._register_hooks()

    def _hook_fn(self, module, input, output):
        if hasattr(module, "last_attn") and module.last_attn is not None:
            self.attn_maps.append(module.last_attn.detach().cpu())

    def _register_hooks(self):
        """
        Attach hooks to all EncoderBlock modules in the model.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, EncoderBlock):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def clear(self):
        self.attn_maps = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def forward_and_capture(self, image_tensor):
        self.clear()
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image_tensor.unsqueeze(0).to(self.device))

        if not self.attn_maps:
            print("No attention maps were captured from EncoderBlock.last_attn.")
        else:
            print(f"Captured attention from {len(self.attn_maps)} encoder blocks.")

        return self.attn_maps

    def compute_rollout(self, discard_ratio: float = 0.0) -> np.ndarray:
        if not self.attn_maps:
            raise RuntimeError("No attention maps available. Call forward_and_capture first.")

        attn_mats = []

        for attn in self.attn_maps:
            A = attn[0].numpy() 

            if discard_ratio > 0.0:
                flat = A.reshape(-1)
                threshold = np.quantile(flat, discard_ratio)
                A[ A < threshold ] = 0.0

           
            A = A / (A.sum(axis=-1, keepdims=True) + 1e-8)

            N = A.shape[0]
            A = A + np.eye(N, dtype=np.float32)
            A = A / (A.sum(axis=-1, keepdims=True) + 1e-8)

            attn_mats.append(A)

        # multiply attention matrices from all layers
        rollout = attn_mats[0]
        for i in range(1, len(attn_mats)):
            rollout = attn_mats[i] @ rollout

        return rollout


def cls_attention_to_grid(cls_attn: np.ndarray) -> np.ndarray:
    """
    Converts CLS-to-patch attention vector into a square grid
    """
    num_patches = cls_attn.shape[0]
    grid_size = int(np.sqrt(num_patches))
    cls_attn = cls_attn[: grid_size * grid_size]  # safety
    grid = cls_attn.reshape(grid_size, grid_size)

    # Normalize to [0,1]
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return grid


def upsample_attention_to_image(attn_grid: np.ndarray, image_size: int) -> np.ndarray:
    """
    Upsample
    """
    heatmap = cv2.resize(attn_grid, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return heatmap


def create_retinal_mask_from_image(original_image: Image.Image) -> np.ndarray:
    """
    Simple retina mask
    """
    original_np = np.array(original_image) 
    gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
    mask = gray > 5
    return mask


def visualize_attention_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    save_path: Optional[str] = None,
    title_prefix: str = "Attention",
    apply_retina_mask: bool = True,
):
    original_np = np.array(original_image) 
    H, W, _ = original_np.shape

    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)

    # mask outside the retinal area
    if apply_retina_mask:
        mask = create_retinal_mask_from_image(original_image)
        heatmap_masked = np.zeros_like(heatmap)
        heatmap_masked[mask] = heatmap[mask]
        heatmap = heatmap_masked

    # Normalize again after masking
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    else:
        heatmap = np.zeros_like(heatmap)

    # Colorize
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(original_np, 0.5, heatmap_color, 0.5, 0)

        # Plot: 1x4 layout (original, heatmap, overlay, colorbar)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5),
        gridspec_kw={"width_ratios": [4, 4, 4, 0.2]},
    )

    # Original image
    axes[0].imshow(original_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap
    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"{title_prefix} Heatmap")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"{title_prefix} Overlay")
    axes[2].axis("off")

    # Colorbar panel
    axes[3].axis("off")
    cbar = fig.colorbar(im, ax=axes[3], fraction=0.8, pad=0.05)
    cbar.ax.tick_params(labelsize=8)
    axes[3].set_title("Attention\nIntensity", fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved attention visualization to {save_path}")
    plt.close(fig)
