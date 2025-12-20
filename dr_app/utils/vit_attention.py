# dr_app/utils/vit_attention.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


def _find_vit_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """Finds the ViT backbone inside a model wrapper."""
    for attr in ["backbone", "vit", "model", "net", "encoder"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    # If it's already a timm ViT, it should have .blocks
    return model


@dataclass
class _PatchedAttention:
    module: torch.nn.Module
    original_forward: callable


class ViTAttentionExtractor:
   
    def __init__(self, vit_model: torch.nn.Module, device: torch.device):
        self.device = device
        self.wrapper = vit_model
        self.vit = _find_vit_backbone(vit_model)

        self.attn_maps: List[torch.Tensor] = []
        self._patched: List[_PatchedAttention] = []

        self._patch_attention_modules()

    def _patch_attention_modules(self) -> None:
       
        if not hasattr(self.vit, "blocks"):
            raise RuntimeError(
                "Could not find transformer blocks on the ViT backbone. "
                "Expected attribute `.blocks` (timm ViT style)."
            )

        for blk in self.vit.blocks:
            if not hasattr(blk, "attn"):
                continue

            attn_mod = blk.attn
            if not hasattr(attn_mod, "forward"):
                continue

            orig_forward = attn_mod.forward

            def wrapped_forward(x, _orig=orig_forward, _self=self, _attn_mod=attn_mod, **kwargs):

                if hasattr(_attn_mod, "qkv") and hasattr(_attn_mod, "num_heads"):
                    B, N, C = x.shape
                    qkv = _attn_mod.qkv(x)
                    qkv = qkv.reshape(B, N, 3, _attn_mod.num_heads, C // _attn_mod.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4) 
                    q, k, v = qkv[0], qkv[1], qkv[2]

                    scale = getattr(_attn_mod, "scale", (q.shape[-1] ** -0.5))
                    attn = (q @ k.transpose(-2, -1)) * scale
                    attn = attn.softmax(dim=-1)

                    _self.attn_maps.append(attn.detach())

           
                return _orig(x, **kwargs)

            
            attn_mod.forward = wrapped_forward
            self._patched.append(_PatchedAttention(attn_mod, orig_forward))

    def clear(self) -> None:
        self.attn_maps = []

    @torch.no_grad()
    def forward_and_capture(self, img_tensor: torch.Tensor) -> List[torch.Tensor]:
       
        self.clear()

        if img_tensor.ndim == 3:
            x = img_tensor.unsqueeze(0)
        elif img_tensor.ndim == 4:
            x = img_tensor
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(img_tensor.shape)}")

        x = x.to(self.device)
        _ = self.wrapper(x)  
        return self.attn_maps

    def compute_rollout(self, discard_ratio: float = 0.0, head_fusion: str = "mean") -> torch.Tensor:
       
        if not self.attn_maps:
            raise RuntimeError("No attention maps available. Call forward_and_capture first.")

       
        attn_stack = self.attn_maps

       
        fused_layers = []
        for a in attn_stack:
            if head_fusion == "mean":
                fused = a.mean(dim=1)
            elif head_fusion == "max":
                fused = a.max(dim=1).values
            else:
                raise ValueError("head_fusion must be 'mean' or 'max'")
            fused_layers.append(fused)

       
        if discard_ratio > 0:
            for i, A in enumerate(fused_layers):
                B, N, _ = A.shape
                A2 = A.clone()
               
                flat = A2.view(B, -1)
                _, idx = flat.sort(dim=-1)
                num_discard = int(flat.shape[-1] * discard_ratio)
                if num_discard > 0:
                    discard_idx = idx[:, :num_discard]
                    flat.scatter_(1, discard_idx, 0.0)
                fused_layers[i] = flat.view(B, N, N)

       
        result = torch.eye(fused_layers[0].shape[-1], device=fused_layers[0].device).unsqueeze(0)
        for A in fused_layers:
            A = A + torch.eye(A.shape[-1], device=A.device).unsqueeze(0)
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            result = A @ result

        
        return result[0]


# -------------------------
# Helpers
# -------------------------
def cls_attention_to_grid(cls_to_patches: torch.Tensor) -> np.ndarray:
    """
    cls_to_patches: (num_patches,) tensor
    returns square grid (sqrt(patches) x sqrt(patches))
    """
    if isinstance(cls_to_patches, torch.Tensor):
        v = cls_to_patches.detach().cpu().numpy()
    else:
        v = np.asarray(cls_to_patches)

    n = v.shape[0]
    s = int(np.sqrt(n))
    if s * s != n:
        raise ValueError(f"Number of patches ({n}) is not a perfect square.")
    return v.reshape(s, s)


def upsample_attention_to_image(grid: np.ndarray, image_size: int) -> np.ndarray:
    """
    grid: (gh, gw)
    returns heatmap (image_size, image_size) float32
    """
    import cv2
    heat = cv2.resize(grid.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return heat
