# # dr_app/utils/vit_attention.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import List, Optional

# import numpy as np
# import torch


# def _find_vit_backbone(model: torch.nn.Module) -> torch.nn.Module:
#     """Finds the ViT backbone inside a model wrapper."""
#     for attr in ["backbone", "vit", "model", "net", "encoder"]:
#         if hasattr(model, attr):
#             return getattr(model, attr)
#     # If it's already a timm ViT, it should have .blocks
#     return model


# @dataclass
# class _PatchedAttention:
#     module: torch.nn.Module
#     original_forward: callable


# class ViTAttentionExtractor:
   
#     def __init__(self, vit_model: torch.nn.Module, device: torch.device):
#         self.device = device
#         self.wrapper = vit_model
#         self.vit = _find_vit_backbone(vit_model)

#         self.attn_maps: List[torch.Tensor] = []
#         self._patched: List[_PatchedAttention] = []

#         self.supported = False
#         self._patch_attention_modules()


#     def _patch_attention_modules(self) -> None:
       
#         if not hasattr(self.vit, "blocks"):
#             # Not a timm ViT backbone (could be torchvision VisionTransformer, etc.)
#             self.supported = False
#             return

#         self.supported = True

#         for blk in self.vit.blocks:
#             if not hasattr(blk, "attn"):
#                 continue

#             attn_mod = blk.attn
#             if not hasattr(attn_mod, "forward"):
#                 continue

#             orig_forward = attn_mod.forward

#             def wrapped_forward(x, _orig=orig_forward, _self=self, _attn_mod=attn_mod, **kwargs):

#                 if hasattr(_attn_mod, "qkv") and hasattr(_attn_mod, "num_heads"):
#                     B, N, C = x.shape
#                     qkv = _attn_mod.qkv(x)
#                     qkv = qkv.reshape(B, N, 3, _attn_mod.num_heads, C // _attn_mod.num_heads)
#                     qkv = qkv.permute(2, 0, 3, 1, 4) 
#                     q, k, v = qkv[0], qkv[1], qkv[2]

#                     scale = getattr(_attn_mod, "scale", (q.shape[-1] ** -0.5))
#                     attn = (q @ k.transpose(-2, -1)) * scale
#                     attn = attn.softmax(dim=-1)

#                     _self.attn_maps.append(attn.detach())

           
#                 return _orig(x, **kwargs)

            
#             attn_mod.forward = wrapped_forward
#             self._patched.append(_PatchedAttention(attn_mod, orig_forward))

#     def clear(self) -> None:
#         self.attn_maps = []

#     @torch.no_grad()
#     def forward_and_capture(self, img_tensor: torch.Tensor) -> List[torch.Tensor]:
       
#         self.clear()

#         if img_tensor.ndim == 3:
#             x = img_tensor.unsqueeze(0)
#         elif img_tensor.ndim == 4:
#             x = img_tensor
#         else:
#             raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(img_tensor.shape)}")

#         x = x.to(self.device)
#         _ = self.wrapper(x)  
#         return self.attn_maps

#     def compute_rollout(self, discard_ratio: float = 0.0, head_fusion: str = "mean") -> torch.Tensor:
#         if not getattr(self, "supported", False):
#             raise RuntimeError("ViT attention capture not supported for this backbone (no .blocks found).")
#         if not self.attn_maps:
#             raise RuntimeError("No attention maps available. Call forward_and_capture first.")

       
#         attn_stack = self.attn_maps

       
#         fused_layers = []
#         for a in attn_stack:
#             if head_fusion == "mean":
#                 fused = a.mean(dim=1)
#             elif head_fusion == "max":
#                 fused = a.max(dim=1).values
#             else:
#                 raise ValueError("head_fusion must be 'mean' or 'max'")
#             fused_layers.append(fused)

       
#         if discard_ratio > 0:
#             for i, A in enumerate(fused_layers):
#                 B, N, _ = A.shape
#                 A2 = A.clone()
               
#                 flat = A2.view(B, -1)
#                 _, idx = flat.sort(dim=-1)
#                 num_discard = int(flat.shape[-1] * discard_ratio)
#                 if num_discard > 0:
#                     discard_idx = idx[:, :num_discard]
#                     flat.scatter_(1, discard_idx, 0.0)
#                 fused_layers[i] = flat.view(B, N, N)

       
#         result = torch.eye(fused_layers[0].shape[-1], device=fused_layers[0].device).unsqueeze(0)
#         for A in fused_layers:
#             A = A + torch.eye(A.shape[-1], device=A.device).unsqueeze(0)
#             A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
#             result = A @ result

        
#         return result[0]


# # -------------------------
# # Helpers
# # -------------------------
# def cls_attention_to_grid(cls_to_patches: torch.Tensor) -> np.ndarray:
#     """
#     cls_to_patches: (num_patches,) tensor
#     returns square grid (sqrt(patches) x sqrt(patches))
#     """
#     if isinstance(cls_to_patches, torch.Tensor):
#         v = cls_to_patches.detach().cpu().numpy()
#     else:
#         v = np.asarray(cls_to_patches)

#     n = v.shape[0]
#     s = int(np.sqrt(n))
#     if s * s != n:
#         raise ValueError(f"Number of patches ({n}) is not a perfect square.")
#     return v.reshape(s, s)


# def upsample_attention_to_image(grid: np.ndarray, image_size: int) -> np.ndarray:
#     """
#     grid: (gh, gw)
#     returns heatmap (image_size, image_size) float32
#     """
#     import cv2
#     heat = cv2.resize(grid.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_CUBIC)
#     return heat


# dr_app/utils/vit_attention.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


# -------------------------
# Helpers to build heatmap
# -------------------------
def cls_attention_to_grid(cls_to_patches: np.ndarray) -> np.ndarray:
    """
    cls_to_patches: (num_patches,) where num_patches = grid*grid
    returns: (grid, grid)
    """
    n = cls_to_patches.shape[0]
    g = int(np.sqrt(n))
    if g * g != n:
        raise ValueError(f"Patch count {n} is not a perfect square.")
    return cls_to_patches.reshape(g, g)


def upsample_attention_to_image(grid: np.ndarray, image_size: int = 224) -> np.ndarray:
    """
    grid: (g, g) -> returns (image_size, image_size) float heatmap
    """
    import cv2

    grid = grid.astype(np.float32)
    grid = cv2.resize(grid, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

    # normalize 0..1
    mn, mx = float(grid.min()), float(grid.max())
    if mx > mn:
        grid = (grid - mn) / (mx - mn + 1e-8)
    else:
        grid = np.zeros_like(grid)
    return grid


# -------------------------
# Attention wrapper
# -------------------------
class _WrappedMHA(nn.Module):
    """
    Wrap torch.nn.MultiheadAttention so it ALWAYS returns attention weights
    and stores them for rollout, while keeping the same (out, weights) tuple.
    """
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, *args, **kwargs):
        # Force attention weights
        kwargs["need_weights"] = True

        # Prefer per-head weights when available
        if "average_attn_weights" in self.mha.forward.__code__.co_varnames:
            kwargs["average_attn_weights"] = False

        out, attn = self.mha(*args, **kwargs)
        self.last_attn = attn  # shape: (B, heads, T, T) or (B, T, T)
        return out, attn


# -------------------------
# Main extractor
# -------------------------
@dataclass
class ViTAttentionExtractor:
    model: nn.Module
    device: torch.device

    def __post_init__(self):
        # Your model is ViTDR, the torchvision ViT is at model.backbone
        self.vit = self.model.backbone if hasattr(self.model, "backbone") else self.model
        self.attn_maps: List[torch.Tensor] = []
        self.supported: bool = False
        self._patch_torchvision_vit()

    def _patch_torchvision_vit(self):
        """
        Torchvision VisionTransformer has:
          vit.encoder.layers  (nn.Sequential of EncoderBlock)
          each EncoderBlock has .self_attention (nn.MultiheadAttention)
        """
        encoder = getattr(self.vit, "encoder", None)
        if encoder is None:
            self.supported = False
            return

        layers = getattr(encoder, "layers", None)
        if layers is None:
            self.supported = False
            return

        # layers is nn.Sequential of EncoderBlock
        if not isinstance(layers, nn.Sequential):
            self.supported = False
            return

        patched_any = False
        for blk in layers:
            if hasattr(blk, "self_attention") and isinstance(blk.self_attention, nn.MultiheadAttention):
                # Only wrap once
                if not isinstance(blk.self_attention, _WrappedMHA):
                    blk.self_attention = _WrappedMHA(blk.self_attention)
                patched_any = True

        self.supported = patched_any

    @torch.no_grad()
    def forward_and_capture(self, x_chw: torch.Tensor):
        """
        x_chw: (C,H,W) single image tensor
        runs model forward to populate wrapper.last_attn per layer
        """
        self.attn_maps.clear()

        if not self.supported:
            return

        x = x_chw.unsqueeze(0).to(self.device)  # (1,C,H,W)

        # forward pass
        _ = self.model(x)

        # collect attentions from wrapped blocks
        layers = self.vit.encoder.layers
        for blk in layers:
            mha = getattr(blk, "self_attention", None)
            if isinstance(mha, _WrappedMHA) and mha.last_attn is not None:
                self.attn_maps.append(mha.last_attn.detach().to("cpu"))

    def compute_rollout(self, discard_ratio: float = 0.0) -> np.ndarray:
        """
        Returns (T, T) rollout matrix, where T = 1 + num_patches (includes CLS token)
        """
        if not self.supported:
            raise RuntimeError("ViT attention not supported for this backbone.")
        if len(self.attn_maps) == 0:
            raise RuntimeError("No attention maps available. Call forward_and_capture first.")

        # stack: list of (B, heads, T, T) or (B, T, T)
        attn = []
        for a in self.attn_maps:
            # keep only batch 0
            if a.dim() == 4:
                # (B, heads, T, T) -> average heads -> (T, T)
                a0 = a[0].mean(dim=0)
            elif a.dim() == 3:
                # (B, T, T) -> (T, T)
                a0 = a[0]
            else:
                continue
            attn.append(a0)

        if len(attn) == 0:
            raise RuntimeError("Attention maps captured but shapes were unexpected.")

        # Attention rollout
        # Add identity and row-normalize
        result = None
        for A in attn:
            T = A.shape[-1]
            A = A.clone()

            if discard_ratio > 0:
                flat = A.view(-1)
                k = int(flat.numel() * discard_ratio)
                if k > 0:
                    _, idx = torch.topk(flat, k, largest=False)
                    flat[idx] = 0
                    A = flat.view(T, T)

            I = torch.eye(T)
            A = A + I
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

            result = A if result is None else A @ result

        return result.cpu().numpy()
