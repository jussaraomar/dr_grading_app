# src/inference.py
import os
from typing import cast, Optional
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import cv2

from huggingface_hub import hf_hub_download
from dr_app.utils.grad_cam import ResNetGradCAM, EfficientNetGradCAM
from dr_app.model_defs.resnet_dr import ResNetDR
from dr_app.model_defs.efficientnet_dr import EfficientNetDR
from dr_app.model_defs.vit_dr import ViTDR
from dr_app.utils.vit_attention import (
    ViTAttentionExtractor,
    cls_attention_to_grid,
    upsample_attention_to_image,
)

# -------------------------
# Settings
# -------------------------
NUM_CLASSES = 5
CNN_SIZE = (512, 512)     
VIT_SIZE = (224, 224)


# -------------------------
# Hugging Face weights
# -------------------------
HF_REPO_ID = "juuuu0/dr-grading-weights"

def _hf_download(filename: str) -> str:
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=os.getenv("HF_TOKEN"), 
    )



MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = os.path.dirname(os.path.dirname(__file__))  # dr_app/

RESNET_PATH = _hf_download("resnet.pth")
EFFNET_PATH = _hf_download("effnet.pth")
VIT_PATH    = _hf_download("vit.pth")


cnn_transform = transforms.Compose([
    transforms.Resize(CNN_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

vit_transform = transforms.Compose([
    transforms.Resize(VIT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# -------------------------
# Model builders
# -------------------------
def build_resnet_model():
    return ResNetDR(num_classes=NUM_CLASSES, pretrained=False)

def build_effnet_model():
    return EfficientNetDR(num_classes=NUM_CLASSES, pretrained=False)

def build_vit_model():
    return ViTDR(num_classes=NUM_CLASSES, pretrained=False)

# def _load_weights(model, path):
#     state = torch.load(path, map_location="cpu", weights_only=True)

#     if isinstance(state, dict) and "model_state_dict" in state:
#         state = state["model_state_dict"]
#     if isinstance(state, dict) and "state_dict" in state:
#         state = state["state_dict"]
#     model.load_state_dict(state, strict=True)
#     return model

def _load_weights(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model

# -------------------------
# global singletons
# -------------------------
_RESNET = None
_EFFNET = None
_VIT = None
_RESNET_CAM = None
_EFFNET_CAM = None
_VIT_ATTN = None


def crop_to_retina(pil_img: Image.Image, margin: int = 10) -> Image.Image:
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > 5
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return pil_img
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - margin); y0 = max(0, y0 - margin)
    x1 = min(img.shape[1], x1 + margin); y1 = min(img.shape[0], y1 + margin)
    return Image.fromarray(img[y0:y1, x0:x1])


def _init():
    global _RESNET, _EFFNET, _VIT, _RESNET_CAM, _EFFNET_CAM, _VIT_ATTN

    if _RESNET is not None and _EFFNET is not None and _VIT is not None:
        return

    _RESNET = _load_weights(build_resnet_model(), RESNET_PATH).to(DEVICE).eval()
    _EFFNET = _load_weights(build_effnet_model(), EFFNET_PATH).to(DEVICE).eval()

    # print("---- EfficientNet named_modules (sample) ----")
    # for name, module in _EFFNET.named_modules():
    #     if "features" in name and ("7" in name or "8" in name):
    #         print(name, type(module))
    # print("---------------------------------------------")


    _VIT    = _load_weights(build_vit_model(), VIT_PATH).to(DEVICE).eval()

    _RESNET_CAM = ResNetGradCAM(_RESNET, DEVICE)
    _EFFNET_CAM = EfficientNetGradCAM(_EFFNET, DEVICE)

    _VIT_ATTN = ViTAttentionExtractor(_VIT, DEVICE)

@torch.no_grad()
def _predict_probs(model, x_batched):
    logits = model(x_batched)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf, probs

def run_all_models(pil_img: Image.Image):
    _init()

    original = pil_img.convert("RGB")
    
    original = crop_to_retina(original)

    cnn_tensor = cast(torch.Tensor, cnn_transform(original))
    vit_tensor = cast(torch.Tensor, vit_transform(original))

    # preprocess
    x_cnn = cnn_tensor.unsqueeze(0).to(DEVICE)
    x_vit = vit_tensor.unsqueeze(0).to(DEVICE)

    # predictions
    r_pred, r_conf, _ = _predict_probs(_RESNET, x_cnn)
    e_pred, e_conf, _ = _predict_probs(_EFFNET, x_cnn)
    v_pred, v_conf, _ = _predict_probs(_VIT, x_vit)

    r_cam = _RESNET_CAM.visualize_cam(
        image_tensor=cnn_tensor,
        original_image=original,
        class_idx=r_pred,
        apply_threshold=True,
        threshold_quantile=0.6,
        mask_outside_retina=True,
    )
    e_cam = _EFFNET_CAM.visualize_cam(
        image_tensor=cnn_tensor,
        original_image=original,
        class_idx=e_pred,
        apply_threshold=True,
        threshold_quantile=0.8,
        mask_outside_retina=True,
    )

    res_overlay = Image.fromarray(r_cam["overlay"]) if r_cam else original
    eff_overlay = Image.fromarray(e_cam["overlay"]) if e_cam else original

   
    # attn_maps = _VIT_ATTN.forward_and_capture(vit_tensor)
    # rollout = _VIT_ATTN.compute_rollout(discard_ratio=0.0)
    # cls_to_patches = rollout[0, 1:]  
    # grid = cls_attention_to_grid(cls_to_patches)
    # heatmap = upsample_attention_to_image(grid, image_size=VIT_SIZE[0])
    # vit_overlay = _vit_attention_overlay(original, heatmap)

    # -------------------------
    # ViT attention rollout
    # -------------------------
    _VIT_ATTN.forward_and_capture(vit_tensor)

    # compute_rollout returns (tokens x tokens)
    try:
        rollout = _VIT_ATTN.compute_rollout(discard_ratio=0.0)
        cls_to_patches = rollout[0, 1:]  # CLS row, skip CLS token
        grid = cls_attention_to_grid(cls_to_patches)
        heatmap = upsample_attention_to_image(grid, image_size=VIT_SIZE[0])
        vit_overlay = _vit_attention_overlay(original, heatmap)
    except Exception as ex:
        # fallback: don’t crash the whole app
        vit_overlay = original


    df = pd.DataFrame([
        {"Model": "ResNet-50", "Predicted Grade": r_pred, "Confidence": round(r_conf, 4)},
        {"Model": "EfficientNet-B2", "Predicted Grade": e_pred, "Confidence": round(e_conf, 4)},
        {"Model": "ViT-B/16", "Predicted Grade": v_pred, "Confidence": round(v_conf, 4)},
    ])

    return df, res_overlay, eff_overlay, vit_overlay


#vit helper
def _vit_attention_overlay(original_image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    original_np = np.array(original_image)
    H, W, _ = original_np.shape

    heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)

    # retina mask
    gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
    mask = gray > 20
    hm = np.zeros_like(heatmap)
    hm[mask] = heatmap[mask]
    heatmap = hm

    # normalize
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_np, 0.5, heatmap_color, 0.5, 0)
    return Image.fromarray(overlay.astype(np.uint8))
