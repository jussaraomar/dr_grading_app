# utils/grad_cam.py
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, device, target_layer_name):
        self.model = model
        self.device = device
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        target_layer = self._find_target_layer()
        if target_layer is None:
            raise ValueError(f"Target layer '{self.target_layer_name}' not found in model")
        
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)

    
    def _find_target_layer(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module
        return None

   
    def _create_retinal_mask_from_image(self, original_image):

        original_np = np.array(original_image)       
        gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
        mask = gray > 20
        return mask

    def generate_cam(self, image_tensor, class_idx=None):
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        image_tensor = image_tensor.to(self.device).unsqueeze(0)
        image_tensor.requires_grad_()
        
        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)
        
        if self.gradients is None:
            raise ValueError("Gradients are None - backward hook not triggered")
        if self.activations is None:
            raise ValueError("Activations are None - forward hook not triggered")
        
        gradients = self.gradients.cpu().detach().numpy()[0]  
        activations = self.activations.cpu().detach().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2)) 
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        return cam, class_idx
    
    def visualize_cam(
        self,
        image_tensor,
        original_image,
        true_class=None,
        class_idx=None,
        save_path=None,
        apply_threshold=True,
        threshold_quantile=0.8,
        mask_outside_retina=True, 
    ):
        try:
            cam, pred_class = self.generate_cam(image_tensor, class_idx)
        except Exception as e:
            print(f" Grad-CAM generation failed: {e}")
            return None
        
        # fresh forward for confidence
        with torch.no_grad():
            image_tensor_eval = image_tensor.to(self.device).unsqueeze(0)
            output = self.model(image_tensor_eval)
            pred_class = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        
       
        cam_display = cam.copy()

        
        if apply_threshold:
            q = np.quantile(cam_display, threshold_quantile)
            cam_display[cam_display < q] = 0.0

        orig_size = original_image.size 
        cam_resized = cv2.resize(cam_display, orig_size) 

       
        if mask_outside_retina:
            mask = self._create_retinal_mask_from_image(original_image)
            cam_resized_masked = np.zeros_like(cam_resized)
            cam_resized_masked[mask] = cam_resized[mask]
        else:
            cam_resized_masked = cam_resized

        # heatmap & overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized_masked), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        original_np = np.array(original_image)
        alpha = 0.35
        overlay = cv2.addWeighted(original_np, alpha, heatmap, 1 - alpha, 0)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(original_np)
        if true_class is not None:
            axes[0].set_title(f'Original Image\nTrue Class: {true_class}', fontsize=12)
        else:
            axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized_masked, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        if true_class is not None:
            title = f'Grad-CAM Overlay\nTrue: {true_class}, Pred: {pred_class}'
        else:
            title = f'Grad-CAM Overlay\nPred: {pred_class}'
        axes[2].set_title(title, fontsize=12)
        axes[2].axis('off')
        
        im = axes[3].imshow(cam_resized_masked, cmap='jet')
        axes[3].set_title(f'Heatmap Intensity\nConf: {confidence:.3f}', fontsize=12)
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved Grad-CAM visualization to {save_path}")
        plt.close(fig)
        
        return {
            'cam': cam,
            'overlay': overlay,
            'pred_class': pred_class,
            'true_class': true_class,
            'confidence': confidence
        }
    
    def __del__(self):
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()


class ResNetGradCAM(GradCAM):
    def __init__(self, model, device):
        super().__init__(model, device, target_layer_name='backbone.layer3')

    def _find_target_layer(self):
        # prefer last conv in layer3
        layer3_convs = []
        for name, module in self.model.named_modules():
            if name.startswith('backbone.layer3') and isinstance(module, nn.Conv2d):
                layer3_convs.append((name, module))
        
        if layer3_convs:
            name, module = layer3_convs[-1]
            print(f" Found ResNet target layer (higher-res): {name}")
            return module
        
        # fallbackss
        possible_layers = [
            'backbone.layer4.2.conv3',
            'backbone.layer4.2.conv2',
            'backbone.layer4.2.conv1',
            'backbone.layer4.1.conv3',
            'backbone.layer4.0.conv3',
        ]
        for layer_name in possible_layers:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    print(f" Found ResNet target layer (fallback): {name}")
                    return module
        
        raise ValueError("Could not find suitable layer for ResNet Grad-CAM")


class EfficientNetGradCAM(GradCAM):
    def __init__(self, model, device):
        super().__init__(model, device, target_layer_name='backbone.features.7.1.block.3.0')

    def _find_target_layer(self):
        possible_layers = [
            'backbone.features.7.1.block.3.0',
            'backbone.features.7.0.block.3.0',
            'backbone.features.8.0',
        ]
        for layer_name in possible_layers:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    print(f" Found EfficientNet target layer: {name}")
                    return module
        
        raise ValueError("Could not find suitable layer for EfficientNet Grad-CAM")
