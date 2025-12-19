# models/vit_dr.py
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTDR(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ViTDR, self).__init__()
        weights = None
        self.backbone = vit_b_16(weights=weights)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
       
        in_features = self.backbone.heads[0].in_features
        self.backbone.heads = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(" ViT Backbone unfrozen - all layers are now trainable")