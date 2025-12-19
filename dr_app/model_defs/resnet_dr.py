# models/resnet_dr.py
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

class ResNetDR(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNetDR, self).__init__()
        weights = None  # because you load your own .pth
        self.backbone = resnet50(weights=weights)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
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
        print(" Backbone unfrozen - all layers are now trainable")