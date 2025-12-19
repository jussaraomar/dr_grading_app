# models/efficientnet_dr.py
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class EfficientNetDR(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(EfficientNetDR, self).__init__()

        weights = None  # because you load your own .pth
        self.backbone = efficientnet_b2(weights=weights)

        # Freeze all layers initially
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
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