import torch
import torch.nn as nn
try:
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    USE_EFFICIENTNET = True
except ImportError:
    # Fallback: use resnet50 if efficientnet not available
    from torchvision.models import resnet50
    efficientnet_b4 = resnet50
    EfficientNet_B4_Weights = None
    USE_EFFICIENTNET = False

from models.fft_branch import FFTBranch

class HybridDeepfakeDetector(nn.Module):
    """
    Hybrid model: EfficientNet-B4 spatial branch + FFT frequency branch.
    Fuses both embeddings for final binary classification.
    """
    def __init__(self, num_classes=1, dropout=0.4, pretrained=True):
        super().__init__()

        # Spatial branch — EfficientNet-B4 (or resnet50 fallback)
        # Handle different APIs: efficientnet uses weights=, resnet uses pretrained=
        if USE_EFFICIENTNET:
            weights = None
            if pretrained and EfficientNet_B4_Weights is not None:
                weights = EfficientNet_B4_Weights.IMAGENET1K_V1
            base = efficientnet_b4(weights=weights)
        else:
            # ResNet API: use pretrained parameter
            base = efficientnet_b4(pretrained=pretrained)
        
        # Get features from the model (works for both efficientnet and resnet)
        if hasattr(base, 'features'):
            self.spatial_features = base.features
            self.spatial_pool = base.avgpool
        else:
            # For resnet, use sequential layers
            self.spatial_features = torch.nn.Sequential(*list(base.children())[:-2])
            self.spatial_pool = torch.nn.AdaptiveAvgPool2d(1)
        
        # Get the output feature size
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            if hasattr(base, 'features'):
                dummy_out = base.features(dummy_input)
            else:
                dummy_out = self.spatial_features(dummy_input)
            spatial_out = dummy_out.shape[1]

        # Frequency branch — FFT
        fft_out = 256
        self.fft_branch = FFTBranch(out_features=fft_out)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(spatial_out + fft_out, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Spatial features
        spatial = self.spatial_features(x)
        spatial = self.spatial_pool(spatial)
        spatial = torch.flatten(spatial, 1)  # (B, 1792)

        # Frequency features
        freq = self.fft_branch(x)  # (B, 256)

        # Concatenate and classify
        fused = torch.cat([spatial, freq], dim=1)  # (B, 2048)
        out = self.classifier(fused)
        return out
