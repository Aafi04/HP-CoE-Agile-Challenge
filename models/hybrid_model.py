import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from models.fft_branch import FFTBranch

class HybridDeepfakeDetector(nn.Module):
    """
    Hybrid model: EfficientNet-B4 spatial branch + FFT frequency branch.
    Fuses both embeddings for final binary classification.
    """
    def __init__(self, num_classes=1, dropout=0.4, pretrained=True):
        super().__init__()

        # Spatial branch — EfficientNet-B4
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b4(weights=weights)
        self.spatial_features = base.features
        self.spatial_pool = base.avgpool
        spatial_out = base.classifier[1].in_features  # 1792

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
