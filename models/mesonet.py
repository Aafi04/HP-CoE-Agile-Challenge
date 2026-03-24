import torch
import torch.nn as nn

class MesoNet4(nn.Module):
    """
    MesoNet-4: lightweight CNN baseline for deepfake detection.
    Original paper: Afchar et al., 2018.
    """
    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
        )
        # Calculate flat size: 224 -> 112 -> 56 -> 28 -> 7, channels=16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * 7 * 7, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
