import torch
import torch.nn as nn
import numpy as np

class FFTBranch(nn.Module):
    """
    Frequency domain branch for deepfake detection.
    Deepfakes often leave artifacts in the frequency domain
    that are invisible to the spatial (CNN) branch.
    """
    def __init__(self, img_size=224, out_features=256):
        super().__init__()
        self.img_size = img_size

        # Learnable layers to process FFT magnitude spectrum
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x: (B, 3, H, W) — convert to grayscale for FFT
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = gray.unsqueeze(1)  # (B, 1, H, W)

        # 2D FFT — extract magnitude spectrum
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)

        # Log scale to compress dynamic range
        magnitude = torch.log1p(magnitude)

        # Normalize per sample
        b = magnitude.shape[0]
        mag_min = magnitude.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        mag_max = magnitude.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)

        # Pass through CNN
        out = self.cnn(magnitude)
        out = self.fc(out)
        return out  # (B, out_features)
