import sys
sys.path.insert(0, '.')
import torch
from models.fft_branch import FFTBranch
from models.hybrid_model import HybridDeepfakeDetector

def test_fft_branch():
    model = FFTBranch(out_features=256)
    model.eval()
    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"FFT branch output shape: {out.shape}")
    assert out.shape == (4, 256)
    print("FFT branch test PASSED")

def test_hybrid():
    model = HybridDeepfakeDetector(num_classes=1, dropout=0.4, pretrained=True)
    model.eval()
    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"Hybrid model output shape: {out.shape}")
    assert out.shape == (4, 1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print("Hybrid model test PASSED")

if __name__ == '__main__':
    test_fft_branch()
    test_hybrid()
