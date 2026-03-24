import sys
sys.path.insert(0, '.')
import torch
from models.mesonet import MesoNet4

def test_mesonet():
    model = MesoNet4(num_classes=1)
    model.eval()
    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")
    assert out.shape == (4, 1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print("MesoNet test PASSED")

if __name__ == '__main__':
    test_mesonet()
