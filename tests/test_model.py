import sys
sys.path.insert(0, '.')
import torch
from models.efficientnet import DeepfakeEfficientNet

def test_model():
    model = DeepfakeEfficientNet(num_classes=1, dropout=0.4, pretrained=True)
    model.eval()
    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")
    assert out.shape == (4, 1)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable:,}")
    print("Model test PASSED")

if __name__ == '__main__':
    test_model()
