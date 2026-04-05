#!/usr/bin/env python3
"""Quick debug: test model on specific Kaggle images"""

import sys
sys.path.insert(0, '.')

import torch
import os
from PIL import Image
from data.augmentations import get_val_transforms
from models.hybrid_model import HybridDeepfakeDetector

# Load model
print("Loading model...")
model = HybridDeepfakeDetector(num_classes=1, pretrained=False)
model.load_state_dict(torch.load('backend/models/hybrid_full_best.pt', map_location='cpu'))
model.eval()

transform = get_val_transforms(224)
dataset_path = r"C:\Users\Aafi\Desktop\Dataset\Test"

print("\n" + "="*60)
print("REAL IMAGES TEST")
print("="*60)
real_path = os.path.join(dataset_path, "Real")
for img_file in sorted(os.listdir(real_path))[:5]:
    img = Image.open(os.path.join(real_path, img_file)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        conf = torch.sigmoid(output).item()
    pred = "DEEPFAKE" if conf > 0.5 else "REAL"
    print(f"{img_file}: confidence={conf:.4f}, predicted={pred}, TRUE=REAL")

print("\n" + "="*60)
print("FAKE IMAGES TEST")
print("="*60)
fake_path = os.path.join(dataset_path, "Fake")
for img_file in sorted(os.listdir(fake_path))[:5]:
    img = Image.open(os.path.join(fake_path, img_file)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        conf = torch.sigmoid(output).item()
    pred = "DEEPFAKE" if conf > 0.5 else "REAL"
    print(f"{img_file}: confidence={conf:.4f}, predicted={pred}, TRUE=DEEPFAKE")
