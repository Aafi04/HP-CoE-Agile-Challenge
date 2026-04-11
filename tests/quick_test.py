#!/usr/bin/env python3
"""
Simple API test - test a few images and show confidence calibration
"""
import requests
import os

BASE_URL = "http://127.0.0.1:8000"

test_images = [
    (r"c:\Users\Aafi\Desktop\Dataset\Test\Fake\fake_0.jpg", "Fake"),
    (r"c:\Users\Aafi\Desktop\Dataset\Test\Real\real_0.jpg", "Real"),
]

for image_path, expected in test_images:
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        resp = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    
    result = resp.json()
    label = result['label']
    conf = result['confidence']
    is_correct = (label == "DEEPFAKE" and expected == "Fake") or (label == "REAL" and expected == "Real")
    
    symbol = "✓" if is_correct else "✗"
    print(f"{symbol} {os.path.basename(image_path):15} | Expected: {expected:4} | Got: {label:9} | Conf: {conf:.4f}")
