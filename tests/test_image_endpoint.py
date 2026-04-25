#!/usr/bin/env python3
import io
import base64
import requests
from PIL import Image

# Create a test image
print("Creating test image...")
img = Image.new('RGB', (224, 224), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# Send to API
print('Testing image endpoint...')
files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
response = requests.post('http://127.0.0.1:8000/predict', files=files, timeout=30)

if response.status_code == 200:
    print('✅ Image endpoint works!')
    data = response.json()

    required_fields = [
        "is_fake",
        "confidence",
        "calibrated_confidence",
        "risk_level",
        "label",
        "heatmap_base64",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"

    assert isinstance(data["is_fake"], bool), "is_fake must be boolean"
    assert isinstance(data["confidence"], (int, float)), "confidence must be numeric"
    assert isinstance(data["calibrated_confidence"], (int, float)), "calibrated_confidence must be numeric"
    assert isinstance(data["risk_level"], str), "risk_level must be string"
    assert isinstance(data["label"], str), "label must be string"

    assert 0.0 <= data["confidence"] <= 1.0, "confidence out of bounds"
    assert 0.0 <= data["calibrated_confidence"] <= 1.0, "calibrated_confidence out of bounds"
    assert data["risk_level"] in {"UNCERTAIN", "BORDERLINE", "CONFIDENT"}, "Invalid risk_level"

    heatmap_value = data.get("heatmap_base64")
    assert heatmap_value is not None, "heatmap_base64 should not be None"
    base64.b64decode(heatmap_value, validate=True)

    print(f'  - is_fake: {data["is_fake"]}')
    print(f'  - confidence: {data["confidence"]}')
    print(f'  - calibrated_confidence: {data["calibrated_confidence"]}')
    print(f'  - risk_level: {data["risk_level"]}')
    print(f'  - label: {data["label"]}')
    print(f'  - has heatmap: {bool(data.get("heatmap_base64"))}')
else:
    print(f'❌ Error: {response.status_code}')
    print(response.text)
