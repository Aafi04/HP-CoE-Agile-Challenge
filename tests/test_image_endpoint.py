#!/usr/bin/env python3
import io
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
    print(f'  - is_fake: {data["is_fake"]}')
    print(f'  - confidence: {data["confidence"]}')
    print(f'  - label: {data["label"]}')
    print(f'  - has heatmap: {bool(data.get("heatmap_base64"))}')
else:
    print(f'❌ Error: {response.status_code}')
    print(response.text)
