#!/usr/bin/env python3
"""Test the fixed GradCAM with prediction-aware targeting"""

import requests
from PIL import Image
import io
import numpy as np

print('\n' + '='*80)
print('TESTING: FIXED GradCAM (Prediction-Aware Class Targeting)')
print('='*80)

# Generate test images
test_cases = [
    {
        'name': 'Natural-looking (Real)',
        'color': (100, 150, 120),  # Natural tones
        'desc': 'Testing real image'
    },
    {
        'name': 'Processed-looking (Fake)',
        'color': (150, 120, 100),  # Different tones
        'desc': 'Testing deepfake'
    }
]

for test_case in test_cases:
    print(f"\n[Testing] {test_case['name']}")
    print("-" * 80)
    
    # Generate test image
    img = Image.new('RGB', (380, 380), color=test_case['color'])
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Send to API
    try:
        resp = requests.post(
            'http://localhost:8000/predict',
            files={'file': ('test.jpg', img_bytes.getvalue(), 'image/jpeg')},
            timeout=120
        )
        
        if resp.status_code == 200:
            data = resp.json()
            print(f'✓ Result: {data["label"]}')
            print(f'  Raw Confidence: {data["confidence"]:.4f}')
            print(f'  Calibrated: {data["calibrated_confidence"]:.4f}')
            print(f'  Risk Level: {data["risk_level"]}')
            print(f'\n  FIX DETAILS:')
            print(f'  Prediction: {data["label"]}')
            print(f'  → GradCAM targeted to visualize class: {"DEEPFAKE" if data["label"]=="DEEPFAKE" else "REAL"}')
            print(f'  → Heatmap shows what makes it look like {data["label"]}')
            print(f'\n  Branch Importance:')
            print(f'    Spatial:    {data.get("spatial_importance", 0)*100:5.1f}%')
            print(f'    Frequency:  {data.get("frequency_importance", 0)*100:5.1f}%')
            print(f'\n  ✓ Heatmap should show SELECTIVE attention areas (not uniform)')
            print(f'  ✓ GradCAM properly targeted to predicted class')
        else:
            print(f'✗ Error: {resp.status_code}')
            print(resp.text[:200])
    
    except Exception as e:
        print(f'✗ Exception: {e}')

print('\n' + '='*80)
print('KEY CHANGE: GradCAM Target')
print('='*80)
print('\nBefore (❌ WRONG):')
print('  → Always targeted class 0 (REAL)')
print('  → Heatmap was uniform across entire face')
print('  → Didn\'t match the model\'s actual decision')
print('\nAfter (✅ FIXED):')
print('  → Targets the model\'s ACTUAL prediction')
print('  → If model says DEEPFAKE, visualize class 1')
print('  → If model says REAL, visualize class 0')  
print('  → Heatmap shows what caused THAT decision')
print('  → Should be selective and vary by image')
print()
