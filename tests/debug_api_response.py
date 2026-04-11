#!/usr/bin/env python3
"""
Debug script to understand API prediction logic
"""
import requests
import json
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def test_single_image(image_path, expected_label):
    """Test single image and get detailed API response"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(image_path)} (Expected: {expected_label})")
    print(f"{'='*70}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("\n" + "="*70)
    print("API RESPONSE DEBUG")
    print("="*70)
    
    # Test specific images
    test_cases = [
        (r"c:\Users\Aafi\Desktop\Dataset\Test\Real\real_0.jpg", "Real"),
        (r"c:\Users\Aafi\Desktop\Dataset\Test\Fake\fake_0.jpg", "Fake"),
    ]
    
    for image_path, expected in test_cases:
        test_single_image(image_path, expected)

if __name__ == "__main__":
    main()
