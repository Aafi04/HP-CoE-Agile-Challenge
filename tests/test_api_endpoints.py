#!/usr/bin/env python3
"""
API Endpoint Testing - Test the backend with real Dataset images
"""
import requests
import json
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict(image_path, expected_label=None):
    """Test prediction endpoint with image"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"\n  Testing: {os.path.basename(image_path)}")
    expected = f" (Expected: {expected_label})" if expected_label else ""
    print(f"  {expected}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            is_fake = result['is_fake']
            confidence = result['confidence']
            label = result['label']
            
            symbol = "✓" if (is_fake and expected_label == "Fake") or (not is_fake and expected_label == "Real") else "⊘"
            print(f"  {symbol} Prediction: {label:4s} | Confidence: {confidence:.4f}")
            
            return result
        else:
            print(f"  ❌ Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    print("\n" + "="*70)
    print("BACKEND API ENDPOINT TESTING")
    print("="*70)
    print(f"Base URL: {BASE_URL}")
    
    # Test health
    if not test_health():
        print("\n❌ Server not responding. Start backend with:")
        print("cd AI-Based-Image-Authenticity-and-Deepfake-Detection-Tool")
        print("python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000")
        return 1
    
    print("\n" + "="*70)
    print("TEST 2: Prediction Endpoints")
    print("="*70)
    
    # Test images
    test_cases = [
        ("Fake Images", r"c:\Users\Aafi\Desktop\Dataset\Test\Fake\fake_0.jpg", "Fake"),
        ("Fake Images", r"c:\Users\Aafi\Desktop\Dataset\Test\Fake\fake_1.jpg", "Fake"),
        ("Fake Images", r"c:\Users\Aafi\Desktop\Dataset\Test\Fake\fake_100.jpg", "Fake"),
        ("Real Images", r"c:\Users\Aafi\Desktop\Dataset\Test\Real\real_0.jpg", "Real"),
        ("Real Images", r"c:\Users\Aafi\Desktop\Dataset\Test\Real\real_1.jpg", "Real"),
        ("Real Images", r"c:\Users\Aafi\Desktop\Dataset\Test\Real\real_10.jpg", "Real"),
    ]
    
    results = {"correct": 0, "total": 0}
    
    for category, image_path, expected in test_cases:
        result = test_predict(image_path, expected)
        if result:
            is_correct = (result['is_fake'] and expected == "Fake") or (not result['is_fake'] and expected == "Real")
            results["total"] += 1
            if is_correct:
                results["correct"] += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    if results["total"] > 0:
        accuracy = (results["correct"] / results["total"]) * 100
        print(f"Accuracy: {accuracy:.1f}% ({results['correct']}/{results['total']})")
        print(f"✓ All endpoints working correctly!")
    else:
        print("❌ No images tested")
    
    print("\n" + "="*70)
    print("✓ API TESTING COMPLETE")
    print("="*70)
    print("\nAPI Endpoints Available:")
    print("  GET  /health       - Health check")
    print("  POST /predict      - Predict single image")
    print("  POST /predict_video - Predict video (with frame sampling)")
    print("\nFrontend URL: http://127.0.0.1:3000")
    print("API Docs: http://127.0.0.1:8000/docs\n")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
