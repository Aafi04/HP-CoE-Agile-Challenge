#!/usr/bin/env python3
"""
Comprehensive Random Dataset Test
Tests the API with random images from the complete Dataset folder
"""
import requests
import os
import random
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def get_random_images(count=10):
    """Get random images from Dataset Test folders"""
    fake_dir = r"c:\Users\Aafi\Desktop\Dataset\Test\Fake"
    real_dir = r"c:\Users\Aafi\Desktop\Dataset\Test\Real"
    
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))]
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
    
    # Get random samples (about half of total count from each)
    num_fake = count // 2
    num_real = count - num_fake
    
    selected_fake = random.sample(fake_images, min(num_fake, len(fake_images)))
    selected_real = random.sample(real_images, min(num_real, len(real_images)))
    
    return selected_fake + selected_real

def test_image(image_path, expected_label):
    """Test single image through API"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            label = result['label']
            confidence = result['confidence']
            
            # Check if correct
            is_correct = (label == "DEEPFAKE" and expected_label == "Fake") or \
                        (label == "REAL" and expected_label == "Real")
            
            return {
                'filename': os.path.basename(image_path),
                'expected': expected_label,
                'predicted': "Fake" if label == "DEEPFAKE" else "Real",
                'label': label,
                'confidence': confidence,
                'correct': is_correct
            }
        else:
            return None
    except Exception as e:
        return None

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE RANDOM DATASET TEST")
    print("="*80)
    print(f"Base URL: {BASE_URL}\n")
    
    # Test health first
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API Server not responding to health check")
            return 1
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Start server with: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000")
        return 1
    
    print("✅ API Server responding\n")
    
    # Get random images
    print("Selecting random test images...")
    test_images = get_random_images(count=20)
    random.shuffle(test_images)
    
    print(f"Testing {len(test_images)} random images...\n")
    print("="*80)
    
    results = []
    correct_count = 0
    
    for i, image_path in enumerate(test_images, 1):
        expected = "Fake" if "Fake" in image_path else "Real"
        result = test_image(image_path, expected)
        
        if result:
            results.append(result)
            symbol = "✓" if result['correct'] else "✗"
            
            if result['correct']:
                correct_count += 1
            
            # Print with proper spacing
            print(f"{symbol} {i:2d}. {result['filename']:30} | "
                  f"Expected: {result['expected']:4} | Got: {result['predicted']:4} | "
                  f"Label: {result['label']:9} | Conf: {result['confidence']:6.4f}")
        else:
            print(f"✗ {i:2d}. {os.path.basename(image_path):30} | ERROR PROCESSING")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if results:
        accuracy = (correct_count / len(results)) * 100
        print(f"Total Tests: {len(results)}")
        print(f"Correct: {correct_count}")
        print(f"Incorrect: {len(results) - correct_count}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        # Confidence analysis
        fake_results = [r for r in results if r['expected'] == "Fake"]
        real_results = [r for r in results if r['expected'] == "Real"]
        
        if fake_results:
            fake_confs = [r['confidence'] for r in fake_results]
            print(f"\nFake Image Confidence:")
            print(f"  - Average: {sum(fake_confs)/len(fake_confs):.4f}")
            print(f"  - Range: {min(fake_confs):.4f} - {max(fake_confs):.4f}")
            print(f"  - Accuracy: {sum(1 for r in fake_results if r['correct'])/len(fake_results)*100:.1f}%")
        
        if real_results:
            real_confs = [r['confidence'] for r in real_results]
            print(f"\nReal Image Confidence:")
            print(f"  - Average: {sum(real_confs)/len(real_confs):.4f}")
            print(f"  - Range: {min(real_confs):.4f} - {max(real_confs):.4f}")
            print(f"  - Accuracy: {sum(1 for r in real_results if r['correct'])/len(real_results)*100:.1f}%")
        
        print("\n" + "="*80)
        if accuracy >= 95:
            print("✅ EXCELLENT - API MODEL PERFORMING WELL")
        elif accuracy >= 80:
            print("✓ GOOD - API MODEL ACCEPTABLE")
        else:
            print("⚠️  OK - API MODEL NEEDS MONITORING")
        print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
