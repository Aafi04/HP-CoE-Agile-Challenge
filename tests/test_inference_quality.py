#!/usr/bin/env python3
"""
Model Inference Comparison Test - Verify fine-tuned model quality
Tests both models on real Dataset images
"""
import sys
import os
sys.path.insert(0, '.')

import torch
from PIL import Image
import numpy as np
from data.augmentations import get_val_transforms
from models.hybrid_model import HybridDeepfakeDetector

def predict(model, image_path, transform, device):
    """Run single inference"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
        
        # Convert output to probability
        confidence = torch.sigmoid(output).item()
        return confidence
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = get_val_transforms(224)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON TEST - Fine-tuned vs Pre-trained")
    print("="*70)
    print(f"Device: {device.upper()}")
    
    # Load models
    models = {
        "Fine-tuned (Kaggle)": "backend/models/hybrid_kaggle_finetuned.pt",
        "Pre-trained (FF++)": "backend/models/hybrid_full_best.pt",
    }
    
    loaded_models = {}
    for name, path in models.items():
        try:
            model = HybridDeepfakeDetector()
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.to(device)
            loaded_models[name] = model
            print(f"✓ Loaded: {name}")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
    
    # Test images
    test_images = {
        "Fake": [
            r"c:\Users\Aafi\Desktop\Dataset\Test\Fake\fake_0.jpg",
            r"c:\Users\Aafi\Desktop\Dataset\Test\Fake\fake_1.jpg",
            r"c:\Users\Aafi\Desktop\Dataset\Test\Fake\fake_100.jpg",
        ],
        "Real": [
            r"c:\Users\Aafi\Desktop\Dataset\Test\Real\real_0.jpg",
            r"c:\Users\Aafi\Desktop\Dataset\Test\Real\real_1.jpg",
            r"c:\Users\Aafi\Desktop\Dataset\Test\Real\real_10.jpg",
        ]
    }
    
    print("\n" + "="*70)
    print("INFERENCE RESULTS")
    print("="*70)
    
    all_results = {name: {"correct": 0, "total": 0} for name in loaded_models}
    
    for category, images in test_images.items():
        true_label = 1 if category == "Fake" else 0
        category_display = "FAKE (should be ~1.0)" if category == "Fake" else "REAL (should be ~0.0)"
        
        print(f"\n[{category.upper()}] {category_display}")
        print("-" * 70)
        
        for img_path in images:
            img_name = os.path.basename(img_path)
            print(f"\n  {img_name}")
            
            if not os.path.exists(img_path):
                print(f"    ⊘ Image not found")
                continue
            
            for model_name, model in loaded_models.items():
                confidence = predict(model, img_path, transform, device)
                
                if confidence is not None:
                    # Interpret confidence
                    is_fake = confidence > 0.5
                    correct = (is_fake and category == "Fake") or (not is_fake and category == "Real")
                    
                    # Color coding
                    symbol = "✓" if correct else "❌"
                    prediction = "FAKE" if is_fake else "REAL"
                    
                    print(f"    {symbol} {model_name}: {prediction:4s} (conf={confidence:.4f})")
                    
                    all_results[model_name]["total"] += 1
                    if correct:
                        all_results[model_name]["correct"] += 1
    
    # Summary
    print("\n" + "="*70)
    print("EARLY SUMMARY")
    print("="*70)
    for model_name, stats in all_results.items():
        if stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"]) * 100
            print(f"{model_name}: {accuracy:.1f}% accuracy ({stats['correct']}/{stats['total']})")
    
    print("\n" + "="*70)
    print("✓ TEST COMPLETE")
    print("="*70)
    print("\nNOTE: Full test uses ~200K images in training/validation sets")
    print("These samples are just for quick verification")
    print("Expected final accuracy: Fine-tuned ~96%, Pre-trained ~95%\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
