#!/usr/bin/env python3
"""
Kaggle Dataset Diagnostics Script
==================================

Tests the deepfake detection model on real Kaggle images to:
1. Measure accuracy on real-world data
2. Identify failure patterns
3. Compare confidence distributions
4. Visualize which images are misclassified

Run with: python tests/test_kaggle_diagnostics.py
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from data.augmentations import get_val_transforms
from evaluation.gradcam import load_model


class KaggleTestDiagnostics:
    def __init__(self, dataset_path=r"C:\Users\Aafi\Desktop\Dataset", 
                 model_path="backend/models/hybrid_full_best.pt",
                 device="cpu"):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.device = device
        self.transform = get_val_transforms(224)
        
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path, 'hybrid', device)
        print(f"Model loaded on {device}")
        
        self.results = defaultdict(list)
        self.misclassified = []
        
    def test_directory(self, split="Test", max_per_class=None):
        """Test all images in a split (Train/Validation/Test)"""
        print(f"\n{'='*60}")
        print(f"Testing on {split} Set")
        print(f"{'='*60}\n")
        
        split_path = os.path.join(self.dataset_path, split)
        if not os.path.exists(split_path):
            print(f"❌ Split path not found: {split_path}")
            return
        
        total_tested = 0
        total_correct = 0
        
        for class_name in ["Real", "Fake"]:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️  Class directory not found: {class_path}")
                continue
            
            class_results = []
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if max_per_class:
                image_files = image_files[:max_per_class]
            
            print(f"\n📁 Testing {class_name} images ({len(image_files)} files)...")
            
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    with torch.no_grad():
                        output = self.model(img_tensor)
                        confidence = torch.sigmoid(output).item()
                    
                    # Determine label
                    predicted_label = "DEEPFAKE" if confidence > 0.5 else "REAL"
                    true_label = "DEEPFAKE" if class_name == "Fake" else "REAL"
                    is_correct = predicted_label == true_label
                    
                    # Store result
                    result = {
                        "file": img_file,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "is_correct": is_correct
                    }
                    
                    class_results.append(result)
                    self.results[class_name].append(result)
                    total_tested += 1
                    
                    if is_correct:
                        total_correct += 1
                    else:
                        self.misclassified.append({
                            "file": img_file,
                            "class": class_name,
                            "path": img_path,
                            "true": true_label,
                            "predicted": predicted_label,
                            "confidence": confidence
                        })
                    
                    # Progress indicator
                    if (i + 1) % 10 == 0:
                        print(f"   [{'='*30}] {i+1}/{len(image_files)}", end='\r')
                
                except Exception as e:
                    print(f"   ❌ Error processing {img_file}: {str(e)}")
                    continue
            
            # Summary for this class
            correct = sum(1 for r in class_results if r["is_correct"])
            accuracy = correct / len(class_results) if class_results else 0
            avg_confidence = np.mean([r["confidence"] for r in class_results]) if class_results else 0
            
            print(f"\n   ✓ {class_name}: {correct}/{len(class_results)} correct ({accuracy:.1%})")
            print(f"   ✓ Avg Confidence: {avg_confidence:.4f}")
        
        # Overall summary
        print(f"\n{'='*60}")
        print(f"OVERALL RESULTS FOR {split.upper()}")
        print(f"{'='*60}")
        print(f"Total Tested: {total_tested}")
        print(f"Total Correct: {total_correct}")
        overall_accuracy = total_correct / total_tested if total_tested > 0 else 0
        print(f"Overall Accuracy: {total_correct}/{total_tested} ({overall_accuracy:.1%})")
        
        return overall_accuracy
    
    def print_misclassified_summary(self):
        """Print summary of misclassified images"""
        if not self.misclassified:
            print("\n✅ No misclassifications!")
            return
        
        print(f"\n{'='*60}")
        print(f"MISCLASSIFIED IMAGES ({len(self.misclassified)} total)")
        print(f"{'='*60}\n")
        
        # Group by type
        false_positives = [m for m in self.misclassified if m["true"] == "REAL"]
        false_negatives = [m for m in self.misclassified if m["true"] == "DEEPFAKE"]
        
        if false_positives:
            print(f"❌ FALSE POSITIVES (Real labeled as Deepfake): {len(false_positives)}")
            for m in false_positives[:5]:  # Show first 5
                print(f"   • {m['file']}: confidence={m['confidence']:.4f}")
        
        if false_negatives:
            print(f"\n❌ FALSE NEGATIVES (Deepfake labeled as Real): {len(false_negatives)}")
            for m in false_negatives[:5]:  # Show first 5
                print(f"   • {m['file']}: confidence={m['confidence']:.4f}")
    
    def analyze_confidence_distribution(self):
        """Analyze confidence distribution by class"""
        print(f"\n{'='*60}")
        print(f"CONFIDENCE DISTRIBUTION ANALYSIS")
        print(f"{'='*60}\n")
        
        for class_name, results in self.results.items():
            if not results:
                continue
            
            confidences = [r["confidence"] for r in results]
            correct_confs = [r["confidence"] for r in results if r["is_correct"]]
            wrong_confs = [r["confidence"] for r in results if not r["is_correct"]]
            
            print(f"{class_name} Class ({len(results)} images):")
            print(f"  Mean Confidence: {np.mean(confidences):.4f}")
            print(f"  Std Dev: {np.std(confidences):.4f}")
            print(f"  Min: {np.min(confidences):.4f}, Max: {np.max(confidences):.4f}")
            
            if correct_confs:
                print(f"  Correct Predictions Avg Conf: {np.mean(correct_confs):.4f}")
            if wrong_confs:
                print(f"  Wrong Predictions Avg Conf: {np.mean(wrong_confs):.4f}")
            print()
    
    def save_results(self, output_file="test_kaggle_results.json"):
        """Save detailed results to JSON file"""
        results_data = {
            "date": str(Path(__file__).stat().st_mtime),
            "model_path": self.model_path,
            "device": self.device,
            "total_tested": sum(len(v) for v in self.results.values()),
            "results_by_class": dict(self.results),
            "misclassified": self.misclassified
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")


def main():
    # Initialize diagnostics
    diagnostics = KaggleTestDiagnostics()
    
    # Test on all splits (limit to 50 per class for faster diagnosis)
    accuracies = {}
    accuracies["Test"] = diagnostics.test_directory("Test", max_per_class=50)
    accuracies["Validation"] = diagnostics.test_directory("Validation", max_per_class=50)
    
    # Print detailed analysis
    diagnostics.analyze_confidence_distribution()
    diagnostics.print_misclassified_summary()
    
    # Save results
    diagnostics.save_results()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for split, accuracy in accuracies.items():
        if accuracy is not None:
            print(f"{split} Set Accuracy: {accuracy:.1%}")
    
    print("\n💡 If accuracy is still ~50%, check:")
    print("  1. Image preprocessing (aspect ratio preserved?)")
    print("  2. Model inference path (right model loaded?)")
    print("  3. Confidence threshold (is it too high/low?)")
    print("  4. Domain shift (Kaggle images very different from FF++?)")


if __name__ == "__main__":
    main()
