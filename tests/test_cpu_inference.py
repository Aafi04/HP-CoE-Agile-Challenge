#!/usr/bin/env python3
"""
Quick CPU Inference Test - Verify project works on CPU
"""
import sys
import os
sys.path.insert(0, '.')

import torch
from models.hybrid_model import HybridDeepfakeDetector

def main():
    print("\n" + "="*60)
    print("CPU INFERENCE TEST")
    print("="*60)
    
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device.upper()}")
    
    # Test models
    models = [
        "backend/models/hybrid_kaggle_finetuned.pt",
        "backend/models/hybrid_full_best.pt",
    ]
    
    for model_path in models:
        if not os.path.exists(model_path):
            print(f"\n⊘ Not found: {model_path}")
            continue
        
        print(f"\n{'─'*60}")
        print(f"Testing: {os.path.basename(model_path)}")
        print(f"{'─'*60}")
        
        try:
            # Load model
            model = HybridDeepfakeDetector()
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            print(f"✓ Model loaded on {device.upper()}")
            
            # Test inference
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"✓ Inference successful")
            print(f"  Output shape: {output.shape}")
            print(f"  Output values: {output}")
            
            # Test on CPU if GPU was used
            if device == "cuda":
                print(f"\n✓ GPU inference works. Now testing CPU...")
                model.to("cpu")
                dummy_input = dummy_input.to("cpu")
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"✓ CPU inference also works!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    print(f"\n{'='*60}")
    print("✓ PROJECT READY FOR LOCAL CPU DEMO")
    print(f"{'='*60}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
