"""
GPU TRAINING SCRIPT FOR GOOGLE COLAB
======================================
Copy this entire script into a Colab cell and run
Requires: Dataset uploaded to Google Drive in /My Drive/Dataset/

Expected runtime: 1-2 hours on GPU (vs 14+ on CPU)
"""

# ============================================================
# CELL 1: Install & Setup (Run first)
# ============================================================

!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q pillow tqdm numpy

# Check GPU
import torch
print(f"✓ GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# ============================================================  
# CELL 2: Mount Google Drive
# ============================================================

from google.colab import drive
drive.mount('/content/gdrive')
print("✓ Google Drive mounted")

# ============================================================
# CELL 3: Setup Code (Load from repo or local)
# ============================================================

import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Set paths
DATASET_ROOT = "/content/gdrive/My Drive/Dataset"  # <-- CHANGE IF NEEDED
CHECKPOINT_DIR = "checkpoints/kaggle_colab"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Dataset root: {DATASET_ROOT}")
print(f"Checkpoint dir: {CHECKPOINT_DIR}")

# Verify dataset exists
if not os.path.exists(DATASET_ROOT):
    print(f"ERROR: Dataset not found at {DATASET_ROOT}")
    print("Please upload Dataset folder to Google Drive first!")
else:
    print(f"✓ Dataset found: {DATASET_ROOT}")
    print(f"  Contents: {os.listdir(DATASET_ROOT)[:5]}")

# ============================================================
# CELL 4: Dataset Loader
# ============================================================

class KaggleDeepfakeDataset(Dataset):
    def __init__(self, data_root, split='TRAIN', transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []
        
        split_dir = self.data_root / split.upper()
        if not split_dir.exists():
            print(f"WARN: Split dir {split_dir} not found")
            return
        
        # Load REAL images
        real_dir = split_dir / 'REAL'
        if real_dir.exists():
            for img_path in real_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 0))
        
        # Load FAKE images
        fake_dir = split_dir / 'FAKE'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 1))
        
        print(f"{split} dataset: {len(self.samples)} samples ({len([s for s in self.samples if s[1]==0])} real, {len([s for s in self.samples if s[1]==1])} fake)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# Transform pipeline
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

# Load datasets
print("\nLoading datasets...")
train_dataset = KaggleDeepfakeDataset(DATASET_ROOT, split='TRAIN', transform=train_transform)
val_dataset = KaggleDeepfakeDataset(DATASET_ROOT, split='VALIDATION', transform=val_transform)

# ============================================================
# CELL 5: Model Setup
# ============================================================

# For this, we need the model architecture. Since it's hard to load dynamically,
# we'll create a simplified version or try to import from the repo.
# For now, assume model is saved as .pth and can be loaded

# Try to clone repo for model code
!git clone -q https://github.com/Aafi04/HP-CoE-Agile-Challenge.git /content/repo 2>/dev/null || true

# Add repo to path
sys.path.insert(0, '/content/repo/AI-Based-Image-Authenticity-and-Deepfake-Detection-Tool')

try:
    # Try to load pre-trained model
    from backend.models.hybrid_model import HybridDeepfakeDetector
    model = HybridDeepfakeDetector()
    print("✓ Loaded HybridDeepfakeDetector architecture")
    
    # Load pre-trained weights (if available in Drive)
    model_weights_path = "/content/gdrive/My Drive/Dataset/hybrid_full_best.pt"
    if os.path.exists(model_weights_path):
        state_dict = torch.load(model_weights_path, map_location='cuda')
        model.load_state_dict(state_dict)
        print(f"✓ Loaded pre-trained weights from {model_weights_path}")
    else:
        print(f"! Pre-trained weights not found at {model_weights_path}")
        print("  Will train from scratch on this architecture")
        
except Exception as e:
    print(f"Could not import model: {e}")
    print("Creating simple CNN...") 
    # Fallback: Create simple model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(256 * 28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )

model = model.cuda()
print(f"✓ Model ready on GPU")

# ============================================================
# CELL 6: Training Setup
# ============================================================

DEVICE = 'cuda'
CONFIG = {
    'lr': 1e-5,
    'num_epochs': 10,
    'batch_size': 64,  # GPU can handle this
    'num_workers': 4,  # CPU workers for data loading
    'weight_decay': 1e-4,
}

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2], device=DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

print(f"""
✓ Training config:
  - Learning rate: {CONFIG['lr']}
  - Batch size: {CONFIG['batch_size']}
  - Epochs: {CONFIG['num_epochs']}
  - Device: {DEVICE}
  - Train batches: {len(train_loader)}
  - Val batches: {len(val_loader)}
""")

# ============================================================
# CELL 7: Run Training (MAIN LOOP - Takes 1-2 hours)
# ============================================================

model.train()
best_val_acc = 0
patience = 2
patience_counter = 0

print("\n" + "="*70)
print("STARTING GPU FINE-TUNING")
print("="*70)

for epoch in range(CONFIG['num_epochs']):
    # === TRAINING ===
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [TRAIN]", leave=True)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        train_loss += loss.item()
        train_correct += ((outputs > 0).float() == labels).sum().item()
        train_total += labels.size(0)
        
        # Update progress bar
        avg_loss = train_loss / (batch_idx + 1)
        avg_acc = train_correct / train_total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.4f}'})
    
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    
    # === VALIDATION ===
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [VAL]", leave=True)
        for images, labels in pbar_val:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            val_correct += ((outputs > 0).float() == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    
    # Print epoch summary
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
    print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
    print(f"{'='*70}\n")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(CHECKPOINT_DIR, 'best_kaggle.pt')
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ Best model saved: {best_model_path} (Accuracy: {val_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"! No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"\n✓ Early stopping triggered after {epoch+1} epochs")
            break

print("\n" + "="*70)
print("✓ TRAINING COMPLETE")
print(f"✓ Best validation accuracy: {best_val_acc:.4f}")
print(f"✓ Model saved to: {CHECKPOINT_DIR}/best_kaggle.pt")
print("="*70)

# ============================================================
# CELL 8: Download Model
# ============================================================

from google.colab import files

print("\nDownloading model...")
files.download(os.path.join(CHECKPOINT_DIR, 'best_kaggle.pt'))
print("✓ Model downloaded!")

# ============================================================
# CELL 9 (Optional): Evaluate
# ============================================================

print(f"\nFinal Results:")
print(f"  Validation Accuracy: {best_val_acc:.2%}")
print(f"  Epochs Run: {epoch+1}/{CONFIG['num_epochs']}")
print(f"  Model File: best_kaggle.pt (ready to download)")
print(f"\nNext steps:")
print(f"  1. Copy best_kaggle.pt to backend/models/kaggle_best.pt locally")
print(f"  2. Update backend/main.py to use the new model")
print(f"  3. Test inference with test_kaggle_diagnostics.py")
print(f"  4. Deploy! 🚀")
