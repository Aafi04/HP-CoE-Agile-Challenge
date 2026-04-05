# 🚀 GOOGLE COLAB GPU TRAINING SETUP

**Fastest Path to 1-2 Hour Training (vs 14+ hours on CPU)**

---

## ✨ Why Colab?
- ✅ **Instant GPU access** (T4 or V100) - no setup required
- ✅ **Free** for limited usage  
- ✅ **Runs in browser** - no VM management
- ✅ **Can mount Google Drive** for dataset access
- ⏱ **Training time: 1-2 hours** (vs 14+ on CPU)

---

## 📋 SETUP STEPS

### Step 1: Prepare Dataset (One-time, ~10 min)

**Option A: Upload to Google Drive** (Recommended)
```bash
# On your local machine, compress dataset
cd C:\Users\Aafi\Desktop
# Upload Dataset folder to your Google Drive (browser, drag-drop)
```

**Option B: Create Cloud Storage bucket**
```bash
# In PowerShell
gsutil mb gs://your-bucket-name-deepfake/
gsutil -m cp -r "C:\Users\Aafi\Desktop\Dataset" gs://your-bucket-name-deepfake/
```

### Step 2: Create Colab Notebook
1. Go to [Google Colab](https://colab.google.com)
2. Click **"New Notebook"**
3. Rename to: `deepfake_gpu_training`
4. **COPY THE CODE BELOW** into cells

---

## 🐍 COLAB NOTEBOOK CODE

### Cell 1: Install & Import Dependencies
```python
# Install required packages
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q pillow tqdm wandb

# Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cell 2: Mount Google Drive (if using Drive for dataset)
```python
from google.colab import drive
drive.mount('/content/gdrive')
print("✓ Google Drive mounted at /content/gdrive")
```

### Cell 3: Clone Repository & Setup Code
```python
import os
os.chdir('/content')

# Clone the repo
!git clone https://github.com/Aafi04/HP-CoE-Agile-Challenge.git
os.chdir('/content/HP-CoE-Agile-Challenge/AI-Based-Image-Authenticity-and-Deepfake-Detection-Tool')

print("✓ Repository cloned")
print("✓ Files ready at:", os.getcwd())
```

### Cell 4: Create GPU-Optimized Training Script
```python
# Create training script with GPU optimizations
training_code = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import tqdm
from datetime import datetime

# CONFIG FOR GPU
CONFIG = {
    'lr': 1e-5,
    'num_epochs': 10,
    'batch_size': 64,  # Larger batch on GPU
    'num_workers': 4,  # Parallel loading on GPU
    'weight_decay': 1e-4,
    'device': 'cuda',  # Force GPU
    'pin_memory': True,  # GPU memory optimization
}

device = torch.device(CONFIG['device'])
print(f"Using device: {device}")

# Load model
print("Loading pre-trained model...")
model = torch.load("/content/gdrive/My Drive/Dataset/model_cache/hybrid_full_best.pt", map_location=device)
# OR if model is in repo:
# from AI-Based-Image-Authenticity-and-Deepfake-Detection-Tool.backend.models.hybrid_model import HybridDeepfakeDetector
# model = HybridDeepfakeDetector().to(device)
# model.load_state_dict(torch.load("backend/models/hybrid_full_best.pt"))

model = model.to(device)
model.train()

# Dataset mapping
class KaggleDeepfakeDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None):
        self.data_root = Path(data_root)
        self.split_dir = self.data_root / split
        self.transform = transform
        self.samples = []
        
        # Load real images
        real_dir = self.split_dir / 'REAL'
        for img_path in real_dir.glob('*.jpg'):
            self.samples.append((str(img_path), 0))
        
        # Load fake images  
        fake_dir = self.split_dir / 'FAKE'
        for img_path in fake_dir.glob('*.jpg'):
            self.samples.append((str(img_path), 1))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# Transforms with GPU optimization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

print("Loading Kaggle dataset...")
data_root = "/content/gdrive/My Drive/Dataset"  # Adjust path based on your Drive structure
train_dataset = KaggleDeepfakeDataset(data_root, split='TRAIN', transform=transform)
val_dataset = KaggleDeepfakeDataset(data_root, split='VALIDATION', transform=transform)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# DataLoaders with GPU optimization
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=CONFIG['pin_memory']
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=CONFIG['pin_memory']
)

# Optimizer & Loss
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2], device=device))
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

# Training loop
os.makedirs('checkpoints/kaggle_colab', exist_ok=True)
best_val_acc = 0
patience = 2
patience_counter = 0

print("\\n" + "="*60)
print("Starting GPU fine-tuning")
print("="*60 + "\\n")

for epoch in range(CONFIG['num_epochs']):
    # Train
    model.train()
    train_loss = 0
    train_acc = 0
    train_count = 0
    
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += ((outputs > 0).float() == labels).sum().item()
        train_count += labels.size(0)
        
        pbar.set_postfix({'loss': f'{train_loss/train_count:.4f}', 'acc': f'{train_acc/train_count:.4f}'})
    
    train_loss /= len(train_loader)
    train_acc /= train_count
    
    # Validation
    model.eval()
    val_loss = 0
    val_acc = 0
    val_count = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            val_acc += ((outputs > 0).float() == labels).sum().item()
            val_count += labels.size(0)
    
    val_loss /= len(val_loader)
    val_acc /= val_count
    
    print(f"Epoch {epoch+1}: TrnLoss={train_loss:.4f} TrnAcc={train_acc:.4f} | ValLoss={val_loss:.4f} ValAcc={val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'checkpoints/kaggle_colab/best_kaggle.pt')
        print(f"✓ Best model saved (accuracy: {val_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping (no improvement for {patience} epochs)")
            break

print(f"\\n✓ Training complete. Best validation accuracy: {best_val_acc:.4f}")
print(f"✓ Model saved to: checkpoints/kaggle_colab/best_kaggle.pt")
'''

# Save to file
with open('/content/train_gpu.py', 'w') as f:
    f.write(training_code)

print("✓ GPU training script created")
```

### Cell 5: Run Training on GPU
```python
import subprocess
result = subprocess.run(['python', '/content/train_gpu.py'], capture_output=False, text=True)
```

### Cell 6: Download Results
```python
# Download trained model
from google.colab import files
files.download('checkpoints/kaggle_colab/best_kaggle.pt')
print("✓ Model downloaded to your local machine")
```

---

## 🔄 ALTERNATIVE: Use GCP VM (if Colab too slow)

```bash
# Create GPU VM on GCP
gcloud compute instances create deepfake-training \
  --zone=us-central1-a \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --machine-type=n1-highmem-8 \
  --image-family=deeplearning-platform-release \
  --image-project=deeplearning-images

# SSH into VM
gcloud compute ssh deepfake-training --zone=us-central1-a

# On VM: Git clone, install deps, run training
```

---

## ⏱ Expected Timeline

| Step | Time |
|------|------|
| Upload dataset to Drive | 5-10 min |
| Create Colab notebook | 2 min |
| Install packages | 2 min |
| Run training on GPU | **1-2 hours** |
| Download model | 2-5 min |
| **TOTAL** | **~2-3 hours** |

**Compare to CPU:** 14+ hours → **7-10x FASTER** ✅

---

## 📝 NOTES

- **Colab Free Tier:** ~9 hours per day, reset at ~ 12am UTC
- **Pro Tier:** $10/month, unlimited usage
- **Dataset Path:** Adjust `/content/gdrive/My Drive/Dataset` based on where you upload it
- **GPU Memory:** T4 (16 GB) or V100 (32 GB) both sufficient for batch_size=64
- **Batch Size:** Can increase to 128 on V100 for even faster training

---

## ✅ QUICK CHECKLIST

- [ ] Upload Dataset folder to Google Drive
- [ ] Go to [Google Colab](https://colab.google.com)
- [ ] Create new notebook
- [ ] Copy-paste the code cells above
- [ ] Run all cells
- [ ] Wait 1-2 hours
- [ ] Download `best_kaggle.pt`
- [ ] Copy to `backend/models/kaggle_best.pt` locally
- [ ] Update main.py to use new model
- [ ] Deploy! 🚀

