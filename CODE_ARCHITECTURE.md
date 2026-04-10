# 🏗️ CODE ARCHITECTURE REFERENCE

**Quick reference for key files, entry points, and how components connect.**

---

## PROJECT STRUCTURE

```
AI-Based-Image-Authenticity-and-Deepfake-Detection-Tool/
├── backend/
│   ├── main.py                           # FastAPI server (inference endpoint)
│   ├── inference_enhancements.py         # TTA + calibration + QA
│   ├── models/
│   │   ├── hybrid_model.py              # HybridDeepfakeDetector (18.9M params)
│   │   ├── efficientnet.py              # EfficientNet-B4 backbone (17.5M)
│   │   ├── fft_branch.py                # FFT frequency analysis (1.4M)
│   │   ├── hybrid_full_best.pt          # Pre-trained weights (95.65% on FF++)
│   │   └── kaggle_best.pt               # Fine-tuned weights (~70% on Kaggle) [TO BE CREATED]
│   ├── Dockerfile                        # Container config
│   └── entrypoint.sh                     # Container startup
│
├── training/
│   ├── finetune_kaggle.py               # ⭐ MAIN ENTRY POINT (Phase 2)
│   ├── train_hybrid.py                  # Full training from scratch
│   ├── train.py                         # Generic training loop
│   └── [other training scripts]
│
├── data/
│   ├── dataset_kaggle.py                # KaggleDeepfakeDataset loader
│   ├── augmentations.py                 # Image transforms (Windows-compatible)
│   └── dataset.py                       # FF++ dataset loader
│
├── tests/
│   ├── test_kaggle_diagnostics.py       # ⭐ VALIDATION SCRIPT (Phase 3)
│   └── [other test files]
│
├── models/ [backup models]
├── evaluation/ [GradCAM, visualization]
├── frontend/ [Next.js UI]
└── README.md
```

---

## PHASE 2: TRAINING (What You'll Run)

### Entry Point: `training/finetune_kaggle.py`

**What it does:**
1. Loads HybridDeepfakeDetector from `backend/models/hybrid_model.py`
2. Loads pre-trained weights from `backend/models/hybrid_full_best.pt`
3. Loads Kaggle dataset from `data/dataset_kaggle.py`
4. Fine-tunes for 10 epochs with LR=1e-5 (very conservative)
5. Saves best model to `checkpoints/kaggle_finetune/best_kaggle.pt`

**Key config (lines ~30-40):**
```python
CONFIG = {
    'lr': 1e-5,          # Very low for domain adaptation
    'num_epochs': 10,
    'batch_size': 16,    # ← Adjust for your GPU VRAM
    'num_workers': 4,    # ← Increase for Linux/college GPU
    'weight_decay': 1e-4,
}
```

**Dataset path (line ~XX):**
```python
data_root = "C:\Users\Aafi\Desktop\Dataset"  # ← Update for college system
```

**Run command:**
```bash
python training/finetune_kaggle.py --data_root /path/to/Dataset
```

---

## PHASE 1: MODEL ARCHITECTURE

### HybridDeepfakeDetector (`backend/models/hybrid_model.py`)

**Architecture:**
```
Input (3, 224, 224)
    ↓
    ├─→ EfficientNet-B4 (efficientnet.py)
    │       Backbone: Pretrained ImageNet
    │       Output: 1792-dim feature vector
    │
    └─→ FFT Branch (fft_branch.py)
            Frequency analyze: Extract spatial patterns
            Output: 512-dim feature vector
    ↓
Fusion: Concatenate [EfficientNet features] + [FFT features]
    ↓ (2304-dim)
Classification Head:
    Linear(2304 → 1024)
    ReLU, Dropout
    Linear(1024 → 1)
    ↓
Output: logits (0-1 via sigmoid)
```

**Total: 18.9M parameters**

---

## PHASE 1: INFERENCE ENHANCEMENTS (`backend/inference_enhancements.py`)

**Components:**
1. **Test-Time Augmentation (TTA)**: Predict on original + h-flip + v-flip, average
2. **Confidence Calibration**: Adjust raw output to compensate for model's bias
3. **Image Quality Filtering**: Detect suspicious brightness/contrast/color
4. **Threshold Adjustment**: 0.5 → 0.3 for more aggressive deepfake detection

**Usage in backend:**
```python
from backend.inference_enhancements import EnhancedPredictor

predictor = EnhancedPredictor(model_path="backend/models/hybrid_full_best.pt")
result = predictor.predict(image)  # Returns {"label": "REAL"/"FAKE", "confidence": 0.8, ...}
```

---

## PHASE 3: VALIDATION (`tests/test_kaggle_diagnostics.py`)

**What it does:**
1. Loads fine-tuned model from `backend/models/kaggle_best.pt`
2. Evaluates on Kaggle test set
3. Prints accuracy, precision, recall, F1
4. Saves confidence distributions to JSON

**Success criterion:** ≥70% accuracy

**Run:**
```bash
python tests/test_kaggle_diagnostics.py
```

---

## BACKEND SERVICE (`backend/main.py`)

**FastAPI endpoints:**
- `GET /`: Health check
- `POST /predict`: Single image prediction
- `POST /predict-video`: Video frame-by-frame prediction

**Uses:**
- `inference_enhancements.EnhancedPredictor` for predictions
- `data/augmentations.py` for preprocessing

**To update after training:**
```python
# Line ~20 in main.py
MODEL_PATH = "backend/models/hybrid_full_best.pt"  # ← Change to
MODEL_PATH = "backend/models/kaggle_best.pt"       # ← After fine-tuning
```

**Run backend:**
```bash
python backend/main.py  # Starts on http://localhost:8000
```

---

## DATASET STRUCTURE

### Training data: `data/dataset_kaggle.py`

**Loader:**
```python
from data.dataset_kaggle import KaggleDeepfakeDataset

dataset = KaggleDeepfakeDataset(
    data_root="C:\Users\Aafi\Desktop\Dataset",
    split="TRAIN",  # or "VALIDATION"
    transform=transform
)

for img, label in dataset:
    # label: 0=REAL, 1=FAKE
    pass
```

**Folder structure:**
```
Dataset/
├── TRAIN/
│   ├── REAL/ (70K images)
│   └── FAKE/ (70K images)
└── VALIDATION/
    ├── REAL/ (19.5K)
    └── FAKE/ (19.5K)
```

---

## KEY FILES TO MODIFY (For College GPU)

| File | Change | Location |
|------|--------|----------|
| `training/finetune_kaggle.py` | batch_size, num_workers, data_root | ~lines 30-40 |
| `backend/main.py` | MODEL_PATH | ~line 20 |
| `tests/test_kaggle_diagnostics.py` | data_root if needed | ~line 10 |

---

## DEPENDENCIES

```
PyTorch:     torch, torchvision, torchaudio
Data:        pillow, numpy
Training:    tqdm
Backend:     fastapi, uvicorn
Frontend:    next.js, react (separate)
```

**Install:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow tqdm fastapi uvicorn
```

---

## GIT COMMITS (Recent)

```
f04b15c  docs: comprehensive GPU solution summary
4dda9b5  docs: add GPU training quickstart
914a5d7  fix: add VS Code Colab extension notebook
3161cce  fix: Windows multiprocessing issues in fine-tuning script
3a26331  feat: add domain shift recovery infrastructure
```

**Repository:** https://github.com/Aafi04/HP-CoE-Agile-Challenge

---

## COMMON TASKS & ENTRY POINTS

| Task | Entry Point | Command |
|------|-------------|---------|
| Fine-tune model | `training/finetune_kaggle.py` | `python training/finetune_kaggle.py --data_root /path` |
| Test accuracy | `tests/test_kaggle_diagnostics.py` | `python tests/test_kaggle_diagnostics.py` |
| Run backend API | `backend/main.py` | `python backend/main.py` |
| Load model | `backend/models/hybrid_model.py` | `from backend.models.hybrid_model import HybridDeepfakeDetector` |
| Inference | `backend/inference_enhancements.py` | `from backend.inference_enhancements import EnhancedPredictor` |

---

**Last Updated:** April 10, 2026
