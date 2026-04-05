# 🚀 VS CODE COLAB EXTENSION - QUICKSTART

**Train directly from VS Code with T4 GPU - No dataset upload needed!**

---

## ⚡ SETUP (One-time, ~5 minutes)

### Step 1: Install Colab Extension in VS Code
```
1. Open VS Code
2. Extensions (Ctrl+Shift+X)
3. Search: "Google Colab"
4. Click "Install" on the official Google extension
5. Wait for installation to complete
```

### Step 2: Open Notebook in VS Code
```
1. VS Code: File > Open File
2. Navigate to: colab_vscode_training.ipynb
3. Double-click to open
4. Should show notebook interface with cells
```

---

## 🎯 BEFORE RUNNING CELLS

### Option A: Use Google Drive (Recommended - No Re-upload)
```
1. Upload Dataset folder to Google Drive (one-time, ~10 min)
   - Open https://drive.google.com
   - Create folder: My Drive/Dataset
   - Upload your local Dataset folder

2. In Cell 4 of notebook, verify path:
   DATASET_ROOT = "/content/gdrive/MyDrive/Dataset"
```

### Option B: Direct Local Reference (Fastest)
If you want to test locally first without GPU:
```
1. Cell 4: Change path to your local Dataset
   DATASET_ROOT = r"C:\Users\Aafi\Desktop\Dataset"

2. Won't work on Colab, but good for local testing
```

---

## ▶️ RUN TRAINING (Do This Now)

### Step 1: Select Colab Runtime
```
1. Open: colab_vscode_training.ipynb in VS Code
2. Top-right: Click "Select a runtime"
3. Choose: "Google Colab"
4. When prompted: Sign in with your Google account
5. Accept permissions
```

### Step 2: Choose T4 GPU
```
After signing in:
1. Top-right: Click runtime dropdown again
2. Click "⚙️ Settings" or runtime name
3. Select: T4 GPU (free tier)
4. Click "Connect" or "Start"
5. Wait for connection indicator (should be green)
```

### Step 3: Run Cells Sequentially
```
1. Click Cell 1 (Install & Verify GPU)
2. Press Ctrl+Enter (or click Run button)
3. Wait for ✓ checkmarks

4. Repeat for Cells 2-6 (setup cells)

5. Cell 7: MAIN TRAINING
   - This runs 1-2 hours on T4 GPU
   - Watch progress bars
   - GPU usage shows at bottom

6. Cell 8: Download model
   - Auto-downloads best_kaggle.pt
   - Check your Downloads folder
```

---

## 📊 WHAT YOU'LL SEE

**Cell 1 Output (Should show):**
```
✓ GPU Available: True
✓ GPU: Tesla T4
✓ VRAM: 15.0GB
✓ CUDA Version: 11.8
```

**Cell 4 Output (Should show):**
```
✓ Dataset found: /content/gdrive/MyDrive/Dataset
✓ Training: 140000 real + some fake...
✓ Validation: 39000 real + some fake...
```

**Cell 7 Output (Training - Main Event):**
```
======================================================================
STARTING GPU FINE-TUNING ON COLAB T4
======================================================================

Epoch 1/10 [TRAIN]: 100%|████| 2188/2188 [06:32<00:00, 0.18s/it, loss=0.1234, acc=0.9234]

Epoch 1: Train Loss=0.1234 Acc=0.9234 | Val Loss=0.1567 Acc=0.8901
✓ Best model saved (accuracy: 0.8901)

[... 9 more epochs ...]

======================================================================
✓ TRAINING COMPLETE: Best accuracy = 0.7523
✓ Model: checkpoints/kaggle_colab/best_kaggle.pt
======================================================================
```

**Cell 8 Output (Download):**
```
Model size: 98.5 MB
Downloading model...
✓ Downloaded to your local Downloads folder!

Next steps:
1. Copy best_kaggle.pt to backend/models/kaggle_best.pt locally
2. Update backend/main.py...
```

---

## ⏱️ TIMELINE

| Step | Time | Action |
|------|------|--------|
| Now | 2 min | Install Colab extension |
| +2 min | 1 min | Open notebook in VS Code |
| +3 min | 2 min | Authenticate with Google |
| +5 min | 1 min | Select T4 GPU |
| +6 min | 2 min | Run setup cells (1-6) |
| +8 min | 60-90 min | **Training runs (Cell 7)** |
| +108 min | 2 min | Download model (Cell 8) |
| +110 min | 10 min | Deploy locally |
| **+120 min** | **~2 hours total** | ✅ **READY** |

---

## 🔧 TROUBLESHOOTING

### "Runtime not recognized"
→ Extension not installed. VS Code > Extensions > Search "Google Colab" > Install

### "GPU not available (showing CPU)"
→ Cell 1 will show `GPU Available: False`
→ Redo Step 2: Select T4 GPU from dropdown

### "Dataset not found"
→ Cell 4 shows error  
→ Make sure Dataset is uploaded to Google Drive at: `My Drive/Dataset/`

### "Cells won't run"
→ Not authenticated  
→ Top-right dropdown > Log in with Google account  
→ Grant permissions

### "Training is still slow"
→ GPU might not be active  
→ Check Cell 1 output shows T4  
→ Monitor resource usage: Bottom panel should show GPU%

### "Memory error during training"
→ Reduce batch_size in Cell 6: `'batch_size': 64` → `32`
→ Restart and rerun

---

## 📝 AFTER TRAINING COMPLETES

```powershell
# 1. Download model comes automatically
#    Check: C:\Users\Aafi\Downloads\best_kaggle.pt

# 2. Copy to project
cp "C:\Users\Aafi\Downloads\best_kaggle.pt" `
   "C:\Users\Aafi\Desktop\Agile Challenge\AI-Based-Image-Authenticity-and-Deepfake-Detection-Tool\backend\models\kaggle_best.pt"

# 3. Update backend/main.py (line ~20)
#    MODEL_PATH = "backend/models/kaggle_best.pt"

# 4. Test locally
cd "C:\Users\Aafi\Desktop\Agile Challenge\AI-Based-Image-Authenticity-and-Deepfake-Detection-Tool"
python tests/test_kaggle_diagnostics.py

# 5. Deploy!
```

---

## ✨ KEY BENEFITS OF VS CODE METHOD

✅ **No dataset upload** - References Drive directly  
✅ **Code stays local** - Full edit/debug in VS Code  
✅ **GPU runs remote** - Execution on Colab GPU  
✅ **Seamless experience** - Cells run in VS Code interface  
✅ **Easy GPU switch** - Change T4/V100/TPU with one click  
✅ **T4 is free** - ~$1/hour alternative to your local CPU  

---

## 🎯 VS CODE COLAB vs WEB COLAB

| Feature | VS Code | Web |
|---------|---------|-----|
| Dataset Upload | ❌ Direct ref | ✅ Required |
| Edit Experience | ✅ Native | 🟡 Browser |
| GPU Switch | ✅ Easy | 🟡 Settings menu |
| Local File Sync | ✅ Auto | ❌ Manual |
| Code History | ✅ Git | ❌ Colab only |
| **Time to train** | **~2-3 hours** | **~2-3 hours** |
| **Setup time** | **5 minutes** | **15 minutes** |

**Winner: VS Code Extension** 🏆

---

## 📞 QUICK CHECKLIST

- [ ] Colab extension installed
- [ ] Google account authenticated
- [ ] colab_vscode_training.ipynb opened
- [ ] T4 GPU selected ("Google Colab" runtime)
- [ ] Green connection indicator (✓ Connected)
- [ ] Cell 1 shows GPU: Tesla T4
- [ ] Cell 4 shows Dataset found
- [ ] Cell 7 running training (watch progress bars)
- [ ] Cell 8 downloads model
- [ ] Model deployed to backend/models/
- [ ] Local tests pass (70%+ accuracy)
- [ ] Ready for deployment! 🚀

---

**Ready to start? Open the notebook and hit Ctrl+Enter on Cell 1!** 👍

