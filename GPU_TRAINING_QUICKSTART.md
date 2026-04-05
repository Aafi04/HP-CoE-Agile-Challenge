# ⚡ QUICKSTART: GPU TRAINING IN 2 HOURS

**Your CPU training was on pace for 14+ hours. Here's the fix.**

---

## 🎯 FASTEST PATH (Google Colab) - Pick ONE

### **OPTION A: Use Your Existing Model (Recommended - 30 min setup)**

**1. Upload Dataset to Google Drive** (one-time, ~10 min)
```
- Open Google Drive on your browser  
- Create folder: My Drive/Dataset
- Drag-drop your local Dataset folder (140K images)
  ├── TRAIN/
  │   ├── REAL/
  │   └── FAKE/
  └── VALIDATION/
      ├── REAL/
      └── FAKE/
```

**2. Copy the Colab Training Script** (2 min)
```
- Go to https://colab.google.com
- Click "New Notebook"  
- Copy-paste the code from: colab_training_script.py (from this repo)
- Run all cells sequentially
```

**3. Wait for Training** (1-2 hours on free GPU)
```
- Colab will show progress bars for each epoch
- Expected accuracy: 70%+ after 10 epochs
```

**4. Download Trained Model** (1 min)
```
- Colab will auto-download best_kaggle.pt to your Downloads
```

**5. Deploy Locally** (10 min)
```powershell
# Copy model locally
cp Downloads/best_kaggle.pt backend/models/kaggle_best.pt

# Update backend/main.py line ~20
# MODEL_PATH = "backend/models/kaggle_best.pt"

# Test it
python -m pytest tests/test_kaggle_diagnostics.py
```

### **OPTION B: Upload Model to Drive First (If needed)**

If Colab can't find pre-trained model:
```powershell
# Upload your pre-trained model to Google Drive too
# Path: My Drive/Dataset/hybrid_full_best.pt
```

---

## 📊 WHAT YOU GET

| Metric | CPU (Our First Try) | Colab GPU |
|--------|-------------------|-----------|
| Time per epoch | ~1.5 hours | ~6-10 min |
| Total for 10 epochs | **14-15 hours** ❌ | **1-2 hours** ✅ |
| Cost | (Your electricity) | **FREE** ($0) |
| GPU Memory | None | 16-32 GB |
| Setup Time | 0 min | 15-20 min |

**Time Saved:** 12-13 hours 🚀

---

## 🔗 IMPORTANT: Colab Link

**Direct Link to Start:**
https://colab.google.com/#create=true

**Instructions:**
1. Click "New Notebook"
2. Top-left: Rename to "deepfake_training"  
3. Change runtime to GPU:
   - Menu > Runtime > Change Runtime Type
   - Hardware Accelerator: Select "GPU"
   - Click "Save"

---

## ✅ CHECKLIST

- [ ] Dataset uploaded to Google Drive (My Drive/Dataset/)
- [ ] Google Colab notebook created  
- [ ] Code cells pasted from colab_training_script.py
- [ ] Runtime changed to GPU
- [ ] All cells executed (8 cells total)
- [ ] Training started (shows "Epoch 1/10")
- [ ] Wait 1-2 hours for completion
- [ ] best_kaggle.pt downloaded
- [ ] Copied to backend/models/kaggle_best.pt locally
- [ ] backend/main.py updated with new model path
- [ ] Ready for deployment! 🚀

---

## 🆘 TROUBLESHOOTING

### "Dataset not found at /content/gdrive/My Drive/Dataset"
→ Check you uploaded Dataset to Google Drive first  
→ Verify folder name is exactly "Dataset"

### "GPU not available"  
→ Menu > Runtime > Change Runtime Type  
→ Select "GPU" hardware accelerator

### "Out of memory"
→ Cell 6: Reduce batch_size from 64 → 32
→ Or use T4 GPU instead of V100 (free tier)

### "Training is still slow on Colab"
→ Check GPU usage: In Colab, left panel shows GPU stats
→ If CPU%, reduce num_workers from 4 → 2

---

## 📞 IF ISSUES PERSIST

1. **Check Colab Runtime:**
   ```
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show GPU
   ```

2. **Verify Dataset Structure:**
   ```
   import os
   print(os.listdir('/content/gdrive/My Drive/Dataset'))
   # Should show: ['TRAIN', 'VALIDATION']
   ```

3. **Check Available VRAM:**
   ```
   print(torch.cuda.get_device_properties(0).total_memory / 1e9)
   # Should be ~16GB or more
   ```

---

## ⏱ TIMELINE

| Time | Action |
|------|--------|
| Now | Upload dataset to Drive (10 min) |
| +10 min | Create Colab notebook (2 min) |
| +12 min | Paste code and setup (5 min) |
| +17 min | Check GPU works (3 min) |
| +20 min | Start training |
| +80-120 min | Training completes (1-2 hours) |
| +82-122 min | Download model (2 min) |
| +85-125 min | Deploy locally (15 min) |
| **+100-140 min** | **READY TO SUBMIT** ✅ |

**Total from now: ~2-3 hours to deployment-ready**

---

## 💡 WHY COLAB WORKS

✅ **Free GPU access** - No credits wasted  
✅ **Pre-installed PyTorch** - No dependency issues  
✅ **Google Drive integration** - Easy dataset access  
✅ **Instant runtime** - No VM startup delays  
✅ **Web-based** - No terminal/SSH needed  

---

**You're going from 14+ hours to 1-2 hours. Let's go! 🚀**

*Choice A is recommended - takes ~3 hours total (including upload), then model is ready.*
