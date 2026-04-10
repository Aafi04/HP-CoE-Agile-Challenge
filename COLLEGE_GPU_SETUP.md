# 🖥️ COLLEGE GPU SETUP & NEXT SESSION PLAN

**Objective:** Identify GPU resources, configure Remote SSH, run fine-tuning, achieve 70%+ accuracy

**📌 QUICK REFERENCE:** All GPU specs, network info, and setup commands are in [`../GPU_SPECS.md`](../GPU_SPECS.md) (keep outside Git repo)

---

## SESSION 1 (Next Session): GPU DISCOVERY & SETUP

### Step 1: Identify GPU Specifications

When you have access to college system, gather:

```
1. GPU Model: nvidia-smi
   - Output shows: Tesla V100, A100, RTX 4090, T4, etc.

2. VRAM: nvidia-smi
   - Example: 16GB, 24GB, 32GB, 80GB

3. Compute Capability: nvidia-smi -q
   - Used for optimization flags

4. Number of GPUs: nvidia-smi -q | grep "Gpu" | wc -l

5. PyTorch Access: python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
```

### Step 2: Configure Remote SSH in VS Code

```
1. Install "Remote - SSH" extension (already done ✓)

2. Add SSH config:
   - Press Ctrl+Shift+P > Remote-SSH: Open Configuration File
   - Add entry:

   Host CollegeGPU
       HostName <college_server_ip_or_hostname>
       User <your_college_username>
       IdentityFile ~/.ssh/id_rsa

3. Connect:
   - Press Ctrl+Shift+P > Remote-SSH: Connect to Host
   - Select "CollegeGPU"
   - Accept host key when prompted

4. Verify connection:
   - Should show "SSH: CollegeGPU" in bottom-left corner
   - Open terminal: Ctrl+`
   - Run: nvidia-smi
```

### Step 3: Transfer Dataset to College System (if needed)

```bash
# Option A: SCP (secure copy)
scp -r "C:\Users\Aafi\Desktop\Dataset" <college_user>@<college_ip>:/path/on/college/

# Option B: If dataset already accessible on college system
# Just update paths in finetune_kaggle.py

# Option C: If college has shared storage (NFS, etc)
# Check with IT
```

### Step 4: Setup Python Environment on College GPU

```bash
# SSH into college system
ssh CollegeGPU

# Check Python version
python --version  # Should be 3.8+

# Install dependencies
pip install torch torchvision torchaudio  # Will auto-detect CUDA
pip install pillow tqdm numpy

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## SESSION 2 (After GPU Access): TRAINING EXECUTION

### Adjust Training Config for Your GPU

**Edit: `training/finetune_kaggle.py`**

Based on your GPU VRAM, set batch_size:

```
GPU VRAM    | Batch Size | Est. Training Time
----------- | ---------- | ------------------
8GB  (T4)   | 16         | 2-3 hours
16GB (V100) | 32         | 1-1.5 hours
24GB (RTX)  | 64         | 45-60 min
40GB (A100) | 128        | 30-45 min
80GB (A100) | 256        | 20-30 min
```

Find in finetune_kaggle.py and modify:

```python
CONFIG = {
    'lr': 1e-5,           # Keep this (domain adaptation)
    'num_epochs': 10,     # Can reduce to 5 if short on time
    'batch_size': 16,     # ← CHANGE BASED ON YOUR GPU VRAM
    'num_workers': 4,     # ← Can increase on Linux (was 0 on Windows)
    'weight_decay': 1e-4, # Keep this
}
```

Also update dataset path:

```python
data_root = "/path/to/Dataset"  # Update to college system path
```

### Run Training

```bash
# SSH into college GPU system
ssh CollegeGPU
cd /path/to/AI-Based-Image-Authenticity-and-Deepfake-Detection-Tool

# Run training (no need for background here if using tmux/screen)
python training/finetune_kaggle.py --data_root /path/to/Dataset

# OR use tmux for background (so you can disconnect)
tmux new-session -d -s training "python training/finetune_kaggle.py --data_root /path/to/Dataset"
tmux attach -t training  # To check progress anytime
```

### Monitor Training

```bash
# Watch progress
nvidia-smi -l 1  # GPU usage every 1 second

# Check specific training metrics
tail -f checkpoints/kaggle_finetune/training.log  # if logging

# Or just read console output from tmux session
tmux capture-pane -t training -p
```

### After Training Completes

```bash
# Verify model saved
ls -lh checkpoints/kaggle_finetune/best_kaggle.pt

# Copy to backend location
cp checkpoints/kaggle_finetune/best_kaggle.pt backend/models/kaggle_best.pt

# Run diagnostics test
python tests/test_kaggle_diagnostics.py

# Expected output: 70%+ accuracy on Kaggle test set
```

---

## TROUBLESHOOTING FOR COLLEGE GPU

| Issue               | Solution                                                     |
| ------------------- | ------------------------------------------------------------ |
| CUDA out of memory  | Reduce batch_size: 64 → 32 → 16                              |
| Connection drops    | Use tmux/screen for persistent terminal                      |
| GPU not detected    | `nvidia-smi` should show device; if not, check CUDA install  |
| Slow training       | Check `nvidia-smi` - if GPU% < 30%, increase batch_size      |
| Dataset not found   | Update data_root path in finetune_kaggle.py                  |
| Module imports fail | Reinstall: `pip install torch torchvision --force-reinstall` |

---

## QUICK REFERENCE: Terminal Commands for College GPU

```bash
# You'll need these:
nvidia-smi                           # Check GPU status
python training/finetune_kaggle.py   # Run training
tmux new-session -d -s train "cmd"   # Background session
tmux attach -t train                 # Attach to session
tmux kill-session -t train           # Close session
```

---

## SUCCESS CRITERIA (Phase 2)

- [x] GPU identified and Remote SSH working
- [x] PyTorch can see GPU
- [ ] Training starts successfully
- [ ] Loss decreases over epochs (15 → 5 → 1 range)
- [ ] Accuracy improves (50% → 70%+ on validation)
- [ ] Best model saved
- [ ] Tests pass: 70%+ accuracy on Kaggle images

---

## THEN (Phase 3): DEPLOYMENT

Once training done and model achieves 70%+ accuracy:

1. Update backend/main.py MODEL_PATH
2. Run full test suite
3. Record demo video
4. Prepare final documentation

---

**Next action:** Share college GPU specs when you have access, and we'll proceed!
