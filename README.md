# DualCore: AI-Based Image Authenticity & Deepfake Detection Tool

A production-grade deepfake detection system combining spatial (CNN) and frequency domain (FFT) analysis to identify forged images and videos with 96.21% accuracy on the Kaggle deepfake dataset.

## Overview

**DualCore** is a hybrid deep learning system designed to detect and analyze deepfakes in images and videos. It leverages:

- **EfficientNet-B4** for spatial domain feature extraction (18.5M parameters)
- **FFT-based frequency analysis** for anomaly detection (~0.4M parameters)  
- **Fusion classifier** combining both branches for robust decision-making
- **Test-time augmentation (TTA)** for inference stability
- **GradCAM visualization** with face-guided heatmaps for explainability
- **Domain-specific confidence calibration** for Kaggle dataset alignment

### Quick Stats

| Metric | Value |
|--------|-------|
| **Model Accuracy** | 96.21% (Kaggle validation) |
| **Controlled Image Set** | 6/6 (100%) |
| **Random Dataset** | 12/13 (92.3%) |
| **Model Size** | 76.6MB |
| **Inference Time (CPU)** | 500-800ms per image |
| **Inference Time (GPU)** | ~200-300ms per image |
| **Parameters** | 18.9M |
| **Status** | **Production Ready** ✅ |

## Project Structure

```
AI-Based-Image-Authenticity-and-Deepfake-Detection-Tool/
├── backend/                    # FastAPI application
│   ├── main.py                # API endpoints (/predict, /predict_video, /health, /model-info)
│   ├── inference_enhancements.py  # Enhanced predictor with TTA & quality checks
│   ├── confidence_calibrator.py   # Domain-specific confidence scaling
│   ├── Dockerfile             # Container image definition
│   ├── entrypoint.sh          # Startup script with model download & verification
│   └── models/                # Model storage (runtime)
│
├── frontend/                   # Next.js web interface (React 19, Tailwind CSS)
│   ├── app/page.tsx           # Main UI with drag-drop upload, live results
│   ├── package.json           # Frontend dependencies (Next.js 16, TypeScript)
│   └── public/                # Static assets
│
├── models/                     # Model architecture definitions
│   ├── hybrid.py              # Spatial + frequency fusion architecture
│   ├── efficientnet_branch.py # EfficientNet-B4 for spatial features
│   └── fft_branch.py          # FFT branch for frequency analysis
│
├── data/                       # Dataset utilities
│   ├── augmentations.py       # Validation augmentation pipeline
│   └── dataset_loader.py      # Kaggle/FaceForensics++ dataset handling
│
├── evaluation/                 # Model analysis & visualization
│   ├── enhanced_gradcam.py    # Advanced GradCAM with face-aware masking
│   ├── gradcam.py             # Core GradCAM implementation
│   └── metrics.py             # Accuracy, ROC, AUC computation
│
├── training/                   # Training scripts
│   ├── train_hybrid.py        # Main training pipeline
│   ├── calibration.py         # Confidence calibration procedures
│   └── configs/               # Hyperparameter configurations
│
├── tests/                      # Test suite (16 test files)
│   ├── test_image_endpoint.py # Image API validation
│   ├── test_video_endpoint.py # Video API validation
│   ├── test_confidence_calibrator.py  # Confidence formula checks
│   ├── test_model_metadata.py # Model identity verification
│   └── benchmark_*.py         # Performance benchmarking tools
│
├── artifacts/                  # Baseline data & benchmark results
│   ├── baselines/             # Pre/post-fix baseline artifacts
│   └── benchmark_*.json       # Performance metrics (latency, throughput)
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # See backend/Dockerfile
└── cloudbuild.yaml            # Google Cloud Build pipeline
```

## Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for frontend)
- **CUDA 11.8+** (optional, for GPU acceleration)
- **Docker** (optional, for containerized deployment)

### 2. Setup Backend

```bash
# Clone repository
git clone https://github.com/Aafi04/HP-CoE-Agile-Challenge.git
cd AI-Based-Image-Authenticity-and-Deepfake-Detection-Tool

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model (must be placed at models/hybrid_kaggle_finetuned.pt)
# Or set MODEL_PATH environment variable if storing elsewhere
export MODEL_PATH="path/to/hybrid_kaggle_finetuned.pt"

# Start FastAPI backend server
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend server will be available at `http://127.0.0.1:8000`

### 3. Setup Frontend

```bash
cd frontend

# Install Node dependencies
npm install

# Start Next.js development server
npm run dev
```

Frontend will be available at `http://localhost:3000`

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_image_endpoint.py -v
pytest tests/test_video_endpoint.py -v

# Run with coverage
pytest tests/ --cov=backend --cov=evaluation
```

## API Reference

### Health & Model Info

**`GET /health`** – Server status and model metadata
```json
{
  "status": "ok",
  "device": "cuda",
  "model": "hybrid",
  "model_path": "/app/models/hybrid_kaggle_finetuned.pt",
  "model_version": "kaggle-finetuned",
  "model_hash": "707e50e22417"
}
```

**`GET /model-info`** – Full model details including hash verification
```json
{
  "model_path": "/app/models/hybrid_kaggle_finetuned.pt",
  "model_version": "kaggle-finetuned",
  "model_hash": "707e50e224173fd...",
  "expected_model_hash": null,
  "hash_matches_expected": true,
  "device": "cuda"
}
```

### Image Analysis

**`POST /predict`** – Deepfake detection for single image

**Request:**
```
Content-Type: multipart/form-data
file: <image file> (JPG, PNG, WEBP)
```

**Response:**
```json
{
  "is_fake": true,
  "confidence": 0.6841,
  "calibrated_confidence": 0.6841,
  "risk_level": "CONFIDENT",
  "label": "DEEPFAKE",
  "heatmap_base64": "data:image/jpeg;base64,...",
  "spatial_importance": 72.5,
  "frequency_importance": 27.5
}
```

**Fields:**
- `is_fake` (bool): Final decision (true = deepfake)
- `confidence` (float): Display-friendly decision confidence [0, 1]
- `calibrated_confidence` (float): Raw calibrated model output
- `risk_level` (str): Confidence margin category (UNCERTAIN | BORDERLINE | CONFIDENT)
- `label` (str): Human-readable verdict (DEEPFAKE | REAL)
- `heatmap_base64` (str): Base64-encoded GradCAM visualization
- `spatial_importance` (%): Feature importance from spatial branch
- `frequency_importance` (%): Feature importance from frequency branch

### Video Analysis

**`POST /predict_video`** – Deepfake detection for video (sampled frame-by-frame)

**Request:**
```
Content-Type: multipart/form-data
file: <video file> (MP4)
```

**Response:**
```json
{
  "is_fake": true,
  "confidence": 0.6946,
  "calibrated_confidence": 0.3054,
  "risk_level": "UNCERTAIN",
  "label": "DEEPFAKE",
  "frame_confidences": [0.7215, 0.6841, 0.5932, ...],
  "top_frame_index": 0,
  "heatmap_base64": "data:image/jpeg;base64,...",
  "frames_analyzed": 5
}
```

**Fields:**
- `frame_confidences` (list): Per-frame calibrated confidence scores
- `top_frame_index` (int): Index of highest-confidence frame in sample
- `frames_analyzed` (int): Number of frames sampled (every 10th frame)
- Other fields: Same as image endpoint

## Architecture & Design

### Confidence Calibration Pipeline

The system uses **domain-specific calibration** for the Kaggle fine-tuned model:

```
Raw Model Output (sigmoid) [0, 1]
    ↓
Calibration Formula: calibrated = (raw × 1.3 - 0.08) ∈ [0, 1]
    ↓
Decision Threshold: 0.3 (if calibrated > 0.3 → DEEPFAKE)
    ↓
Risk Level Assignment:
  - distance < 0.1 from threshold → UNCERTAIN
  - distance < 0.2 from threshold → BORDERLINE
  - distance ≥ 0.2 from threshold → CONFIDENT
    ↓
Display Confidence: max(calibrated, 1 - calibrated)
```

### Model Architecture

**Hybrid Spatial + Frequency Fusion:**

```
Input Image (224×224)
    ├─ Spatial Branch (EfficientNet-B4)
    │   ├─ Block layers (feature extraction)
    │   └─ Output: Spatial features (512-dim)
    │
    ├─ Frequency Branch (FFT Analysis)
    │   ├─ 2D FFT → Power spectrum
    │   ├─ Log compression → Histogram features
    │   └─ Output: Frequency features (256-dim)
    │
    └─ Fusion Classifier
        ├─ Concatenate features [512 + 256]
        ├─ Dense layers with dropout
        └─ Sigmoid → [0, 1] confidence
```

### Test-Time Augmentation (TTA)

During inference, the model evaluates **4 augmented versions** of each image:
1. Original
2. Horizontal flip
3. Color jitter (brightness ±10%, contrast ±10%)
4. Rotation ±5°

Predictions are averaged; if variance exceeds threshold, risk_level → UNCERTAIN.

### GradCAM Visualization

- **Target layer**: EfficientNet block7 (spatial_features.7.1.block.3.0)
- **Face detection**: Haar cascade for face-guided masking
- **Elliptical masking**: Soft focus on detected face region (avoids corner bias)
- **Output**: Heatmap overlay on original image (base64-encoded JPEG)

## Environment Variables

### Backend Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | Auto-resolved | Path to model checkpoint (.pt file) |
| `MODEL_VERSION` | `kaggle-finetuned` | Model version tag |
| `EXPECTED_MODEL_HASH` | (unset) | Optional SHA-256 hash for verification |
| `DEVICE` | Auto-detect | `cuda` or `cpu` |

### Docker/Cloud Deployment

| Variable | Example | Description |
|----------|---------|-------------|
| `GCS_MODEL_URI` | `gs://bucket/models/hybrid_kaggle_finetuned.pt` | GCS path for model download |
| `MODEL_DIR` | `/app/models` | Container model directory |
| `MODEL_FILENAME` | `hybrid_kaggle_finetuned.pt` | Filename after download |

## Deployment

### Docker (Local)

```bash
# Build image
docker build -f backend/Dockerfile -t deepfake-detector:latest .

# Run container
docker run -p 8000:8080 \
  -e MODEL_PATH=/app/models/hybrid_kaggle_finetuned.pt \
  deepfake-detector:latest
```

### Google Cloud (Cloud Run / Cloud Build)

```bash
# Requires GCS model URI and service account permissions
gcloud builds submit --config cloudbuild.yaml
```

Substitutions in `cloudbuild.yaml`:
- `_IMAGE_URI`: Container registry path
- `_MODEL_VERSION`: Model version tag (default: `kaggle-finetuned`)

### Environment Setup (Production)

1. Store model in GCS (e.g., `gs://your-bucket/models/hybrid_kaggle_finetuned.pt`)
2. Ensure container has GCS read permissions (service account)
3. Set `GCS_MODEL_URI` env var in Cloud Run
4. Optional: Set `EXPECTED_MODEL_HASH` for verification

## Confidence Scoring Guide

### Understanding Risk Levels

- **CONFIDENT** (distance ≥ 0.2 from threshold)
  - Decision is clear and well-supported
  - Use for high-stakes decisions (automated blocking, flagging)
  
- **BORDERLINE** (distance 0.1–0.2 from threshold)
  - Some uncertainty, but lean is clear
  - Recommend human review for borderline cases
  
- **UNCERTAIN** (distance < 0.1 from threshold)
  - Very close to decision boundary
  - **Always** escalate to human review
  - Could indicate ambiguous image or model limitation

### Example Interpretations

| Raw | Calibrated | Threshold | Decision | Risk Level | Interpretation |
|-----|-----------|-----------|----------|------------|-----------------|
| 0.1 | 0.04 | 0.3 | REAL | CONFIDENT | Clearly authentic |
| 0.3 | 0.31 | 0.3 | DEEPFAKE | BORDERLINE | Slight deepfake signal, borderline |
| 0.35 | 0.35 | 0.3 | DEEPFAKE | UNCERTAIN | Very close call, needs review |
| 0.7 | 0.83 | 0.3 | DEEPFAKE | CONFIDENT | Clear deepfake detection |

## Benchmarks & Performance

### Endpoint Latency (Measured on CPU, April 24, 2026)

| Endpoint | Mean | Median | P95 | P99 |
|----------|------|--------|-----|-----|
| POST /predict | 4,077ms | 4,063ms | 4,243ms | 4,315ms |
| POST /predict_video (30f) | 7,100ms | 5,863ms | 11,627ms | 11,781ms |

### Throughput

| Scenario | Throughput | Notes |
|----------|-----------|-------|
| Single-thread `/predict` | 0.24 img/s | Baseline |
| Concurrent x4 `/predict` | 0.27 img/s | **Best throughput** |
| Video frame rate | 2.6 fps | End-to-end |

### Pipeline Breakdown (CPU)

| Stage | Mean | % of Total | Bottleneck |
|-------|------|-----------|-----------|
| Preprocessing | 13.8ms | 0.3% | — |
| Model forward | 1,125ms | 25% | — |
| **GradCAM** | **3,292ms** | **75%** | ⚠️ Primary |
| Post-processing | 0.3ms | <0.1% | — |

**Optimization Note:** GradCAM generation dominates latency. For real-time scenarios, consider disabling heatmaps or offloading to background jobs.

## Testing

### Unit & Integration Tests

```bash
# Image endpoint tests
pytest tests/test_image_endpoint.py -v

# Video endpoint tests
pytest tests/test_video_endpoint.py -v

# Confidence calibration
pytest tests/test_confidence_calibrator.py -v

# Model metadata & identity
pytest tests/test_model_metadata.py -v

# Full suite
pytest tests/ -v
```

### Benchmark Suite

```bash
# Inference latency (image + video + stage breakdown)
python tests/benchmark_inference_latency.py --warmup-runs 3 --image-runs 30 --video-runs 10

# Throughput under load
python tests/benchmark_throughput.py --concurrency 1 2 4 8 --duration 45

# Model component timing
python tests/benchmark_model_components.py --runs 30

# Dataset performance
python tests/benchmark_datasets.py --samples 600

# Capture baseline artifacts
python tests/capture_baseline_artifacts.py
```

Benchmark results are saved to `artifacts/` directory.

## Advanced Configuration

### Custom Confidence Calibration

To use different calibration parameters:

```python
from backend.confidence_calibrator import ConfidenceCalibrator

# Create with custom domain
cal = ConfidenceCalibrator(domain='kaggle')  # Default

# For FaceForensics++ pre-trained model:
cal_ff = ConfidenceCalibrator(domain='faceforensics')

# Get metrics
metrics = cal.get_metrics(raw_confidence=0.65)
```

### Test-Time Augmentation Control

In `backend/main.py`:

```python
PREDICTOR = EnhancedPredictor(
    MODEL, DEVICE,
    use_tta=True,              # Enable/disable TTA
    use_quality_check=True     # Enable/disable quality filtering
)
```

### Model Identity Verification

To enforce model hash verification at startup:

```bash
export EXPECTED_MODEL_HASH="707e50e224173fd8b..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Will fail if loaded model hash doesn't match
```

## Known Limitations & Future Work

### Current Limitations

1. **GradCAM Latency**: Visualization adds ~75% of total inference time
2. **Frame Sampling**: Video analysis samples every 10th frame (not real-time continuous)
3. **Face Detection**: Relies on Haar cascades; may miss non-frontal faces
4. **Cross-Domain**: 96.21% accuracy on Kaggle; performance varies on other datasets

### Roadmap

- [ ] Optimize GradCAM for real-time performance
- [ ] Add attention mechanism visualization
- [ ] Frame-by-frame temporal consistency analysis
- [ ] Ensemble with adversarial robustness
- [ ] Mobile/edge inference (ONNX export)
- [ ] Streaming video processing
- [ ] Multilingual interface

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes with clear messages
4. Push to branch and open a pull request
5. Ensure all tests pass (`pytest tests/ -v`)

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use this work in research, please cite:

```bibtex
@software{dualcore2026,
  title={DualCore: AI-Based Image Authenticity & Deepfake Detection Tool},
  author={Aafi, Mohd and Mishra, Girish},
  year={2026},
  url={https://github.com/Aafi04/HP-CoE-Agile-Challenge}
}
```

## Support & Contact

For issues, questions, or feature requests, please open an issue on GitHub or contact the project leads.

---

**Project Status:** ✅ Production Ready (April 22, 2026 stabilized)  
**Last Updated:** April 24, 2026  
**Model Accuracy:** 96.21% (Kaggle validation)
