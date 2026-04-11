# AI-Based Image Authenticity & Deepfake Detection Tool

## Overview

This project develops a state-of-the-art deep learning solution for detecting and analyzing deepfakes and evaluating image authenticity. It combines multiple neural network architectures including EfficientNet, FFT analysis, and fusion models to provide robust deepfake detection capabilities.

## Project Structure

```
├── /data/              – Dataset scripts and preprocessing utilities
├── /models/            – Model definitions (EfficientNet, FFT branch, fusion)
├── /training/          – Training loops and hyperparameter configurations
├── /inference/         – Standalone inference pipeline with temperature scaling
├── /backend/           – FastAPI application and Docker configuration
├── /frontend/          – Next.js web application
├── /evaluation/        – Metrics, benchmark scripts, and GradCAM visualization
├── /notebooks/         – Colab/Jupyter experiments and exploration
├── /tests/             – Pytest unit tests
├── README.md           – Project documentation
├── requirements.txt    – Python dependencies
└── .gitignore         – Git ignore rules
```

## Features

- Multi-architecture neural network approach (EfficientNet, FFT analysis)
- Model fusion for improved detection accuracy
- Temperature scaling for confidence calibration
- Standalone inference pipeline
- RESTful API backend with FastAPI
- Interactive web interface with Next.js
- Comprehensive evaluation and benchmarking tools
- GradCAM visualization for model interpretability

## Getting Started

(Documentation in progress)

## Technology Stack

- **ML Framework:** PyTorch
- **API:** FastAPI
- **Frontend:** Next.js
- **Containerization:** Docker
- **Testing:** pytest

## 📚 Documentation & Reports

**📋 Comprehensive Project Documentation:**
See [`../significant-markdowns/`](../significant-markdowns/) for detailed reports and working documents:

- **FINAL_REPORT.md** - Complete project report with all findings, architecture, and metrics
- **SESSION_SUMMARY_APR11.md** - Detailed work session notes with bug fixes and solutions
- **API_TESTING_SUMMARY.md** - API endpoint testing results and validation
- **PROJECT_STATUS.md** - Phase tracking and progress monitoring
- **DOCUMENTATION_INDEX.md** - Quick reference guide and navigation

For complete project status, architecture details, and testing results, start with [`FINAL_REPORT.md`](../significant-markdowns/FINAL_REPORT.md).

## Contributing

(Contribution guidelines in progress)

## License

(License information to be added)
