#!/bin/bash
set -e

echo "Downloading model from GCS..."
mkdir -p /app/checkpoints
gsutil cp gs://deepfake-detection-data-2026/models/hybrid_best.pt /app/checkpoints/hybrid_best.pt

echo "Starting server..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8080
