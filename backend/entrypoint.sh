#!/bin/bash
set -e

MODEL_DIR="${MODEL_DIR:-/app/models}"
MODEL_FILENAME="${MODEL_FILENAME:-hybrid_kaggle_finetuned.pt}"
GCS_MODEL_URI="${GCS_MODEL_URI:-gs://deepfake-detection-data-2026/models/hybrid_kaggle_finetuned.pt}"
MODEL_VERSION="${MODEL_VERSION:-kaggle-finetuned}"

echo "Preparing model directory..."
mkdir -p "$MODEL_DIR"

echo "Downloading model from GCS: $GCS_MODEL_URI"
MODEL_PATH="$MODEL_DIR/$MODEL_FILENAME"
gsutil cp "$GCS_MODEL_URI" "$MODEL_PATH"

if [ ! -s "$MODEL_PATH" ]; then
	echo "Downloaded model file is missing or empty: $MODEL_PATH"
	exit 1
fi

ACTUAL_MODEL_HASH="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
echo "Downloaded model hash: ${ACTUAL_MODEL_HASH}"

if [ -n "${EXPECTED_MODEL_HASH:-}" ] && [ "${ACTUAL_MODEL_HASH}" != "${EXPECTED_MODEL_HASH}" ]; then
	echo "Model hash mismatch. expected=${EXPECTED_MODEL_HASH} actual=${ACTUAL_MODEL_HASH}"
	exit 1
fi

export MODEL_PATH
export MODEL_VERSION
export EXPECTED_MODEL_HASH="${EXPECTED_MODEL_HASH:-}"
echo "Using model path: $MODEL_PATH"

echo "Starting server..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8080
