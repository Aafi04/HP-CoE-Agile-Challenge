import sys
sys.path.insert(0, '.')
import io
import os
import base64
import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from data.augmentations import get_val_transforms
from evaluation.gradcam import load_model, generate_gradcam

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global model state ---
MODEL = None
DEVICE = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.environ.get('MODEL_PATH', 'checkpoints/hybrid_best.pt')
MODEL_TYPE = 'hybrid'
TRANSFORM = get_val_transforms(224)

class PredictionResponse(BaseModel):
    is_fake: bool
    confidence: float
    label: str
    heatmap_base64: Optional[str] = None

@app.on_event("startup")
async def load_model_on_startup():
    global MODEL
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    MODEL = load_model(MODEL_PATH, MODEL_TYPE, DEVICE)
    print("Model loaded successfully.")

@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_TYPE}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    confidence, heatmap, is_fake = generate_gradcam(
        MODEL, image_tensor, MODEL_TYPE, DEVICE
    )

    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', heatmap_bgr)
    heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

    label = "DEEPFAKE" if is_fake else "REAL"

    return PredictionResponse(
        is_fake=is_fake,
        confidence=round(confidence, 4),
        label=label,
        heatmap_base64=heatmap_b64
    )
