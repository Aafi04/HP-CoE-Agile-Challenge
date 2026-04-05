import sys
sys.path.insert(0, '.')
import io
import os
import base64
import tempfile
import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
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
DEVICE = "cpu"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "backend", "models", "hybrid_full_best.pt")
MODEL_TYPE = 'hybrid'
TRANSFORM = get_val_transforms(224)
FRAME_SAMPLE_RATE = 10

class PredictionResponse(BaseModel):
    is_fake: bool
    confidence: float
    label: str
    heatmap_base64: Optional[str] = None

class VideoResponse(BaseModel):
    is_fake: bool
    confidence: float
    label: str
    frame_confidences: List[float]
    top_frame_index: int
    heatmap_base64: str
    frames_analyzed: int

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

@app.post("/predict_video", response_model=VideoResponse)
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are supported.")
    
    temp_file = None
    try:
        # Save video to temp file
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            temp_file = tmp.name
        
        # Open video
        cap = cv2.VideoCapture(temp_file)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not read video file.")
        
        frame_confidences = []
        frame_index = 0
        sampled_frame_index = 0
        top_confidence = 0
        top_frame_idx = 0
        top_heatmap = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every FRAME_SAMPLE_RATE frames
            if frame_index % FRAME_SAMPLE_RATE == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Apply transform and run through model
                image_tensor = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
                confidence, heatmap, is_fake = generate_gradcam(
                    MODEL, image_tensor, MODEL_TYPE, DEVICE
                )
                
                frame_confidences.append(round(confidence, 4))
                
                # Track highest confidence frame
                if confidence > top_confidence:
                    top_confidence = confidence
                    top_frame_idx = sampled_frame_index
                    top_heatmap = heatmap
                
                sampled_frame_index += 1
            
            frame_index += 1
        
        cap.release()
        
        if not frame_confidences:
            raise HTTPException(status_code=400, detail="No frames could be extracted from video.")
        
        # Encode top heatmap
        heatmap_bgr = cv2.cvtColor(top_heatmap, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', heatmap_bgr)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Compute mean confidence
        mean_confidence = round(np.mean(frame_confidences), 4)
        is_fake_overall = mean_confidence > 0.5
        label = "DEEPFAKE" if is_fake_overall else "REAL"
        
        return VideoResponse(
            is_fake=is_fake_overall,
            confidence=mean_confidence,
            label=label,
            frame_confidences=frame_confidences,
            top_frame_index=top_frame_idx,
            heatmap_base64=heatmap_b64,
            frames_analyzed=sampled_frame_index
        )
    
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
