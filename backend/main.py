import sys
sys.path.insert(0, '.')
import io
import os
import base64
import hashlib
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
from evaluation.gradcam import load_model
from evaluation.enhanced_gradcam import HybridGradCAM
from backend.inference_enhancements import EnhancedPredictor
from backend.confidence_calibrator import ConfidenceCalibrator

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global model state ---
MODEL = None
PREDICTOR = None  # Enhanced predictor with mitigations
CONFIDENCE_CAL = None  # Kaggle domain-specific calibration
GRADCAM_ANALYZER = None  # Enhanced dual-domain GradCAM
LOADED_MODEL_HASH = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_VERSION = os.getenv("MODEL_VERSION", "kaggle-finetuned")
EXPECTED_MODEL_HASH = os.getenv("EXPECTED_MODEL_HASH")


def _compute_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _resolve_model_path() -> str:
    configured_model_path = os.getenv("MODEL_PATH")
    if configured_model_path:
        return configured_model_path

    candidates = [
        "/app/models/hybrid_kaggle_finetuned.pt",
        os.path.join(BASE_DIR, "backend", "models", "hybrid_kaggle_finetuned.pt"),
        os.path.join(BASE_DIR, "models", "hybrid_kaggle_finetuned.pt"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No Kaggle-finetuned model found. Set MODEL_PATH or place hybrid_kaggle_finetuned.pt in a known location."
    )


MODEL_PATH = _resolve_model_path()
MODEL_TYPE = 'hybrid'
TRANSFORM = get_val_transforms(224)
FRAME_SAMPLE_RATE = 10

# Face detector for face-guided GradCAM.
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def detect_largest_face_bbox(img_rgb: np.ndarray):
    """
    Detect largest face and return expanded bbox as (x0, y0, x1, y1) in image coords.
    Returns None if no face is detected.
    """
    if FACE_CASCADE.empty():
        return None

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=6,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None

    img_h, img_w = img_rgb.shape[:2]
    img_cx = img_w * 0.5
    img_cy = img_h * 0.5

    # Prefer faces that are both large and close to image center.
    def _score(face):
        x, y, w, h = [int(v) for v in face]
        area_ratio = (w * h) / float(max(1, img_w * img_h))
        cx = x + (w * 0.5)
        cy = y + (h * 0.5)
        dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5
        dist_ratio = dist / float(max(1.0, (img_w ** 2 + img_h ** 2) ** 0.5))
        return area_ratio - (0.35 * dist_ratio)

    x, y, w, h = [int(v) for v in max(faces, key=_score)]

    # Moderate context expansion around detected face.
    mx = int(0.12 * w)
    my = int(0.15 * h)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(img_w, x + w + mx)
    y1 = min(img_h, y + h + my)

    # Guard against overly large boxes that re-include irrelevant corners.
    bbox_area_ratio = ((x1 - x0) * (y1 - y0)) / float(max(1, img_w * img_h))
    if bbox_area_ratio > 0.55:
        cx = int(x + (w * 0.5))
        cy = int(y + (h * 0.5))
        side = int(max(w, h) * 1.35)
        half = max(1, side // 2)
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(img_w, cx + half)
        y1 = min(img_h, cy + half)

    if x1 <= x0 or y1 <= y0:
        return None

    return (int(x0), int(y0), int(x1), int(y1))

class PredictionResponse(BaseModel):
    is_fake: bool
    confidence: float
    calibrated_confidence: float
    risk_level: str
    label: str
    heatmap_base64: Optional[str] = None
    spatial_importance: Optional[float] = None
    frequency_importance: Optional[float] = None

class VideoResponse(BaseModel):
    is_fake: bool
    confidence: float
    calibrated_confidence: Optional[float] = None
    risk_level: Optional[str] = None
    label: str
    frame_confidences: List[float]
    top_frame_index: int
    heatmap_base64: str
    frames_analyzed: int

@app.on_event("startup")
async def load_model_on_startup():
    global MODEL, PREDICTOR, CONFIDENCE_CAL, GRADCAM_ANALYZER, LOADED_MODEL_HASH
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    MODEL = load_model(MODEL_PATH, MODEL_TYPE, DEVICE)
    LOADED_MODEL_HASH = _compute_sha256(MODEL_PATH)
    if EXPECTED_MODEL_HASH and LOADED_MODEL_HASH.lower() != EXPECTED_MODEL_HASH.lower():
        raise RuntimeError(
            "Loaded model hash does not match EXPECTED_MODEL_HASH. "
            f"expected={EXPECTED_MODEL_HASH}, actual={LOADED_MODEL_HASH}"
        )
    PREDICTOR = EnhancedPredictor(
        MODEL, DEVICE,
        use_tta=True,  # Enable test-time augmentation
        use_quality_check=True  # Enable image quality filtering
    )
    # Initialize Kaggle domain-specific calibration
    CONFIDENCE_CAL = ConfidenceCalibrator(domain='kaggle')
    # Initialize enhanced dual-domain GradCAM
    GRADCAM_ANALYZER = HybridGradCAM(model=MODEL, device=DEVICE)
    print("Model and enhanced predictor loaded successfully.")
    print(f"Confidence calibration: Kaggle domain (fine-tuned model)")
    print(f"GradCAM: Enhanced dual-domain (spatial + frequency branches)")
    print(f"Face detector loaded: {not FACE_CASCADE.empty()}")
    print(f"Model version: {MODEL_VERSION}")
    print(f"Model hash: {LOADED_MODEL_HASH[:12]}")
    if EXPECTED_MODEL_HASH:
        print("Expected model hash verification: PASS")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "model_version": MODEL_VERSION,
        "model_hash": LOADED_MODEL_HASH[:12] if LOADED_MODEL_HASH else None,
        "expected_hash_set": EXPECTED_MODEL_HASH is not None,
    }


@app.get("/model-info")
async def model_info():
    return {
        "model_path": MODEL_PATH,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "model_hash": LOADED_MODEL_HASH,
        "expected_model_hash": EXPECTED_MODEL_HASH,
        "hash_matches_expected": (
            (EXPECTED_MODEL_HASH is None)
            or (LOADED_MODEL_HASH is not None and LOADED_MODEL_HASH.lower() == EXPECTED_MODEL_HASH.lower())
        ),
        "device": DEVICE,
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # Use enhanced predictor with domain shift mitigations
    _, raw_confidence, details = PREDICTOR.predict(
        image_tensor, image, return_details=True
    )

    # Apply the single source of truth for calibration and decisioning.
    calibration_metrics = CONFIDENCE_CAL.get_metrics(raw_confidence)
    calibrated_confidence = calibration_metrics['calibrated_confidence']
    risk_level = calibration_metrics['risk_level']
    is_fake = calibration_metrics['decision'] == 'DEEPFAKE'
    decision_confidence = calibration_metrics['decision_confidence']
    if details.get('tta_uncertain'):
        risk_level = 'UNCERTAIN'

    # FIXED: Use enhanced dual-domain GradCAM (spatial + frequency)
    # with proper normalization, ReLU, colormap, and alpha blending
    try:
        # Prepare original image as uint8 in 0-255 range for heatmap blending
        img_np = np.array(image)  # RGB, 0-255

        face_bbox = detect_largest_face_bbox(img_np)

        if face_bbox is not None:
            x0, y0, x1, y1 = face_bbox
            print(f"[GradCAM Debug] face bbox detected (x0,y0,x1,y1): {face_bbox}")

            face_crop_rgb = img_np[y0:y1, x0:x1]
            face_crop_pil = Image.fromarray(face_crop_rgb)
            face_tensor = TRANSFORM(face_crop_pil).unsqueeze(0).to(DEVICE)

            gradcam_result = GRADCAM_ANALYZER.generate_dual_visualization(
                face_tensor, face_crop_rgb, debug=True
            )
            face_heatmap = gradcam_result.get('spatial_heatmap', None)

            base_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            if face_heatmap is not None and face_heatmap.shape[:2] == (y1 - y0, x1 - x0):
                # Blend only inside an ellipse centered on the face crop to avoid corner leakage.
                roi_h, roi_w = face_heatmap.shape[:2]
                mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                center = (roi_w // 2, roi_h // 2)
                axes = (max(1, int(roi_w * 0.40)), max(1, int(roi_h * 0.47)))
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

                roi_orig = base_bgr[y0:y1, x0:x1]
                roi_blend = roi_orig.copy()
                m = (mask.astype(np.float32) / 255.0)[..., None]
                roi_blend = (face_heatmap.astype(np.float32) * m + roi_orig.astype(np.float32) * (1.0 - m)).astype(np.uint8)
                base_bgr[y0:y1, x0:x1] = roi_blend
                heatmap = base_bgr
            else:
                # Fallback to full-image GradCAM if crop composition fails.
                gradcam_result = GRADCAM_ANALYZER.generate_dual_visualization(
                    image_tensor, img_np, debug=True
                )
                heatmap = gradcam_result.get('spatial_heatmap', None)
        else:
            print("[GradCAM Debug] no face bbox detected, using full-image GradCAM")
            gradcam_result = GRADCAM_ANALYZER.generate_dual_visualization(
                image_tensor, img_np, debug=True
            )
            heatmap = gradcam_result.get('spatial_heatmap', None)

        spatial_importance = gradcam_result.get('spatial_importance', 0)
        frequency_importance = gradcam_result.get('freq_importance', 0)
        gradcam_debug = gradcam_result.get('gradcam_debug', {})

        if gradcam_debug:
            print(f"[GradCAM Debug] layer used: {gradcam_debug.get('target_layer_name')}")
            print(f"[GradCAM Debug] gradient shape: {gradcam_debug.get('gradient_shape_before_pooling')}")
            print(
                "[GradCAM Debug] cam min/max before normalization: "
                f"{gradcam_debug.get('cam_min_before_normalization')}, "
                f"{gradcam_debug.get('cam_max_before_normalization')}"
            )
            print(f"[GradCAM Debug] raw CAM argmax (y, x): {gradcam_debug.get('raw_cam_argmax_yx')}")
            print(f"[GradCAM Debug] image argmax (x, y): {gradcam_debug.get('argmax_image_xy')}")
            print(f"[GradCAM Debug] face center bbox (x0, y0, x1, y1): {gradcam_debug.get('face_center_bbox_xyxy')}")
            print(f"[GradCAM Debug] argmax in face center bbox: {gradcam_debug.get('argmax_in_face_center_bbox')}")
    except Exception as e:
        # Fallback if enhanced GradCAM fails
        print(f"Enhanced GradCAM failed: {e}, using fallback")
        heatmap = None
        spatial_importance = 0.5
        frequency_importance = 0.5

    if heatmap is not None:
        # Heatmap is already in BGR format from OpenCV operations
        # Just encode directly to base64
        _, buffer = cv2.imencode('.jpg', heatmap)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
    else:
        heatmap_b64 = None

    label = "DEEPFAKE" if is_fake else "REAL"

    return PredictionResponse(
        is_fake=is_fake,
        confidence=round(decision_confidence, 4),
        calibrated_confidence=round(calibrated_confidence, 4),
        risk_level=risk_level,
        label=label,
        heatmap_base64=heatmap_b64,
        spatial_importance=round(spatial_importance * 100, 1) if spatial_importance else None,
        frequency_importance=round(frequency_importance * 100, 1) if frequency_importance else None
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
        frame_raw_confidences = []
        frame_index = 0
        sampled_frame_index = 0
        top_confidence = -1.0
        top_frame_idx = 0
        top_heatmap = None
        video_tta_uncertain = False
        
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
                
                # Apply transform and run through the same predictor path as image endpoint.
                image_tensor = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
                _, raw_confidence, frame_details = PREDICTOR.predict(
                    image_tensor, pil_image, return_details=True
                )

                frame_metrics = CONFIDENCE_CAL.get_metrics(raw_confidence)
                calibrated_confidence = float(frame_metrics['calibrated_confidence'])
                frame_confidences.append(round(calibrated_confidence, 4))
                frame_raw_confidences.append(float(raw_confidence))
                if frame_details.get('tta_uncertain'):
                    video_tta_uncertain = True

                try:
                    gradcam_result = GRADCAM_ANALYZER.generate_dual_visualization(
                        image_tensor, frame_rgb, debug=False
                    )
                    frame_heatmap = gradcam_result.get('spatial_heatmap', None)
                except Exception:
                    frame_heatmap = None

                if frame_heatmap is None:
                    frame_heatmap = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Track highest confidence frame
                if calibrated_confidence > top_confidence:
                    top_confidence = calibrated_confidence
                    top_frame_idx = sampled_frame_index
                    top_heatmap = frame_heatmap
                
                sampled_frame_index += 1
            
            frame_index += 1
        
        cap.release()
        
        if not frame_confidences:
            raise HTTPException(status_code=400, detail="No frames could be extracted from video.")
        
        # Encode top heatmap (already BGR in this code path)
        _, buffer = cv2.imencode('.jpg', top_heatmap)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

        # Compute overall decision from averaged raw confidence and shared calibrator.
        mean_raw_confidence = float(np.mean(frame_raw_confidences))
        overall_metrics = CONFIDENCE_CAL.get_metrics(mean_raw_confidence)
        is_fake_overall = overall_metrics['decision'] == 'DEEPFAKE'
        label = overall_metrics['decision']
        risk_level = overall_metrics['risk_level']
        if video_tta_uncertain:
            risk_level = 'UNCERTAIN'
        
        return VideoResponse(
            is_fake=is_fake_overall,
            confidence=round(overall_metrics['decision_confidence'], 4),
            calibrated_confidence=round(overall_metrics['calibrated_confidence'], 4),
            risk_level=risk_level,
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
