"""
Backend improvements for domain shift mitigation
================================================

Quick fixes that don't require retraining:
1. Threshold adjustment (confidence calibration)
2. Test-time augmentation (TTA)
3. Image quality filtering
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T


# Confidence calibration - lowered threshold for Kaggle domain
CONFIDENCE_THRESHOLD = 0.3  # Default: 0.5, Adjusted for domain shift
ENABLE_TTA = True  # Test-time augmentation
ENABLE_QUALITY_CHECK = True  # Image quality filtering

# TTA augmentations
TTA_TRANSFORMS = [
    T.Compose([
        T.Lambda(lambda x: x),  # Identity
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    T.Compose([
        T.RandomHorizontalFlip(p=1.0),  # Horizontal flip
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
]


def check_image_quality(image_pil):
    """
    Check if image quality is sufficient for reliable prediction.
    
    Returns:
        is_valid: bool
        reason: str (explanation if invalid)
    """
    try:
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Check image size
        h, w = img_cv.shape[:2]
        if h < 64 or w < 64:
            return False, f"Image too small: {w}x{h}"
        
        # Check brightness (prevent pure black/white images)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 30 or mean_brightness > 225:
            return False, f"Suspicious brightness: {mean_brightness:.0f}"
        
        # Check contrast (prevent uniform images)
        contrast = np.std(gray)
        if contrast < 10:
            return False, f"Low contrast: {contrast:.1f}"
        
        # Check for valid color distribution
        b, g, r = cv2.split(img_cv)
        if np.std([np.mean(b), np.mean(g), np.mean(r)]) < 5:
            return False, "Suspicious color distribution (nearly grayscale)"
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Quality check error: {str(e)}"


def predict_with_tta(model, image_tensor, device, transforms_list=None):
    """
    Predict with test-time augmentation.
    
    Applies multiple transformations to input and averages predictions.
    Provides confidence estimate via standard deviation.
    
    Args:
        model: PyTorch model
        image_tensor: (1, 3, 224, 224) normalized tensor
        device: 'cpu' or 'cuda'
        transforms_list: list of augmentation functions
        
    Returns:
        mean_confidence: float
        std_confidence: float (certainty measure)
        predictions: list of individual predictions
    """
    if transforms_list is None:
        transforms_list = TTA_TRANSFORMS
    
    predictions = []
    
    with torch.no_grad():
        # Original prediction
        output = model(image_tensor.to(device))
        confidence = torch.sigmoid(output).item()
        predictions.append(confidence)
        
        # Augmented predictions
        # Horizontal flip
        flipped = torch.flip(image_tensor, [-1])
        output = model(flipped.to(device))
        confidence = torch.sigmoid(output).item()
        predictions.append(confidence)
        
        # Vertical flip
        flipped_v = torch.flip(image_tensor, [-2])
        output = model(flipped_v.to(device))
        confidence = torch.sigmoid(output).item()
        predictions.append(confidence)
    
    predictions = np.array(predictions)
    mean_confidence = np.mean(predictions)
    std_confidence = np.std(predictions)
    
    return mean_confidence, std_confidence, predictions


def confidence_calibration(raw_confidence, method='scale'):
    """
    Apply confidence calibration to account for domain shift.
    
    The model is overly conservative on Kaggle images (predicts "real" for everything).
    This function adjusts confidence to compensate.
    
    Args:
        raw_confidence: Model output after sigmoid (0-1)
        method: 'scale' (multiply), 'shift' (subtract offset), or 'none'
        
    Returns:
        calibrated_confidence: Adjusted confidence
    """
    if method == 'shift':
        # Shift: if model gives 0.1, adjust to "fake" (higher score)
        # Idea: model tends to be too low, shift everything up slightly
        return np.clip(raw_confidence * 1.5 - 0.1, 0, 1)
    
    elif method == 'scale':
        # Scale: confidence is compressed, spread it out
        # Center around 0.5 and expand variance
        return np.clip((raw_confidence - 0.5) * 1.3 + 0.5, 0, 1)
    
    else:
        return raw_confidence


class EnhancedPredictor:
    """
    Wrapper around model inference with domain shift mitigations.
    
    Combines:
    - Image quality checking
    - Test-time augmentation
    - Confidence calibration
    - Lowered threshold
    """
    
    def __init__(self, model, device='cpu', use_tta=True, use_quality_check=True):
        self.model = model
        self.device = device
        self.use_tta = use_tta
        self.use_quality_check = use_quality_check
    
    def predict(self, image_tensor, image_pil=None, return_details=False):
        """
        Predict with all mitigations applied.
        
        Args:
            image_tensor: (1, 3, 224, 224) normalized tensor
            image_pil: PIL Image (for quality check)
            return_details: If True, return confidence and details
            
        Returns:
            is_fake: bool
            confidence: float (0-1)
            details: dict (if return_details=True)
        """
        details = {}
        
        # Step 1: Quality check
        if self.use_quality_check and image_pil is not None:
            is_valid, reason = check_image_quality(image_pil)
            details['quality_check'] = {'valid': is_valid, 'reason': reason}
            if not is_valid:
                # If quality is bad, be more conservative
                # Flag as uncertain ("SUSPICIOUS")
                details['warning'] = f"Poor image quality: {reason}"
        
        # Step 2: Get prediction (with or without TTA)
        if self.use_tta:
            mean_conf, std_conf, predictions = predict_with_tta(
                self.model, image_tensor, self.device
            )
            confidence = mean_conf
            details['tta_predictions'] = predictions.tolist()
            details['tta_std'] = float(std_conf)
        else:
            with torch.no_grad():
                output = self.model(image_tensor.to(self.device))
                confidence = torch.sigmoid(output).item()
        
        # Step 3: Confidence calibration
        raw_confidence = confidence
        confidence = confidence_calibration(confidence, method='shift')
        details['raw_confidence'] = float(raw_confidence)
        details['calibrated_confidence'] = float(confidence)
        
        # Step 4: Decision with adjusted threshold
        is_fake = confidence > CONFIDENCE_THRESHOLD
        
        details['threshold'] = CONFIDENCE_THRESHOLD
        details['decision'] = 'DEEPFAKE' if is_fake else 'REAL'
        
        if return_details:
            return is_fake, confidence, details
        else:
            return is_fake, confidence


# For backward compatibility, create single functions
def simple_predict(model, image_tensor, device='cpu'):
    """Simple prediction without enhancements (backward compatible)"""
    with torch.no_grad():
        output = model(image_tensor.to(device))
        confidence = torch.sigmoid(output).item()
    return confidence > CONFIDENCE_THRESHOLD, confidence


def enhanced_predict(model, image_tensor, image_pil, device='cpu'):
    """Enhanced prediction with all mitigations"""
    predictor = EnhancedPredictor(
        model, device,
        use_tta=ENABLE_TTA,
        use_quality_check=ENABLE_QUALITY_CHECK
    )
    return predictor.predict(image_tensor, image_pil, return_details=False)
