import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from models.hybrid_model import HybridDeepfakeDetector
from models.efficientnet import DeepfakeEfficientNet

def load_model(model_path, model_type='hybrid', device='cuda'):
    if model_type == 'hybrid':
        model = HybridDeepfakeDetector(num_classes=1, pretrained=False)
    else:
        model = DeepfakeEfficientNet(num_classes=1, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    model.to(device)
    return model

def get_gradcam_target_layer(model, model_type='hybrid'):
    if model_type == 'hybrid':
        # Last conv block of EfficientNet spatial branch
        return [model.spatial_features[-1]]
    else:
        return [model.features[-1]]

def generate_gradcam(model, image_tensor, model_type='hybrid', device='cuda'):
    """
    image_tensor: (1, 3, 224, 224) normalized tensor
    Returns:
        confidence: float, probability of being deepfake
        heatmap_overlay: numpy array (224, 224, 3) BGR for display
        is_fake: bool
    """
    target_layers = get_gradcam_target_layer(model, model_type)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [BinaryClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]  # (224, 224)

    # Get confidence score
    with torch.no_grad():
        output = model(image_tensor.to(device))
        confidence = torch.sigmoid(output).item()

    # Denormalize image for overlay
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1).astype(np.float32)

    # Overlay heatmap
    heatmap_overlay = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    return confidence, heatmap_overlay, confidence > 0.5

def gradcam_from_path(image_path, model_path, model_type='hybrid', device='cuda'):
    """
    Convenience function: takes image path, returns results.
    """
    from PIL import Image
    from data.augmentations import get_val_transforms

    transform = get_val_transforms(224)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model = load_model(model_path, model_type, device)
    return generate_gradcam(model, image_tensor, model_type, device)
