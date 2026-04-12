"""
Enhanced GradCAM for Hybrid Model
==================================
Visualize both spatial and frequency branch contributions.
Shows which domain (spatial CNN vs FFT) is most important for each prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple
from models.hybrid_model import HybridDeepfakeDetector
from models.efficientnet import DeepfakeEfficientNet


class HybridGradCAM:
    """Enhanced GradCAM for hybrid spatial + frequency models"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.target_layer_name, self.target_layer, self.target_layer_candidates = self._select_spatial_target_layer()

        # Print layer diagnostics once at startup to verify target layer selection.
        spatial_tail = self._get_spatial_module_tail()
        print(f"[GradCAM] spatial layers[-10:-1]: {spatial_tail}")
        print(f"[GradCAM] selected target layer: {self.target_layer_name}")

    def _get_spatial_module_tail(self) -> List[str]:
        """PyTorch equivalent of model.layers[-10:-1] for the spatial branch."""
        if not hasattr(self.model, 'spatial_features'):
            return []

        names = [f"spatial_features.{name}" for name, _ in self.model.spatial_features.named_modules() if name]
        if len(names) <= 1:
            return names

        start = max(0, len(names) - 10)
        end = max(0, len(names) - 1)
        return names[start:end]

    def _select_spatial_target_layer(self) -> Tuple[str, nn.Module, List[str]]:
        """
        Select the last spatial Conv2d before global pooling.

        For torchvision EfficientNet, this maps to top conv (spatial_features.8.0),
        with block7 project conv as the immediate fallback candidate.
        """
        if not hasattr(self.model, 'spatial_features'):
            raise RuntimeError("Model has no spatial_features branch for GradCAM.")

        conv_candidates: List[Tuple[str, nn.Module]] = []
        for name, module in self.model.spatial_features.named_modules():
            if isinstance(module, nn.Conv2d):
                full_name = f"spatial_features.{name}"
                # Exclude SE squeeze/excitation FC convs; they are not spatial feature maps.
                if ".block.2.fc" in full_name:
                    continue
                conv_candidates.append((full_name, module))

        if not conv_candidates:
            raise RuntimeError("No spatial Conv2d candidates found for GradCAM.")

        selected_name, selected_layer = conv_candidates[-1]

        # EfficientNet-specific preference: block7 project conv is often more localized
        # than top conv for GradCAM (equivalent to block7a_project_conv style targeting).
        preferred_suffixes = ["7.1.block.3.0", "7.0.block.3.0"]
        for suffix in preferred_suffixes:
            for name, layer in reversed(conv_candidates):
                if name.endswith(suffix):
                    selected_name, selected_layer = name, layer
                    break
            if selected_name.endswith(suffix):
                break

        return selected_name, selected_layer, [name for name, _ in conv_candidates]
    
    def _create_proper_heatmap(self, cam, original_img):
        """
        Create a properly formatted heatmap with correct colormap and normalization.
        
        Fixes:
        1. Apply ReLU to eliminate negative values
        2. Proper normalization to [0, 1]
        3. Correct colormap (blue=low, red=high)
        4. Resize to match original image
        5. Alpha blending at 0.5
        """
        # Step 1: Apply ReLU to eliminate negative values
        cam = np.maximum(cam, 0)
        
        # Step 2: Resize CAM to match original image dimensions
        img_h, img_w = original_img.shape[:2]
        cam_resized = cv2.resize(cam, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        
        # Step 3: Proper normalization to [0, 1]
        cam_min = cam_resized.min()
        cam_max = cam_resized.max()
        if cam_max - cam_min > 1e-8:
            cam_normalized = (cam_resized - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam_normalized = cam_resized
        
        # Step 4: Scale to uint8 [0, 255]
        cam_uint8 = (cam_normalized * 255).astype(np.uint8)
        
        # Step 5: Apply JET colormap (blue=low, red=high) and ensure not inverted
        heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        # Convert original image to BGR if needed (ensure it's uint8)
        if original_img.dtype != np.uint8:
            original_img = (original_img * 255).astype(np.uint8)
        if original_img.shape[2] == 3 and len(original_img.shape) == 3:
            # Assume RGB, convert to BGR for OpenCV
            original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        else:
            original_img_bgr = original_img
        
        # Step 6: Correct alpha blending (0.5 opacity for both)
        superimposed = cv2.addWeighted(original_img_bgr, 0.5, heatmap_colored, 0.5, 0)
        
        return superimposed
    
    def get_spatial_gradcam(self, image_tensor, target_class=None, debug=False, return_debug=False):
        """Get GradCAM from spatial (EfficientNet) branch
        
        target_class: which class to visualize (0=REAL, 1=DEEPFAKE)
                      if None, uses model's prediction
        """
        image_tensor = image_tensor.to(self.device)
        image_tensor = image_tensor.requires_grad_(True)

        activations: Dict[str, torch.Tensor] = {}
        gradients: Dict[str, torch.Tensor] = {}

        def _forward_hook(_, __, output):
            activations['value'] = output

        def _backward_hook(_, __, grad_output):
            gradients['value'] = grad_output[0]

        fwd_handle = self.target_layer.register_forward_hook(_forward_hook)
        bwd_handle = self.target_layer.register_full_backward_hook(_backward_hook)

        try:
            self.model.zero_grad(set_to_none=True)

            # Forward pass must happen before backward while hooks are active.
            output = self.model(image_tensor)
            pred_score = torch.sigmoid(output).item()
            if target_class is None:
                target_class = 1 if pred_score > 0.5 else 0

            # Binary classifier has one logit: class-1 uses +logit, class-0 uses -logit.
            logit = output.view(-1)[0]
            target_score = logit if target_class == 1 else -logit
            target_score.backward()

            grads_val = gradients.get('value')
            acts_val = activations.get('value')
            if grads_val is None or acts_val is None:
                raise RuntimeError("GradCAM hooks did not capture gradients/activations.")

            # Global average pooling over N,H,W gives per-channel weights.
            grads_shape = tuple(grads_val.shape)
            pooled_grads = torch.mean(grads_val, dim=(0, 2, 3))
            pooled_grads = torch.clamp(pooled_grads, min=0.0)

            # Weighted sum across channels.
            cam_tensor = torch.sum(acts_val[0] * pooled_grads[:, None, None], dim=0)
            raw_cam = cam_tensor.detach().cpu().numpy()
            raw_argmax_y, raw_argmax_x = np.unravel_index(np.argmax(raw_cam), raw_cam.shape)

            # ReLU on final CAM before normalization.
            spatial_cam = np.maximum(raw_cam, 0)
            cam_min = float(spatial_cam.min())
            cam_max = float(spatial_cam.max())

            debug_info: Dict[str, Any] = {
                'target_layer_name': self.target_layer_name,
                'spatial_layers_tail': self._get_spatial_module_tail(),
                'gradient_shape_before_pooling': grads_shape,
                'cam_min_before_normalization': cam_min,
                'cam_max_before_normalization': cam_max,
                'raw_cam_argmax_yx': (int(raw_argmax_y), int(raw_argmax_x)),
                'raw_cam_shape': tuple(int(v) for v in raw_cam.shape),
                'target_class': int(target_class),
            }

            if debug:
                print(f"[GradCAM Debug] target layer: {debug_info['target_layer_name']}")
                print(f"[GradCAM Debug] spatial layers[-10:-1]: {debug_info['spatial_layers_tail']}")
                print(f"[GradCAM Debug] gradient shape before pooling: {debug_info['gradient_shape_before_pooling']}")
                print(
                    "[GradCAM Debug] cam min/max before normalization: "
                    f"{debug_info['cam_min_before_normalization']:.6f}, "
                    f"{debug_info['cam_max_before_normalization']:.6f}"
                )
                print(f"[GradCAM Debug] raw CAM argmax (y, x): {debug_info['raw_cam_argmax_yx']}")

            if return_debug:
                return spatial_cam, debug_info

            return spatial_cam
        finally:
            fwd_handle.remove()
            bwd_handle.remove()
    
    def get_branch_importance(self, image_tensor):
        """
        Measure importance of spatial vs frequency branch
        Returns: (spatial_importance, freq_importance) where they sum to 1.0
        """
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad = True
        
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)

            # Forward pass
            spatial_features = self.model.spatial_features(image_tensor)
            spatial_pool = self.model.spatial_pool(spatial_features)
            spatial_flat = torch.flatten(spatial_pool, 1)
            spatial_flat.retain_grad()
            
            freq_features = self.model.fft_branch(image_tensor)
            freq_features.retain_grad()
            
            # Fused input to classifier
            fused = torch.cat([spatial_flat, freq_features], dim=1)
            output = self.model.classifier(fused)
            
            # Compute gradients w.r.t. both branches
            output.backward(retain_graph=True)
            
            # Measure gradient magnitude for each branch
            spatial_grad = spatial_flat.grad
            freq_grad = freq_features.grad
            
            if spatial_grad is not None and freq_grad is not None:
                spatial_importance_mag = spatial_grad.norm(p=2).item()
                freq_importance_mag = freq_grad.norm(p=2).item()
                
                total = spatial_importance_mag + freq_importance_mag
                if total > 0:
                    spatial_imp = spatial_importance_mag / total
                    freq_imp = freq_importance_mag / total
                else:
                    spatial_imp = 0.5
                    freq_imp = 0.5
            else:
                spatial_imp = 0.5
                freq_imp = 0.5
        
        return spatial_imp, freq_imp
    
    def analyze_fft_attention(self, image_tensor):
        """
        Analyze where FFT branch is focusing
        Returns attention pattern in frequency domain
        """
        # This is advanced - shows what patterns FFT learned to focus on
        image_tensor = image_tensor.to(self.device)
        
        # Extract FFT features at intermediate layers
        gray = 0.299 * image_tensor[:, 0] + 0.587 * image_tensor[:, 1] + 0.114 * image_tensor[:, 2]
        gray = gray.unsqueeze(1)
        
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        magnitude = torch.log1p(magnitude)
        
        # Normalize
        b = magnitude.shape[0]
        mag_min = magnitude.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        mag_max = magnitude.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)
        
        # Get FFT features
        freq_features = self.model.fft_branch.cnn(magnitude)
        
        # Average across spatial dims to see feature map activations
        freq_activation = freq_features.mean(dim=(2, 3))  # (B, 64)
        
        return freq_activation.detach().cpu().numpy()
    
    def generate_dual_visualization(self, image_tensor, img_orig_np, debug=False):
        """
        Generate combined visualization showing:
        1. Spatial branch GradCAM (targeted to actual prediction)
        2. Branch importance
        3. Frequency domain analysis
        
        All visualizations use proper normalization, ReLU, colormap, and blending
        """
        image_tensor = image_tensor.to(self.device)
        
        # Get prediction FIRST to determine which class to visualize
        with torch.no_grad():
            output = self.model(image_tensor)
            confidence = torch.sigmoid(output).item()
            target_class = 1 if confidence > 0.5 else 0
        
        # Get spatial GradCAM targeted to the PREDICTED class
        spatial_cam, gradcam_debug = self.get_spatial_gradcam(
            image_tensor,
            target_class=target_class,
            debug=debug,
            return_debug=True,
        )
        
        # Get importance scores
        spatial_imp, freq_imp = self.get_branch_importance(image_tensor)
        
        # Get FFT analysis
        fft_activation = self.analyze_fft_attention(image_tensor)
        
        # Create properly formatted spatial heatmap overlay
        # This applies all the fixes: ReLU, normalization, correct colormap, alpha blending
        if img_orig_np is not None:
            spatial_heatmap = self._create_proper_heatmap(spatial_cam, img_orig_np)

            # Map raw CAM argmax to image coordinates and verify center-40% bbox.
            img_h, img_w = img_orig_np.shape[:2]
            cam_h, cam_w = spatial_cam.shape
            argmax_y, argmax_x = gradcam_debug['raw_cam_argmax_yx']

            if cam_h > 1 and cam_w > 1:
                argmax_x_img = int(round(argmax_x * (img_w - 1) / (cam_w - 1)))
                argmax_y_img = int(round(argmax_y * (img_h - 1) / (cam_h - 1)))
            else:
                argmax_x_img = int(argmax_x)
                argmax_y_img = int(argmax_y)

            x0, x1 = int(img_w * 0.3), int(img_w * 0.7)
            y0, y1 = int(img_h * 0.3), int(img_h * 0.7)
            argmax_in_face_center = (x0 <= argmax_x_img <= x1) and (y0 <= argmax_y_img <= y1)

            gradcam_debug.update({
                'argmax_image_xy': (argmax_x_img, argmax_y_img),
                'face_center_bbox_xyxy': (x0, y0, x1, y1),
                'argmax_in_face_center_bbox': argmax_in_face_center,
            })

            if debug:
                print(f"[GradCAM Debug] argmax mapped to image (x, y): {(argmax_x_img, argmax_y_img)}")
                print(f"[GradCAM Debug] face center bbox (x0, y0, x1, y1): {(x0, y0, x1, y1)}")
                print(f"[GradCAM Debug] argmax in face center bbox: {argmax_in_face_center}")
        else:
            spatial_heatmap = None
        
        return {
            'spatial_heatmap': spatial_heatmap,
            'spatial_importance': spatial_imp,
            'freq_importance': freq_imp,
            'confidence': confidence,
            'fft_activation_pattern': fft_activation,
            'decision': 'DEEPFAKE' if confidence > 0.3 else 'REAL',
            'decision_confidence': max(confidence, 1 - confidence),
            'target_class': target_class,  # Track which class GradCAM targeted
            'gradcam_debug': gradcam_debug,
        }


def load_model(model_path, model_type='hybrid', device='cuda'):
    """Load model for visualization"""
    if model_type == 'hybrid':
        model = HybridDeepfakeDetector(num_classes=1, pretrained=False)
    else:
        model = DeepfakeEfficientNet(num_classes=1, pretrained=False)
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    model.to(device)
    return model


def analyze_prediction(model, image_tensor, model_type='hybrid', device='cpu'):
    """
    Comprehensive prediction analysis
    Returns: dict with confidence, branch importance, and visualization
    """
    if model_type == 'hybrid':
        analyzer = HybridGradCAM(model, device)
        result = analyzer.generate_dual_visualization(image_tensor, None)
    else:
        # Fallback for non-hybrid models
        with torch.no_grad():
            output = model(image_tensor)
            confidence = torch.sigmoid(output).item()
        result = {
            'confidence': confidence,
            'decision': 'DEEPFAKE' if confidence > 0.3 else 'REAL',
            'spatial_importance': 1.0,
            'freq_importance': 0.0  # Only spatial branch for non-hybrid
        }
    
    return result


if __name__ == "__main__":
    import sys
    from PIL import Image
    from data.augmentations import get_val_transforms
    
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'backend/models/hybrid_kaggle_finetuned.pt'
    image_path = 'test_image.jpg'
    
    # Load model
    model = load_model(model_path, 'hybrid', device)
    
    # Load image
    transform = get_val_transforms(224)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Analyze
    result = analyze_prediction(model, image_tensor, 'hybrid', device)
    
    print(f"\nPrediction Analysis:")
    print(f"  Decision: {result['decision']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Spatial Branch Importance: {result['spatial_importance']:.1%}")
    print(f"  Frequency Branch Importance: {result['freq_importance']:.1%}")
