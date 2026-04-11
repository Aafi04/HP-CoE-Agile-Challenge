import torchvision.transforms as T
from PIL import Image


class ResizeWithPad:
    """Custom transform: resize with aspect ratio preservation and padding"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def __call__(self, img):
        """Resize image while preserving aspect ratio"""
        img.thumbnail((self.target_size[0], self.target_size[1]), Image.Resampling.LANCZOS)
        pad_w = (self.target_size[0] - img.width) // 2
        pad_h = (self.target_size[1] - img.height) // 2
        padded = Image.new('RGB', self.target_size, (128, 128, 128))  # Gray padding
        padded.paste(img, (pad_w, pad_h))
        return padded


def resize_with_pad(img, target_size=(224, 224)):
    """
    Resize image while preserving aspect ratio and adding padding.
    This prevents distortion that can destroy deepfake artifacts.
    
    Args:
        img: PIL Image
        target_size: (height, width) tuple
        
    Returns:
        PIL Image padded to target_size with original aspect ratio preserved
    """
    img.thumbnail((target_size[0], target_size[1]), Image.Resampling.LANCZOS)
    pad_w = (target_size[0] - img.width) // 2
    pad_h = (target_size[1] - img.height) // 2
    padded = Image.new('RGB', target_size, (128, 128, 128))  # Gray padding
    padded.paste(img, (pad_w, pad_h))
    return padded


def get_train_transforms(img_size=224):
    transforms_list = [
        ResizeWithPad((img_size, img_size)),
    ]
    
    # Handle API differences in torchvision versions
    try:
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))
    except TypeError:
        transforms_list.append(T.RandomHorizontalFlip())
    
    transforms_list.extend([
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # RandomErasing may not be available in older versions
    if hasattr(T, 'RandomErasing'):
        try:
            transforms_list.append(T.RandomErasing(p=0.2, scale=(0.02, 0.1)))
        except (TypeError, AttributeError):
            pass
    
    return T.Compose(transforms_list)


def get_val_transforms(img_size=224):
    return T.Compose([
        ResizeWithPad((img_size, img_size)),  # Pickleable class instead of lambda
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
