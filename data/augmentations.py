import torchvision.transforms as T
from PIL import Image

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
    return T.Compose([
        T.Lambda(lambda x: resize_with_pad(x, (img_size, img_size))),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])

def get_val_transforms(img_size=224):
    return T.Compose([
        T.Lambda(lambda x: resize_with_pad(x, (img_size, img_size))),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
