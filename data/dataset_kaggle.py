import os
from PIL import Image
from torch.utils.data import Dataset

class KaggleDeepfakeDataset(Dataset):
    """
    Kaggle deepfake dataset loader.
    
    Expected structure:
    root_dir/
      Train/
        Real/   (*.jpg, *.png)
        Fake/   (*.jpg, *.png)
      Validation/
        Real/
        Fake/
      Test/
        Real/
        Fake/
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to Kaggle dataset root
            split: 'train', 'validation', or 'test'
            transform: torchvision transforms
        """
        self.transform = transform
        self.samples = []
        
        # Standardize split name
        split_name = split.strip().lower()
        if split_name == 'train':
            split_dir = os.path.join(root_dir, 'Train')
        elif split_name == 'validation':
            split_dir = os.path.join(root_dir, 'Validation')
        elif split_name == 'test':
            split_dir = os.path.join(root_dir, 'Test')
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Load real images (label=0)
        real_dir = os.path.join(split_dir, 'Real')
        if os.path.exists(real_dir):
            self._add_samples(real_dir, label=0)
        else:
            raise FileNotFoundError(f"Real directory not found: {real_dir}")
        
        # Load fake images (label=1)
        fake_dir = os.path.join(split_dir, 'Fake')
        if os.path.exists(fake_dir):
            self._add_samples(fake_dir, label=1)
        else:
            raise FileNotFoundError(f"Fake directory not found: {fake_dir}")
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
        
    def _add_samples(self, directory, label):
        """Add all image files from directory"""
        exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        for fname in os.listdir(directory):
            if fname.lower().endswith(exts):
                path = os.path.join(directory, fname)
                self.samples.append((path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a blank image instead of crashing
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
