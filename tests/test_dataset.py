import sys
sys.path.insert(0, '.')
from data.dataset import DeepfakeDataset
from data.augmentations import get_train_transforms, get_val_transforms

DATA_ROOT = '/home/mdaafi04/data'

def test_faceforensics():
    ds = DeepfakeDataset(DATA_ROOT, split='train', 
                         transform=get_train_transforms(), 
                         dataset='faceforensics')
    print(f"FF++ train samples: {len(ds)}")
    img, label = ds[0]
    print(f"Image shape: {img.shape}, Label: {label}")
    assert img.shape == (3, 224, 224)
    print("FF++ test PASSED")

def test_celebdf():
    ds = DeepfakeDataset(DATA_ROOT, split='Train',
                         transform=get_val_transforms(),
                         dataset='celebdf')
    print(f"CelebDF train samples: {len(ds)}")
    img, label = ds[0]
    print(f"Image shape: {img.shape}, Label: {label}")
    assert img.shape == (3, 224, 224)
    print("CelebDF test PASSED")

if __name__ == '__main__':
    test_faceforensics()
    test_celebdf()
