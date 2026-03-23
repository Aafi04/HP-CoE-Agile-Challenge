import sys
sys.path.insert(0, '.')
from data.dataset import DeepfakeDataset
from data.augmentations import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

DATA_ROOT = '/home/mdaafi04/data'

def check_balance(dataset_name, split, transform):
    ds = DeepfakeDataset(DATA_ROOT, split=split,
                         transform=transform, dataset=dataset_name)
    real = sum(1 for _, l in ds.samples if l == 0)
    fake = sum(1 for _, l in ds.samples if l == 1)
    print(f"{dataset_name} {split}: {len(ds)} total | {real} real | {fake} fake | ratio {fake/real:.2f}")
    return ds

def test_dataloader(ds, batch_size=32):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    batch_imgs, batch_labels = next(iter(loader))
    print(f"Batch shape: {batch_imgs.shape}, Labels: {batch_labels[:8].tolist()}")
    assert batch_imgs.shape == (batch_size, 3, 224, 224)
    print("DataLoader test PASSED\n")

if __name__ == '__main__':
    # FF++ splits
    for split in ['train', 'val', 'test']:
        ds = check_balance('faceforensics', split, get_val_transforms())
    print()

    # CelebDF splits
    for split in ['Train', 'Val', 'Test']:
        ds = check_balance('celebdf', split, get_val_transforms())
    print()

    # DataLoader test on FF++ train
    print("Testing DataLoader...")
    ds = check_balance('faceforensics', 'train', get_train_transforms())
    test_dataloader(ds, batch_size=32)

    # DataLoader test on CelebDF train
    ds = check_balance('celebdf', 'Train', get_train_transforms())
    test_dataloader(ds, batch_size=32)
