import sys
sys.path.insert(0, '.')
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from data.dataset import DeepfakeDataset
from data.augmentations import get_train_transforms, get_val_transforms
from models.mesonet import MesoNet4

CONFIG = {
    'data_root': '/home/mdaafi04/data',
    'dataset': 'faceforensics',
    'img_size': 224,
    'batch_size': 64,
    'num_epochs': 10,
    'lr': 1e-3,
    'num_workers': 4,
    'checkpoint_dir': 'checkpoints',
    'max_train_samples': 20000,
    'max_val_samples': 5000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def get_subset(dataset, max_samples):
    if max_samples and len(dataset) > max_samples:
        indices = torch.randperm(len(dataset))[:max_samples].tolist()
        return Subset(dataset, indices)
    return dataset

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc='Train', leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    return total_loss / total, correct / total

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc='Val', leave=False)
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    return total_loss / total, correct / total

def main():
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    device = CONFIG['device']
    print(f"Using device: {device}")

    train_ds = DeepfakeDataset(CONFIG['data_root'], split='train',
                                transform=get_train_transforms(CONFIG['img_size']),
                                dataset=CONFIG['dataset'])
    val_ds = DeepfakeDataset(CONFIG['data_root'], split='val',
                              transform=get_val_transforms(CONFIG['img_size']),
                              dataset=CONFIG['dataset'])

    train_ds = get_subset(train_ds, CONFIG['max_train_samples'])
    val_ds = get_subset(val_ds, CONFIG['max_val_samples'])
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=CONFIG['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=CONFIG['num_workers'],
                            pin_memory=True)

    model = MesoNet4(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])

    wandb.init(project='deepfake-detection', config=CONFIG, name='mesonet4-baseline')

    best_val_acc = 0
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{CONFIG['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        wandb.log({'train_loss': train_loss, 'train_acc': train_acc,
                   'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{CONFIG['checkpoint_dir']}/mesonet_best.pt")
            print(f"Best model updated: val_acc={val_acc:.4f}")

    wandb.finish()
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
