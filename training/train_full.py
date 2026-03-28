import sys
sys.path.insert(0, '.')
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from data.dataset import DeepfakeDataset
from data.augmentations import get_train_transforms, get_val_transforms
from models.hybrid_model import HybridDeepfakeDetector

CONFIG = {
    'data_root': '/home/mdaafi04/data',
    'dataset': 'faceforensics',
    'img_size': 224,
    'batch_size': 32,
    'num_epochs': 15,
    'lr': 1e-4,
    'dropout': 0.4,
    'num_workers': 4,
    'checkpoint_dir': 'checkpoints/full',
    'save_every': 5,
    'max_train_samples': None,  # Full dataset
    'max_val_samples': None,    # Full dataset
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def get_pos_weight(dataset):
    labels = [l for _, l in dataset.samples]
    n_neg = labels.count(0)
    n_pos = labels.count(1)
    return torch.tensor([n_neg / n_pos], dtype=torch.float)

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

    print(f"Full train samples: {len(train_ds)} | Full val samples: {len(val_ds)}")

    # Handle class imbalance with pos_weight
    pos_weight = get_pos_weight(train_ds).to(device)
    print(f"pos_weight: {pos_weight.item():.4f}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=CONFIG['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=CONFIG['num_workers'],
                            pin_memory=True)

    model = HybridDeepfakeDetector(num_classes=1, dropout=CONFIG['dropout']).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])

    wandb.init(project='deepfake-detection', config=CONFIG, name='hybrid-full-dataset')

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

        if epoch % CONFIG['save_every'] == 0:
            ckpt_path = f"{CONFIG['checkpoint_dir']}/hybrid_full_epoch{epoch}.pt"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc}, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{CONFIG['checkpoint_dir']}/hybrid_full_best.pt")
            print(f"Best model updated: val_acc={val_acc:.4f}")

    wandb.finish()
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
