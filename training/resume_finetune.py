"""
Resume fine-tuning from checkpoint if training is interrupted
==============================================================

Usage:
    python training/resume_finetune.py --resume_from 2 --data_root /path/to/data --pretrained backend/models/hybrid_full_best.pt
    
This script loads a checkpoint from a specific epoch and resumes training.
"""

import sys
sys.path.insert(0, '.')

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.dataset_kaggle import KaggleDeepfakeDataset
from data.augmentations import get_train_transforms, get_val_transforms
from models.hybrid_model import HybridDeepfakeDetector

CONFIG = {
    'img_size': 224,
    'batch_size': 96,
    'num_epochs': 10,
    'lr': 1e-5,
    'weight_decay': 1e-4,
    'num_workers': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'checkpoint_dir': 'checkpoints/kaggle_finetune',
}


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Train', leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return total_loss / total, correct / total


def val_epoch(model, loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
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
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
    
    return total_loss / total, correct / total


def main(args):
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    device = CONFIG['device']
    
    print(f"Using device: {device}")
    
    # Load model
    model = HybridDeepfakeDetector(num_classes=1, pretrained=False, dropout=0.4)
    model.to(device)
    
    # Load checkpoint to resume from
    resume_epoch = args.resume_from
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], f'checkpoint_epoch{resume_epoch}.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Available checkpoints in {CONFIG['checkpoint_dir']}:")
        if os.path.exists(CONFIG['checkpoint_dir']):
            for f in sorted(os.listdir(CONFIG['checkpoint_dir'])):
                print(f"     - {f}")
        return
    
    print(f"Loading checkpoint from epoch {resume_epoch}: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("✓ Model loaded from checkpoint")
    
    # Load datasets
    print(f"Loading Kaggle dataset from: {args.data_root}")
    train_ds = KaggleDeepfakeDataset(
        args.data_root,
        split='train',
        transform=get_train_transforms(CONFIG['img_size'])
    )
    val_ds = KaggleDeepfakeDataset(
        args.data_root,
        split='validation',
        transform=get_val_transforms(CONFIG['img_size'])
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Scheduler: create a new one, but we'll skip to the current epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    
    # Skip scheduler steps to match the resumed epoch
    for _ in range(resume_epoch):
        scheduler.step()
    
    # Training loop
    best_val_acc = 0
    best_model_path = os.path.join(CONFIG['checkpoint_dir'], 'best_kaggle.pt')
    
    print(f"\nResuming fine-tuning from epoch {resume_epoch}")
    print(f"Will train epochs {resume_epoch + 1} to {CONFIG['num_epochs']}")
    print("="*60)
    
    for epoch in range(resume_epoch, CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved to {best_model_path} (acc={val_acc:.4f})")
        
        # Save checkpoint for this epoch
        checkpoint_path = os.path.join(
            CONFIG['checkpoint_dir'],
            f'checkpoint_epoch{epoch+1}.pt'
        )
        torch.save(model.state_dict(), checkpoint_path)
    
    print("\n" + "="*60)
    print(f"Fine-tuning complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")
    
    # Test on validation set one more time
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = val_epoch(model, val_loader, criterion, device)
    print(f"\nFinal validation accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resume fine-tuning from checkpoint')
    parser.add_argument(
        '--resume_from',
        type=int,
        required=True,
        help='Epoch number to resume from (will start training at epoch N+1)'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='./Dataset',
        help='Path to Kaggle dataset root'
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        default='backend/models/hybrid_full_best.pt',
        help='Path to pre-trained model (unused for resume, but kept for consistency)'
    )
    
    args = parser.parse_args()
    main(args)
