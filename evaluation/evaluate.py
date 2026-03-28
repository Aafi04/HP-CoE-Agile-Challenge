import sys
sys.path.insert(0, '.')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from data.dataset import DeepfakeDataset
from data.augmentations import get_val_transforms
from models.hybrid_model import HybridDeepfakeDetector
from models.efficientnet import DeepfakeEfficientNet
from models.mesonet import MesoNet4

def evaluate_model(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Evaluating', leave=False):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)

def compute_metrics(probs, labels, threshold=0.5):
    preds = (probs > threshold).astype(int)
    return {
        'accuracy': (preds == labels).mean(),
        'auc_roc': roc_auc_score(labels, probs),
        'f1': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'confusion_matrix': confusion_matrix(labels, preds)
    }

def plot_roc_curves(results, output_path='evaluation/roc_curves.png'):
    plt.figure(figsize=(8, 6))
    for name, (probs, labels) in results.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.4f})')
    plt.plot([0,1],[0,1],'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ROC curves saved: {output_path}")

def plot_confusion_matrix(cm, model_name, output_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    classes = ['Real', 'Fake']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=12)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {output_path}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('evaluation/figures', exist_ok=True)

    # Test dataset - FF++
    print("Loading FF++ test set...")
    test_ds_ff = DeepfakeDataset('/home/mdaafi04/data', split='test',
                                  transform=get_val_transforms(224),
                                  dataset='faceforensics')
    test_loader_ff = DataLoader(test_ds_ff, batch_size=64,
                                shuffle=False, num_workers=4, pin_memory=True)

    # Test dataset - CelebDF
    print("Loading CelebDF test set...")
    test_ds_celeb = DeepfakeDataset('/home/mdaafi04/data', split='Test',
                                     transform=get_val_transforms(224),
                                     dataset='celebdf')
    test_loader_celeb = DataLoader(test_ds_celeb, batch_size=64,
                                   shuffle=False, num_workers=4, pin_memory=True)

    # Load models
    models = {}

    print("Loading Hybrid model...")
    hybrid = HybridDeepfakeDetector(num_classes=1, pretrained=False).to(device)
    hybrid.load_state_dict(torch.load('checkpoints/full/hybrid_full_best.pt', map_location=device))
    models['Hybrid (EfficientNet+FFT)'] = hybrid

    print("Loading EfficientNet baseline...")
    effnet = DeepfakeEfficientNet(num_classes=1, pretrained=False).to(device)
    effnet.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=device))
    models['EfficientNet-B4'] = effnet

    print("Loading MesoNet baseline...")
    meso = MesoNet4(num_classes=1).to(device)
    meso.load_state_dict(torch.load('checkpoints/mesonet_best.pt', map_location=device))
    models['MesoNet4'] = meso

    # Evaluate all models on FF++ test set
    print("\n" + "="*60)
    print("EVALUATION ON FACEFORENSICS++ TEST SET")
    print("="*60)

    roc_results_ff = {}
    for name, model in models.items():
        probs, labels = evaluate_model(model, test_loader_ff, device)
        metrics = compute_metrics(probs, labels)
        roc_results_ff[name] = (probs, labels)
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        plot_confusion_matrix(
            metrics['confusion_matrix'], name,
            f"evaluation/figures/cm_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')}.png"
        )

    plot_roc_curves(roc_results_ff, 'evaluation/figures/roc_ff++.png')

    # Cross-dataset evaluation on CelebDF
    print("\n" + "="*60)
    print("CROSS-DATASET EVALUATION ON CELEB-DF v2")
    print("="*60)

    roc_results_celeb = {}
    for name, model in models.items():
        probs, labels = evaluate_model(model, test_loader_celeb, device)
        metrics = compute_metrics(probs, labels)
        roc_results_celeb[name] = (probs, labels)
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")

    plot_roc_curves(roc_results_celeb, 'evaluation/figures/roc_celebdf.png')

    print("\nEvaluation complete. Figures saved in evaluation/figures/")

if __name__ == '__main__':
    main()
