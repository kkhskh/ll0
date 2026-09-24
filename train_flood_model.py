"""
Quick Training Script for Flood Detection
Minimal setup to test the flood pipeline
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flood_detection_pipeline import (
    Sen1Floods11Dataset,
    FloodSegmentationModel,
    FloodDetectionLoss,
    create_data_splits
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def calculate_metrics(pred, target, valid_mask, threshold=0.5):
    """Calculate IoU, F1, Precision, Recall"""
    pred_binary = (pred > threshold).float()
    
    # Apply valid mask
    pred_binary = pred_binary * valid_mask
    target = target * valid_mask
    
    # Calculate metrics
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()
    tn = ((1 - pred_binary) * (1 - target)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'iou': iou.item(),
        'f1': f1.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_metrics = {'iou': [], 'f1': [], 'precision': [], 'recall': []}
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        imagery = batch['imagery'].to(device)
        flood_mask = batch['flood_mask'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        # Forward pass
        pred = model(imagery)
        loss, loss_dict = criterion(pred, flood_mask, valid_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            metrics = calculate_metrics(pred.squeeze(1), flood_mask, valid_mask)
            for k, v in metrics.items():
                all_metrics[k].append(v)
        
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': loss.item(),
            'iou': metrics['iou'],
            'f1': metrics['f1']
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    return avg_loss, avg_metrics


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_metrics = {'iou': [], 'f1': [], 'precision': [], 'recall': []}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            imagery = batch['imagery'].to(device)
            flood_mask = batch['flood_mask'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            # Forward pass
            pred = model(imagery)
            loss, loss_dict = criterion(pred, flood_mask, valid_mask)
            
            # Calculate metrics
            metrics = calculate_metrics(pred.squeeze(1), flood_mask, valid_mask)
            for k, v in metrics.items():
                all_metrics[k].append(v)
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': loss.item(),
                'iou': metrics['iou']
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    return avg_loss, avg_metrics


def visualize_predictions(model, dataset, device, num_samples=4, save_path=None):
    """Visualize model predictions"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        
        imagery = sample['imagery'].unsqueeze(0).to(device)
        flood_mask = sample['flood_mask'].cpu().numpy()
        valid_mask = sample['valid_mask'].cpu().numpy()
        
        with torch.no_grad():
            pred = model(imagery)
            pred = pred.squeeze().cpu().numpy()
        
        # Show RGB (or SAR if no optical)
        if imagery.shape[1] >= 5:  # Has S2 data
            # Assume bands are [VV, VH, B, G, R, NIR, SWIR1, SWIR2]
            rgb = imagery[0, [4, 3, 2], :, :].cpu().numpy()
            rgb = np.transpose(rgb, (1, 2, 0))
            rgb = np.clip(rgb * 3, 0, 1)  # Adjust brightness
        else:  # Only SAR
            sar = imagery[0, 0, :, :].cpu().numpy()
            rgb = np.stack([sar, sar, sar], axis=-1)
        
        # Plot
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title('RGB/SAR')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(flood_mask, cmap='Blues', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='Blues', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Show difference
        diff = np.abs(pred - flood_mask) * valid_mask
        axes[i, 3].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[i, 3].set_title('Error')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Sen1Floods11 flood segmentation model")
    parser.add_argument("--data-dir", default="datasets/floods/sen1floods11")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default="checkpoints/flood")
    parser.add_argument("--log-dir", default="logs/flood")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--s2-bands", nargs="*", type=int, default=[1, 2, 3, 7, 11, 12])
    parser.add_argument("--s1-only", action="store_true", help="Use only Sentinel-1 SAR inputs")
    parser.add_argument("--no-official-splits", action="store_true", help="Ignore official Sen1Floods11 split CSVs")
    return parser.parse_args()


def main():
    args = parse_args()
    CONFIG = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'use_s1': True,
        'use_s2': not args.s1_only,
        's2_bands': args.s2_bands,
        'device': args.device,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'num_workers': args.num_workers,
        'prefer_official_splits': not args.no_official_splits,
    }
    
    print("=" * 60)
    print("FLOOD DETECTION TRAINING")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Data directory: {CONFIG['data_dir']}")
    
    # Create directories
    Path(CONFIG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # Create data splits
    print("\nCreating data splits...")
    train_chips, val_chips, test_chips = create_data_splits(
        CONFIG['data_dir'],
        prefer_official_splits=CONFIG['prefer_official_splits'],
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = Sen1Floods11Dataset(
        CONFIG['data_dir'],
        train_chips,
        mode='train',
        use_s1=CONFIG['use_s1'],
        use_s2=CONFIG['use_s2'],
        s2_bands=CONFIG['s2_bands'],
        augment=True
    )
    
    val_dataset = Sen1Floods11Dataset(
        CONFIG['data_dir'],
        val_chips,
        mode='val',
        use_s1=CONFIG['use_s1'],
        use_s2=CONFIG['use_s2'],
        s2_bands=CONFIG['s2_bands'],
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    test_dataset = Sen1Floods11Dataset(
        CONFIG['data_dir'],
        test_chips,
        mode='test',
        use_s1=CONFIG['use_s1'],
        use_s2=CONFIG['use_s2'],
        s2_bands=CONFIG['s2_bands'],
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    # Calculate input channels
    in_channels = 0
    if CONFIG['use_s1']:
        in_channels += 2
    if CONFIG['use_s2']:
        in_channels += len(CONFIG['s2_bands'])
    
    print(f"Input channels: {in_channels}")
    
    # Create model
    print("Creating model...")
    model = FloodSegmentationModel(in_channels=in_channels)
    model.to(CONFIG['device'])
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = FloodDetectionLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs']
    )
    
    # Training loop
    best_iou = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': []
    }
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG['device']
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, CONFIG['device']
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_metrics['iou']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        
        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_iou,
                'config': CONFIG
            }, Path(CONFIG['checkpoint_dir']) / 'best_model.pth')
            print(f"✓ Saved best model (IoU: {best_iou:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, Path(CONFIG['checkpoint_dir']) / f'checkpoint_epoch_{epoch+1}.pth')
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation IoU: {best_iou:.4f}")
    
    # Save training history
    with open(Path(CONFIG['log_dir']) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    best_checkpoint_path = Path(CONFIG['checkpoint_dir']) / 'best_model.pth'
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print("\nEvaluating best checkpoint on held-out test split...")
        test_loss, test_metrics = validate(model, test_loader, criterion, CONFIG['device'])
        print(f"Test Loss: {test_loss:.4f} | Test IoU: {test_metrics['iou']:.4f}")
    else:
        test_loss, test_metrics = float('nan'), {}

    with open(Path(CONFIG['log_dir']) / 'metrics_summary.json', 'w') as f:
        json.dump(
            {
                'best_val_iou': best_iou,
                'test_loss': test_loss,
                'test_metrics': test_metrics,
                'config': CONFIG,
            },
            f,
            indent=2,
        )
    
    # Visualize some predictions
    print("\nGenerating visualizations...")
    visualize_predictions(
        model,
        val_dataset,
        CONFIG['device'],
        num_samples=4,
        save_path=Path(CONFIG['log_dir']) / 'predictions.png'
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
