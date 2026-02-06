"""
Training script for off-road lane detection
Supports pretrained encoders and multiple architectures
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from preprocessing import ORFDDataset, Rellis3DDataset, get_combined_dataloaders
from model import (
    get_model, UNet, UNetSmall,
    CombinedLoss, CombinedFocalDiceLoss,
    calculate_iou, calculate_dice
)


def get_dataloaders(orfd_root=None, rellis_root=None, batch_size=8, img_size=256, num_workers=0):
    """Create train, validation, and test dataloaders from one or both datasets"""
    return get_combined_dataloaders(
        orfd_root=orfd_root,
        rellis_root=rellis_root,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers
    )


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_iou = 0
    total_dice = 0

    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_iou += calculate_iou(outputs, masks)
        total_dice += calculate_dice(outputs, masks)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{calculate_iou(outputs, masks):.4f}'
        })

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
            total_dice += calculate_dice(outputs, masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


def save_checkpoint(model, optimizer, scheduler, epoch, loss, iou, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'iou': iou,
    }, path)


def plot_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(history['train_iou'], label='Train')
    axes[1].plot(history['val_iou'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('IoU')
    axes[1].legend()

    axes[2].plot(history['train_dice'], label='Train')
    axes[2].plot(history['val_dice'], label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Dice')
    axes[2].set_title('Dice Score')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    # Handle empty string paths
    if args.orfd_root == '':
        args.orfd_root = None
    if args.rellis_root == '':
        args.rellis_root = None

    if not args.orfd_root and not args.rellis_root:
        raise ValueError("At least one dataset path must be provided (--orfd_root or --rellis_root)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{args.arch}_{args.encoder}" if args.arch != 'unet_basic' else args.arch
    output_dir = os.path.join(args.output_dir, f'{model_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        orfd_root=args.orfd_root,
        rellis_root=args.rellis_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

    print(f"\nCreating model: {args.arch} with encoder: {args.encoder}")
    if args.arch == 'unet_basic':
        model = UNet(in_channels=3, out_channels=1)
    elif args.arch == 'unet_small':
        model = UNetSmall(in_channels=3, out_channels=1)
    else:
        model = get_model(
            arch=args.arch,
            encoder=args.encoder,
            pretrained=not args.no_pretrained,
            in_channels=3,
            classes=1
        )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    if args.loss == 'combined':
        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    elif args.loss == 'focal_dice':
        criterion = CombinedFocalDiceLoss(focal_weight=0.5, dice_weight=0.5)
    else:
        criterion = CombinedLoss()
    print(f"Loss function: {args.loss}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' and args.amp else None
    if scaler:
        print("Using automatic mixed precision (AMP)")

    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [],
        'val_loss': [], 'val_iou': [], 'val_dice': []
    }

    best_val_iou = 0
    patience_counter = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} (lr: {optimizer.param_groups[0]['lr']:.2e})")
        print("-" * 40)

        train_loss, train_iou, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device)

        if args.scheduler == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_iou)

        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)

        print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, val_iou,
                os.path.join(output_dir, 'best_model.pth')
            )
            print(f"  -> New best model saved! IoU: {val_iou:.4f}")
        else:
            patience_counter += 1

        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, val_iou,
            os.path.join(output_dir, 'latest_model.pth')
        )

        plot_history(history, os.path.join(output_dir, 'training_history.png'))

        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {patience_counter} epochs without improvement")
            break

    print("\n" + "=" * 60)
    print("Final evaluation on test set...")

    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_iou, test_dice = validate(model, test_loader, criterion, device)
    print(f"Test - Loss: {test_loss:.4f}, IoU: {test_iou:.4f}, Dice: {test_dice:.4f}")

    print(f"\nTraining complete! Best validation IoU: {best_val_iou:.4f}")
    print(f"Model saved to: {output_dir}")

    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f"Model: {args.arch} + {args.encoder}\n")
        f.write(f"Image size: {args.img_size}\n")
        f.write(f"Datasets: ORFD={args.orfd_root}, Rellis={args.rellis_root}\n")
        f.write(f"Best validation IoU: {best_val_iou:.4f}\n")
        f.write(f"Test IoU: {test_iou:.4f}\n")
        f.write(f"Test Dice: {test_dice:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation model for off-road lane detection')

    parser.add_argument('--orfd_root', type=str, default='datasets/ORFD',
                        help='Path to ORFD dataset (set to empty string to disable)')
    parser.add_argument('--rellis_root', type=str, default=None,
                        help='Path to Rellis-3D dataset (optional)')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for models')

    parser.add_argument('--arch', type=str, default='unet',
                        choices=['unet', 'unetpp', 'deeplabv3', 'deeplabv3p', 'fpn', 'pspnet', 'unet_basic', 'unet_small'],
                        help='Model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Encoder backbone (resnet34, resnet50, efficientnet-b0, mobilenet_v2, etc.)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained encoder weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size (default 512)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default 8)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Data loading workers')

    parser.add_argument('--loss', type=str, default='focal_dice',
                        choices=['combined', 'focal_dice'],
                        help='Loss function')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau'],
                        help='Learning rate scheduler')

    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')

    args = parser.parse_args()
    main(args)