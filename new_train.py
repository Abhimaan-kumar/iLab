import os
import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


def build_transforms(img_size: int):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def make_dataloaders(train_dir: str, val_dir: str | None, img_size: int, batch_size: int, num_workers: int = 4):
    train_transform, val_transform = build_transforms(img_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = None
    val_dataset = None
    if val_dir is not None and os.path.isdir(val_dir):
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset


def build_model(num_classes: int, pretrained: bool = True):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and 'cpu' not in args.device else 'cpu')
    print(f"Using device: {device}")

    # Ensure training folder exists
    if not os.path.isdir(args.train_dir):
        raise FileNotFoundError(f"Training directory not found: {args.train_dir}")

    # Build dataloaders
    train_loader, val_loader, train_dataset, val_dataset = make_dataloaders(args.train_dir, args.val_dir, args.img_size, args.batch_size, num_workers=args.num_workers)

    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")

    model = build_model(num_classes, pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    best_model_path = Path(args.output_dir) / 'best_resnet50.pth'
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_samples = 0

        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', unit='batch')
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size
            loop.set_postfix(loss=running_loss / num_samples)

        scheduler.step()

        epoch_loss = running_loss / max(1, num_samples)
        print(f'Epoch {epoch} training loss: {epoch_loss:.4f}')

        # Validation
        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            print(f'Epoch {epoch} validation accuracy: {val_acc:.4f}')
            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_acc': val_acc}, best_model_path)
                print(f'Best model saved to: {best_model_path} (val_acc={val_acc:.4f})')
        else:
            # when there's no val set, save model each epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, Path(args.output_dir) / f'resnet50_epoch{epoch}.pth')

    print('Training complete')
    if best_val_acc > 0:
        print(f'Best validation accuracy: {best_val_acc:.4f}')


def build_parser():
    p = argparse.ArgumentParser(description='Train ResNet50 on a folder-organized dataset (ImageFolder)')
    p.add_argument('--train-dir', required=True, help='Training directory (folder with class subfolders). All images here are used for training')
    p.add_argument('--val-dir', default=None, help='Optional validation directory (folder with class subfolders)')
    p.add_argument('--img-size', type=int, default=224, help='Input image size')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size')
    p.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    p.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    p.add_argument('--pretrained', action='store_true', help='Use pretrained ImageNet weights')
    p.add_argument('--device', default='cuda', help='Device to use, e.g. "cuda" or "cpu". Falls back to cpu if CUDA not available')
    p.add_argument('--output-dir', default='training_output', help='Directory to save model checkpoints')
    p.add_argument('--num-workers', type=int, default=4, help='Number of dataloader worker processes')
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    start = time.time()
    train(args)
    print(f'Total time: {time.time() - start:.1f}s')


if __name__ == '__main__':
    main()
