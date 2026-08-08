"""Train a document rotation classifier on the Kaggle rotation-angle-detection dataset.

Usage:
    python -m tools.train.train_rotation [--epochs 10] [--batch-size 32] [--lr 1e-4]

Dataset layout (under data/datasets/rotation-angle-detection/):
    not-rot/   -> rot0   (0°)
    cw-90/     -> rot90  (90° clockwise)
    cw-180/    -> rot180 (180°)
    acw-90/    -> rot270 (270° clockwise = 90° anti-clockwise)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "datasets" / "rotation-angle-detection"
OUTPUT_DIR = REPO_ROOT / "models" / "rotation_classifier"

CLASS_TO_ROT = {"not-rot": "rot0", "cw-90": "rot90", "cw-180": "rot180", "acw-90": "rot270"}
ROT_LABELS = ["rot0", "rot90", "rot180", "rot270"]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    if pretrained:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        logger.info("Loaded ImageNet pretrained weights")
    else:
        model = models.mobilenet_v3_small(weights=None)
        logger.info("Training from scratch (random init)")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train rotation classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if not DATASET_DIR.exists():
        logger.error("Dataset not found at %s", DATASET_DIR)
        return

    device = get_device()
    logger.info("Device: %s", device)

    train_tf, val_tf = build_transforms()
    full_dataset = datasets.ImageFolder(str(DATASET_DIR), transform=train_tf)

    folder_to_rot = {cls_name: CLASS_TO_ROT.get(cls_name, cls_name) for cls_name in full_dataset.classes}
    logger.info("Classes: %s", folder_to_rot)
    logger.info("Total images: %d", len(full_dataset))

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    val_ds.dataset = datasets.ImageFolder(str(DATASET_DIR), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"))
    logger.info("Train: %d, Val: %d", train_size, val_size)

    model = build_model(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            "Epoch %2d/%d  train_loss=%.4f train_acc=%.4f  val_loss=%.4f val_acc=%.4f  [%.1fs]",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, elapsed,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")
            logger.info("  -> New best val_acc=%.4f, saved to %s", best_acc, OUTPUT_DIR / "best.pt")

    torch.save(model.state_dict(), OUTPUT_DIR / "last.pt")

    metadata = {
        "architecture": "mobilenet_v3_small",
        "num_classes": 4,
        "class_mapping": {str(i): folder_to_rot[c] for i, c in enumerate(full_dataset.classes)},
        "folder_mapping": folder_to_rot,
        "best_val_acc": best_acc,
        "epochs": args.epochs,
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Training complete. Best val_acc=%.4f", best_acc)
    logger.info("Model saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
