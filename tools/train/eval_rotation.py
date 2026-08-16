"""Evaluate the trained rotation classifier: per-class accuracy + confusion matrix.

Overall accuracy hides asymmetric failure. A model that nails rot0 but confuses
rot90/rot270 is far worse for the pipeline than the headline number suggests,
because a 180-degree miss is unrecoverable downstream while rot0 is the common case.

Usage:
    python -m tools.train.eval_rotation [--weights models/rotation_classifier/best.pt]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "datasets" / "rotation-angle-detection"
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "rotation_classifier" / "best.pt"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate rotation classifier")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if not args.weights.exists():
        logger.error("Weights not found: %s", args.weights)
        return

    device = get_device()

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Same split + seed as training, so this is the held-out set the model never saw.
    dataset = datasets.ImageFolder(str(DATASET_DIR), transform=val_tf)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    _, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    meta_path = args.weights.parent / "metadata.json"
    class_map = {}
    if meta_path.exists():
        class_map = {int(k): v for k, v in json.loads(meta_path.read_text()).get("class_mapping", {}).items()}
    names = [class_map.get(i, dataset.classes[i]) for i in range(len(dataset.classes))]

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(dataset.classes))
    model.load_state_dict(torch.load(str(args.weights), map_location=device, weights_only=True))
    model.to(device).eval()

    n = len(dataset.classes)
    confusion = torch.zeros(n, n, dtype=torch.long)
    for imgs, labels in loader:
        preds = model(imgs.to(device)).argmax(1).cpu()
        for t, p in zip(labels, preds):
            confusion[t.long(), p.long()] += 1

    total = int(confusion.sum())
    correct = int(confusion.diag().sum())
    logger.info("\nHeld-out set: %d images", total)
    logger.info("Overall accuracy: %.4f (%d/%d)\n", correct / total, correct, total)

    logger.info("Per-class accuracy:")
    for i, name in enumerate(names):
        support = int(confusion[i].sum())
        acc = int(confusion[i, i]) / support if support else 0.0
        logger.info("  %-8s %.4f  (%d/%d)", name, acc, int(confusion[i, i]), support)

    logger.info("\nConfusion matrix (rows=true, cols=predicted):")
    logger.info("%-10s%s", "", "".join(f"{c:>10}" for c in names))
    for i, name in enumerate(names):
        logger.info("%-10s%s", name, "".join(f"{int(v):>10}" for v in confusion[i]))

    # Surface the failure mode that matters most downstream.
    worst, worst_pair = 0, None
    for i in range(n):
        for j in range(n):
            if i != j and int(confusion[i, j]) > worst:
                worst, worst_pair = int(confusion[i, j]), (names[i], names[j])
    if worst_pair:
        logger.info("\nLargest confusion: %s predicted as %s (%d times)", worst_pair[0], worst_pair[1], worst)


if __name__ == "__main__":
    main()
