"""Lightweight rotation classifier using a trained MobileNetV3-Small model.

Predicts document rotation from a BGR image: rot0, rot90, rot180, rot270.
Falls back to rot0 if model weights are not available.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models" / "rotation_classifier"
WEIGHTS_PATH = MODEL_DIR / "best.pt"
METADATA_PATH = MODEL_DIR / "metadata.json"

ROT_LABELS = ["rot0", "rot90", "rot180", "rot270"]


class RotationClassifier:
    def __init__(self, weights_path: Path | None = None):
        self._model = None
        self._class_map: dict[int, str] = {}
        self._device = None

        wp = weights_path or WEIGHTS_PATH
        if wp.exists():
            self._load(wp)
        else:
            logger.warning("Rotation model not found at %s — using fallback (rot0)", wp)

    def _load(self, weights_path: Path) -> None:
        import torch
        import torch.nn as nn
        from torchvision import models

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
        model.load_state_dict(torch.load(str(weights_path), map_location=self._device, weights_only=True))
        model.to(self._device)
        model.eval()
        self._model = model

        meta_path = weights_path.parent / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._class_map = {int(k): v for k, v in meta.get("class_mapping", {}).items()}
        else:
            self._class_map = dict(enumerate(ROT_LABELS))

        logger.info("Rotation classifier loaded from %s (device=%s)", weights_path, self._device)

    def predict(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        """Predict rotation class and confidence.

        Returns (rotation_label, confidence) e.g. ("rot90", 0.97).
        """
        if self._model is None:
            return "rot0", 0.0

        import torch
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = tf(img_rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        label = self._class_map.get(idx.item(), ROT_LABELS[idx.item()])
        return label, conf.item()
