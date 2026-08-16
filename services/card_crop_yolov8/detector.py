import logging

import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

NON_TEXT_FIELDS = frozenset({"photo", "signature", "emblem", "goi_symbol", "logo"})


class FieldDetector:
    def __init__(self, weights_path: str, conf: float = 0.25,
                 field_map: Optional[Dict[str, str]] = None):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.field_map = field_map or {}

    def _map_label(self, label: str) -> str | None:
        if self.field_map:
            mapped = self.field_map.get(label)
            if mapped is None:
                logger.warning("Unmapped YOLO class '%s' — skipping (add to config/models.yaml field_map)", label)
                return None
            if mapped in NON_TEXT_FIELDS:
                return None
            return mapped
        return label.lower()

    def detect(self, image_bgr: np.ndarray) -> List[Dict]:
        results = self.model.predict(image_bgr, conf=self.conf, verbose=False)

        fields = []
        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.cpu().numpy()
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                raw_label = self.model.names[cls_id]
                label = self._map_label(raw_label)
                if label is None:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].astype(int)
                conf = float(boxes.conf[i])

                fields.append({
                    "field": label,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": conf
                })

        return fields
