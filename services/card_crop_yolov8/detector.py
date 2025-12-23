import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict

class FieldDetector:
    def __init__(self, weights_path: str, conf: float = 0.25):
        self.model = YOLO(weights_path)
        self.conf = conf
    
    def detect(self, image_bgr: np.ndarray) -> List[Dict]:
        results = self.model.predict(image_bgr, conf=self.conf, verbose=False)
        
        fields = []
        for r in results:
            if r.boxes is None:
                continue
            
            boxes = r.boxes.cpu().numpy()
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                label = self.model.names[cls_id]
                x1, y1, x2, y2 = boxes.xyxy[i].astype(int)
                conf = float(boxes.conf[i])
                
                fields.append({
                    "field": label,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": conf
                })
        
        return fields
