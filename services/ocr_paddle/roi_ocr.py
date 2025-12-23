import cv2
import numpy as np
import re
from paddleocr import PaddleOCR
from typing import Dict, List

class ROIOCR:
    def __init__(self, lang="en"):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        if h < 40 or w < 80:
            crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def clean_text(self, field_name: str, text: str) -> str:
        text = text.strip()
        field_key = field_name.lower()
        
        if "dob" in field_key or "date" in field_key:
            match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
            return match.group(1) if match else text
        
        if "pan_number" in field_key:
            match = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text.upper())
            return match.group(0) if match else text
            
        if "aadhar" in field_key or "aadhaar" in field_key:
            digits = re.sub(r"\D", "", text)
            return digits if len(digits) == 12 else text
            
        return text

    def extract(self, image: np.ndarray, fields: List[Dict]) -> List[Dict]:
        results: List[Dict] = []
        h, w = image.shape[:2]

        for f in fields:
            x1, y1, x2, y2 = f["bbox"]
            x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
            x2, y2 = min(w, x2 + 5), min(h, y2 + 5)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0: continue

            enhanced = self._preprocess(crop)
            ocr_out = self.ocr.ocr(enhanced, cls=True)

            text, ocr_conf = "", 0.0
            if ocr_out and ocr_out[0]:
                lines = [line[1][0] for line in ocr_out[0]]
                confs = [float(line[1][1]) for line in ocr_out[0]]
                text = self.clean_text(f["field"], " ".join(lines))
                ocr_conf = (sum(confs) / len(confs)) if confs else 0.0

            results.append({
                "field": f["field"],
                "text": text,
                "det_conf": float(f["conf"]),
                "ocr_conf": float(ocr_conf),
                "bbox": f["bbox"]
            })

        collapsed: Dict[str, Dict] = {}
        for r in results:
            key = r["field"].lower()
            score = 0.5 * r["det_conf"] + 0.5 * r["ocr_conf"]
            if key not in collapsed or score > collapsed[key]["_score"]:
                collapsed[key] = {**r, "_score": score}

        return [{k: v for k, v in r.items() if k != "_score"} for r in collapsed.values()]