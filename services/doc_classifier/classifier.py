import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

from services.card_crop_yolov8.detector import FieldDetector


def rotate_bgr(img: np.ndarray, rot: str) -> np.ndarray:
    if rot == "rot0":
        return img
    if rot == "rot90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180)
    if rot == "rot270":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unknown rotation: {rot}")


def normalize_key(field_name: str) -> str:
    k = field_name.lower().strip()
    mapping = {
        "aadhaar_number": "aadhaar_number",
        "aadhar_number": "aadhaar_number",
        "dob": "date_of_birth",
        "date_of_birth": "date_of_birth",
    }
    return mapping.get(k, k)


@dataclass
class DocRoute:
    doc_type: str  # "pan" | "aadhaar" | "unknown"
    rotation: str
    best_score: float
    score_pan: float
    score_aadhaar: float
    margin: float
    confident: bool
    details: Dict[str, Any]


class DocClassifier:
    """
    Detection-only router (fast):
    - Runs PAN detector and Aadhaar detector on a few rotations
    - Scores how well each detector explains the image using expected fields
    - Adds penalties for "wrong" fields (to reduce confident misroutes)
    - Returns 'unknown' if ambiguous (then API can fallback to OCR+schema)
    """

    EXPECTED: Dict[str, Set[str]] = {
        "pan": {"pan_number", "name", "father_name", "date_of_birth"},
        "aadhaar": {"aadhaar_number", "name", "gender", "date_of_birth", "address"},
    }

    def __init__(
        self,
        pan_detector: FieldDetector,
        aadhaar_detector: FieldDetector,
        rotations: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        min_best_score: float = 1.30,
        min_margin: float = 0.20,
        wrong_field_penalty: float = 0.35,
    ):
        self.pan_detector = pan_detector
        self.aadhaar_detector = aadhaar_detector
        self.rotations = rotations or ["rot0", "rot180", "rot90", "rot270"]
        self.conf_threshold = float(conf_threshold)
        self.min_best_score = float(min_best_score)
        self.min_margin = float(min_margin)
        self.wrong_field_penalty = float(wrong_field_penalty)

    def _score_detector(
        self,
        detector: FieldDetector,
        img: np.ndarray,
        expected: Set[str],
        unexpected: Set[str],
    ) -> Tuple[float, Dict[str, Any]]:
        dets = detector.detect(img)

        best_conf: Dict[str, float] = {}
        for d in dets:
            k = normalize_key(d.get("field", ""))
            c = float(d.get("conf", 0.0))
            if k and (k not in best_conf or c > best_conf[k]):
                best_conf[k] = c

        present_expected = {k for k in expected if best_conf.get(k, 0.0) >= self.conf_threshold}
        present_unexpected = {k for k in unexpected if best_conf.get(k, 0.0) >= self.conf_threshold}

        coverage = len(present_expected) / max(1, len(expected))
        conf_sum_expected = float(sum(best_conf.get(k, 0.0) for k in expected))

        # Penalize detections for fields that shouldn't exist for this doc type
        conf_sum_unexpected = float(sum(best_conf.get(k, 0.0) for k in present_unexpected))
        penalty = self.wrong_field_penalty * conf_sum_unexpected

        # Score: coverage dominates + confidence, minus penalty
        score = (2.0 * coverage) + (1.0 * conf_sum_expected) - penalty

        meta = {
            "coverage": float(coverage),
            "present_expected": sorted(list(present_expected)),
            "present_unexpected": sorted(list(present_unexpected)),
            "conf_sum_expected": float(conf_sum_expected),
            "conf_sum_unexpected": float(conf_sum_unexpected),
            "penalty": float(penalty),
            "raw_count": int(len(dets)),
        }
        return float(score), meta

    def predict(self, img_bgr: np.ndarray) -> DocRoute:
        best = None  # (best_score, chosen_doc, rotation, score_pan, score_aad, meta)

        for rot in self.rotations:
            img_r = rotate_bgr(img_bgr, rot)

            pan_score, pan_meta = self._score_detector(
                self.pan_detector,
                img_r,
                expected=self.EXPECTED["pan"],
                unexpected=self.EXPECTED["aadhaar"],
            )
            aad_score, aad_meta = self._score_detector(
                self.aadhaar_detector,
                img_r,
                expected=self.EXPECTED["aadhaar"],
                unexpected=self.EXPECTED["pan"],
            )

            if pan_score >= aad_score:
                cand = (pan_score, "pan", rot, pan_score, aad_score, {"pan": pan_meta, "aadhaar": aad_meta})
            else:
                cand = (aad_score, "aadhaar", rot, pan_score, aad_score, {"pan": pan_meta, "aadhaar": aad_meta})

            if best is None or cand[0] > best[0]:
                best = cand

        assert best is not None
        best_score, chosen, rot, score_pan, score_aadhaar, details = best
        margin = abs(float(score_pan) - float(score_aadhaar))

        confident = (best_score >= self.min_best_score) and (margin >= self.min_margin)
        doc_type = chosen if confident else "unknown"

        return DocRoute(
            doc_type=doc_type,
            rotation=rot,
            best_score=float(best_score),
            score_pan=float(score_pan),
            score_aadhaar=float(score_aadhaar),
            margin=float(margin),
            confident=bool(confident),
            details=details,
        )
