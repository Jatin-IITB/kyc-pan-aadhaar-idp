from __future__ import annotations

import logging
from pathlib import Path
from functools import lru_cache

import yaml

from services.card_crop_yolov8.detector import FieldDetector
from services.doc_classifier.classifier import DocClassifier
from services.ocr_paddle.roi_ocr import ROIOCR
from services.pipeline import KYCPipeline, PipelineConfig
from services.extraction.llm_cleaner import LLMKycCleaner, LLMCleanerConfig
from services.extraction.vlm_extractor import VLMConfig, VLMExtractor
from services.graph.deps import PipelineDeps
from services.graph.workflow import build_kyc_graph

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


class _NullDetector:
    """Stand-in when YOLO weights are missing — returns no detections, forcing VLM fallback."""
    def detect(self, image_bgr):
        return []


def _load_models_config() -> dict:
    cfg_path = REPO_ROOT / "config" / "models.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _build_deps() -> PipelineDeps:
    cfg = _load_models_config()
    yolo_cfg = cfg.get("yolov8", {})

    pan_cfg = yolo_cfg.get("pan_fields", {})
    aad_cfg = yolo_cfg.get("aadhar_fields", {})

    pan_w = REPO_ROOT / pan_cfg.get("weights", "models/yolov8/pan_field_detector_v1/best.pt")
    aad_w = REPO_ROOT / aad_cfg.get("weights", "models/yolov8/aadhar_field_detector_v1/best.pt")
    pan_conf = float(pan_cfg.get("conf", 0.25))
    aad_conf = float(aad_cfg.get("conf", 0.25))

    if pan_w.exists():
        pan_detector = FieldDetector(str(pan_w), conf=pan_conf)
    else:
        logger.warning("PAN YOLO weights not found at %s — using null detector (VLM-only mode)", pan_w)
        pan_detector = _NullDetector()

    if aad_w.exists():
        aadhaar_detector = FieldDetector(str(aad_w), conf=aad_conf)
    else:
        logger.warning("Aadhaar YOLO weights not found at %s — using null detector (VLM-only mode)", aad_w)
        aadhaar_detector = _NullDetector()

    try:
        ocr = ROIOCR(lang="en")
    except Exception:
        logger.warning("PaddleOCR unavailable — VLM-only extraction mode")
        ocr = None
    llm_cfg = cfg.get("llm", {})
    llm_cleaner = LLMKycCleaner(config=LLMCleanerConfig(
        model=llm_cfg.get("model", "qwen3:8b"),
        timeout_s=float(llm_cfg.get("timeout_s", 20)),
    ))
    doc_classifier = DocClassifier(
        pan_detector=pan_detector,
        aadhaar_detector=aadhaar_detector,
        conf_threshold=0.25,
        min_best_score=1.30,
        min_margin=0.20,
        wrong_field_penalty=0.35,
    )

    vlm_cfg = cfg.get("vlm", {})
    try:
        vlm_extractor = VLMExtractor(
            config=VLMConfig(
                model=vlm_cfg.get("model", "llama3.2-vision:11b"),
                timeout_s=float(vlm_cfg.get("timeout_s", 30)),
            )
        )
    except Exception:
        vlm_extractor = None

    # Disabled by default: the v1 model (trained on the Kaggle rotation set) does not
    # transfer to PAN/Aadhaar cards — it scored 0/4 on real PAN rotations and was
    # confidently wrong (rot180 read as rot0 at 0.99). Until a model clears
    # tools/train/eval_rotation.py on in-domain data, the detector-based rotation
    # search stays authoritative. Opt in via rotation.enabled in config/models.yaml.
    rotation_classifier = None
    rot_cfg = cfg.get("rotation", {})
    if rot_cfg.get("enabled", False):
        try:
            from services.doc_classifier.rotation_model import RotationClassifier
            rotation_classifier = RotationClassifier()
        except Exception:
            logger.warning("RotationClassifier failed to load — using brute-force rotation search")
    else:
        logger.info("Rotation classifier disabled — using brute-force rotation search")

    policy_verifier = None
    try:
        from services.rag.policy_verifier import PolicyVerifier
        policy_verifier = PolicyVerifier()
    except Exception:
        logger.debug("PolicyVerifier unavailable — skipping policy verification")

    return PipelineDeps(
        pan_detector=pan_detector,
        aadhaar_detector=aadhaar_detector,
        ocr=ocr,
        doc_classifier=doc_classifier,
        llm_cleaner=llm_cleaner,
        vlm_extractor=vlm_extractor,
        policy_verifier=policy_verifier,
        rotation_classifier=rotation_classifier,
        config=PipelineConfig(),
    )


@lru_cache(maxsize=1)
def get_pipeline() -> KYCPipeline:
    deps = _build_deps()
    return KYCPipeline(
        pan_detector=deps.pan_detector,
        aadhaar_detector=deps.aadhaar_detector,
        ocr=deps.ocr,
        doc_classifier=deps.doc_classifier,
        llm_cleaner=deps.llm_cleaner,
        config=deps.config,
    )


@lru_cache(maxsize=1)
def get_graph():
    deps = _build_deps()
    return build_kyc_graph(pipeline_deps=deps), deps


def check_hot_reload() -> bool:
    try:
        from services.active_learning.model_registry import ModelRegistry
        registry = ModelRegistry()
        models = registry.list_models()
        active = models.get("active", {})

        needs_reload = False
        for model_name, info in active.items():
            weights_path = info.get("weights_path", "")
            if weights_path and Path(weights_path).exists():
                cached = _hot_reload_cache.get(model_name)
                if cached != info.get("version"):
                    needs_reload = True
                    _hot_reload_cache[model_name] = info["version"]

        if needs_reload:
            logger.info("Hot reload triggered — clearing cached pipeline and graph")
            get_pipeline.cache_clear()
            get_graph.cache_clear()
            return True
    except Exception:
        pass
    return False


_hot_reload_cache: dict = {}
