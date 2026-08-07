from __future__ import annotations

from services.decisioning.auto_clear import AutoClearEngine
from services.decisioning.calibrator import ConfidenceCalibrator
from services.decisioning.thresholds import SPOOF_REJECT_THRESHOLD, SPOOF_REVIEW_THRESHOLD
from services.graph.state import CaseState


def decide_node(state: CaseState) -> CaseState:
    dt = state.get("doc_type", "unknown")
    is_valid = state.get("schema_valid", False)
    attempt_rescue = state.get("attempt_rescue", False)
    quality_meta = state.get("quality_meta", {})
    extraction = state.get("extraction_normalized", {})
    clf_info = state.get("classifier_info")
    rot = state.get("rotation_hint", "rot0")
    validation_msg = state.get("validation_message", "")
    rescue_result = state.get("llm_rescue_result", {})
    extraction_path = state.get("extraction_path", "yolo")
    score = state.get("validation_score", 0.0)
    spoof_score = state.get("spoof_score", 0.0)
    forensics_result = state.get("forensics_result", {})
    policy_result = state.get("policy_result", {})
    cross_doc_result = state.get("cross_doc_result", {})

    critical_count = 0
    for k in ("aadhaar_number", "pan_number", "name", "date_of_birth"):
        val = extraction.get(k, {})
        if isinstance(val, dict) and val.get("value"):
            critical_count += 1

    if not state.get("quality_passed", True):
        status = "REJECTED_QUALITY"
        result = {
            "document_type": dt,
            "quality_check": quality_meta,
            "extraction": {},
            "validation": {"is_valid": False, "message": quality_meta.get("rejection_reason", "Rejected")},
            "status": status,
        }
        return {**state, "decision": status, "final_result": result}

    if spoof_score > SPOOF_REJECT_THRESHOLD:
        status = "REJECTED_SPOOF"
        result = {
            "document_type": dt,
            "quality_check": quality_meta,
            "forensics": forensics_result,
            "extraction": extraction,
            "validation": {"is_valid": False, "message": "Document flagged as spoofed/tampered"},
            "status": status,
        }
        return {**state, "decision": status, "final_result": result}

    if not is_valid and critical_count < 2:
        status = "REJECTED_CONTENT"
        result = {
            "document_type": dt,
            "quality_check": quality_meta,
            "extraction": {},
            "validation": {"is_valid": False, "message": "Insufficient content"},
            "status": status,
        }
        return {**state, "decision": status, "final_result": result}

    if attempt_rescue:
        if is_valid:
            existing = quality_meta.get("rejection_reason", "")
            suffix = "RESCUED: Readable Content"
            quality_meta = {**quality_meta, "rejection_reason": f"{existing} | {suffix}" if existing else suffix}
            status = "SUCCESS"
        else:
            status = "REJECTED_QUALITY"
            result = {
                "document_type": dt,
                "quality_check": quality_meta,
                "extraction": {},
                "validation": {"is_valid": False, "message": quality_meta.get("rejection_reason", "Rejected")},
                "status": status,
            }
            return {**state, "decision": status, "final_result": result}
    else:
        status = "SUCCESS" if is_valid else "PARTIAL_SUCCESS"

    calibrator = ConfidenceCalibrator()
    policy_compliant = policy_result.get("compliant", True) if policy_result else True
    cross_doc_consistency = cross_doc_result.get("consistency_score", 1.0) if cross_doc_result else 1.0
    critical_contradictions = sum(1 for c in cross_doc_result.get("contradictions", []) if c.get("severity") == "CRITICAL") if cross_doc_result else 0

    calibration = calibrator.calibrate(
        extraction_score=score,
        forensics_score=spoof_score,
        policy_score=1.0 if policy_compliant else 0.3,
        cross_doc_score=cross_doc_consistency,
    )

    engine = AutoClearEngine()
    auto_clear_result = engine.evaluate(
        calibration,
        spoof_score=spoof_score,
        policy_compliant=policy_compliant,
        critical_contradictions=critical_contradictions,
        quality_passed=state.get("quality_passed", True),
    )

    final_rec = auto_clear_result["final_recommendation"]
    if final_rec == "AUTO_CLEAR" and status == "SUCCESS":
        status = "AUTO_CLEARED"
    elif final_rec == "REJECT":
        status = "REJECTED_CALIBRATION"
    elif final_rec == "REVIEW" and status == "SUCCESS":
        status = "REVIEW"

    if spoof_score > SPOOF_REVIEW_THRESHOLD and status not in ("REJECTED_SPOOF", "REJECTED_CALIBRATION"):
        status = "REVIEW_SPOOF"

    result = {
        "document_type": dt,
        "chosen_rotation": rot,
        "classifier": clf_info,
        "quality_check": quality_meta,
        "forensics": forensics_result,
        "extraction": extraction,
        "validation": {"is_valid": is_valid, "message": validation_msg},
        "selection": {"is_valid": is_valid, "message": validation_msg, "score": score},
        "routing_mode": extraction_path,
        "calibration": {
            "calibrated_confidence": calibration["calibrated_confidence"],
            "raw_scores": calibration.get("raw_scores", {}),
            "recommendation": final_rec,
            "overrides": auto_clear_result.get("overrides_applied", []),
        },
        "status": status,
    }

    if policy_result:
        result["policy"] = policy_result
    if cross_doc_result and not cross_doc_result.get("skipped"):
        result["cross_doc"] = cross_doc_result
    if rescue_result:
        result["llm_rescue"] = rescue_result

    return {
        **state,
        "decision": status,
        "final_confidence": calibration["calibrated_confidence"],
        "calibration_result": calibration,
        "final_result": result,
    }
