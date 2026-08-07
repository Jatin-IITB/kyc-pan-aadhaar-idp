# apps/review_ui/main.py
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(layout="wide", page_title="KYC Review Console")

from apps.review_ui.adapters import FileSystemAdapter
from apps.review_ui.domain import ReviewResult
from apps.common.settings import load_settings

SETTINGS = load_settings()
EVAL_PATH = str(SETTINGS.eval_results_path)
IMG_ROOTS = [str(p) for p in SETTINGS.image_roots]

FIELD_COLORS = {
    "pan_number": "#FF6B6B",
    "aadhaar_number": "#FF6B6B",
    "name": "#4ECDC4",
    "father_name": "#45B7D1",
    "date_of_birth": "#96CEB4",
    "gender": "#FFEAA7",
    "address": "#DDA0DD",
    "photo": "#FFB347",
}
DEFAULT_COLOR = "#A8E6CF"

RISK_COLORS = {
    "LOW": "#2ECC71",
    "MEDIUM": "#F39C12",
    "HIGH": "#E74C3C",
    "CRITICAL": "#C0392B",
}


@st.cache_resource
def get_adapter():
    return FileSystemAdapter(EVAL_PATH, IMG_ROOTS)


def load_image(path: str) -> Optional[Image.Image]:
    if not path or not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def draw_bboxes(img: Image.Image, extraction: Dict[str, Any]) -> Image.Image:
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for field_name, field_data in extraction.items():
        if not isinstance(field_data, dict):
            continue
        bbox = field_data.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        color = FIELD_COLORS.get(field_name, DEFAULT_COLOR)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = field_name.replace("_", " ").title()
        conf = field_data.get("det_conf", 0)
        tag = f"{label} ({conf:.0%})" if conf else label
        text_bbox = draw.textbbox((x1, y1 - 16), tag, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 16), tag, fill="white", font=font)

    return overlay


def render_forensics_panel(forensics: Dict[str, Any]) -> None:
    if not forensics:
        st.info("No forensics data available.")
        return

    spoof_score = forensics.get("spoof_score", 0.0)
    risk_level = forensics.get("risk_level", "LOW")
    risk_st_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "red"}.get(risk_level, "gray")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Spoof Score", f"{spoof_score:.2f}")
    with col2:
        st.markdown(f"**Risk Level:** :{risk_st_color}[{risk_level}]")
    with col3:
        st.metric("Recommendation", forensics.get("recommendation", "N/A"))

    component_scores = forensics.get("component_scores", {})
    if component_scores:
        st.markdown("**Component Breakdown**")
        for component, score in component_scores.items():
            label = component.replace("_", " ").title()
            st.progress(min(1.0, score), text=f"{label}: {score:.2f}")

    evidence = forensics.get("evidence", [])
    if evidence:
        st.markdown("**Evidence Flags**")
        for e in evidence:
            icon = "🔴" if e.get("score", 0) > 0.5 else "🟡"
            st.markdown(f"- {icon} **{e.get('type', '?')}** ({e.get('score', 0):.2f}): {e.get('detail', '')}")


def render_calibration_panel(calibration: Dict[str, Any], result: Dict[str, Any]) -> None:
    if not calibration:
        st.info("No calibration data available.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        conf = calibration.get("calibrated_confidence", 0)
        st.metric("Calibrated Confidence", f"{conf:.2%}")
    with col2:
        st.metric("Recommendation", calibration.get("recommendation", "N/A"))
    with col3:
        st.metric("Status", result.get("status", "N/A"))

    raw_scores = calibration.get("raw_scores", {})
    if raw_scores:
        st.markdown("**Signal Weights**")
        weights = {"extraction": 0.35, "forensics": 0.25, "policy": 0.25, "cross_doc": 0.15}
        for signal, score in raw_scores.items():
            w = weights.get(signal, 0)
            label = signal.replace("_", " ").title()
            st.progress(min(1.0, score), text=f"{label} ({w:.0%} weight): {score:.2f}")

    overrides = calibration.get("overrides", [])
    if overrides:
        st.markdown("**Override Rules Applied**")
        for o in overrides:
            st.warning(f"**{o.get('rule', '?')}** → {o.get('action', '?')}: {o.get('reason', '')}")


def render_cross_doc_panel(cross_doc: Dict[str, Any]) -> None:
    if not cross_doc:
        st.info("No cross-document analysis available (single document case).")
        return

    col1, col2 = st.columns(2)
    with col1:
        consistency = cross_doc.get("consistency_score", 1.0)
        st.metric("Consistency Score", f"{consistency:.2%}")
    with col2:
        rec = cross_doc.get("recommendation", "PASS")
        color = "green" if rec == "PASS" else "red" if rec == "REJECT" else "orange"
        st.markdown(f"**Recommendation:** :{color}[{rec}]")

    contradictions = cross_doc.get("contradictions", [])
    if contradictions:
        st.markdown("**Contradictions Found**")
        for c in contradictions:
            severity = c.get("severity", "INFO")
            icon = "🔴" if severity == "CRITICAL" else "🟡" if severity == "WARNING" else "🔵"
            field = c.get("field", "?")
            values = c.get("values", [])
            docs = c.get("docs", [])
            st.markdown(
                f"- {icon} **{field}** ({severity}): "
                f"`{values[0] if values else '?'}` ({docs[0] if docs else '?'}) vs "
                f"`{values[1] if len(values) > 1 else '?'}` ({docs[1] if len(docs) > 1 else '?'})"
            )
    else:
        st.success("No contradictions detected across documents.")

    entity = cross_doc.get("entity_resolution", {})
    if entity:
        st.markdown("**Entity Resolution**")
        match_score = entity.get("match_score", 1.0)
        canonical = entity.get("canonical_name", "")
        same = entity.get("is_same_person", True)
        st.markdown(
            f"- Match Score: **{match_score:.2f}** | "
            f"Canonical Name: **{canonical}** | "
            f"Same Person: {'✅' if same else '❌'}"
        )


def render_policy_panel(policy: Dict[str, Any]) -> None:
    if not policy:
        return

    overall = policy.get("overall_status", "UNKNOWN")
    color = "green" if overall == "COMPLIANT" else "red" if overall == "NON_COMPLIANT" else "orange"
    st.markdown(f"**Overall:** :{color}[{overall}]")

    checks = policy.get("checks", [])
    for check in checks:
        status = check.get("status", "?")
        icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚪"
        req = check.get("requirement", "")
        st.markdown(f"- {icon} {req}")
        if check.get("explanation"):
            st.caption(f"  {check['explanation']}")


def get_result_data(job_id: str) -> Dict[str, Any]:
    adapter = get_adapter()
    if not adapter.results_path.exists():
        return {}
    try:
        data = json.loads(adapter.results_path.read_text(encoding="utf-8"))
        items = data.get("results", []) if isinstance(data, dict) else data
        for item in items:
            if item.get("filename") == job_id:
                return item.get("result", {}) or {}
    except Exception:
        pass
    return {}


# --- Main App ---
adapter = get_adapter()
st.title("KYC Review Console")

# Sidebar
st.sidebar.header("Controls")
filter_status = st.sidebar.radio(
    "Status Filter", ["INVALID", "REJECTED", "VALID", "ALL"], key="status_filter"
)
reviewer_name = st.sidebar.text_input(
    "Reviewer", value=os.getenv("USER", "analyst"), key="reviewer_name"
)

show_bboxes = st.sidebar.checkbox("Show Detection Bboxes", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Keyboard Shortcuts**")
st.sidebar.code("N → Next document\nP → Previous document\nA → Approve (save)\nR → Reject (save)")

if st.sidebar.button("Refresh Data"):
    st.rerun()

# Data Loading
jobs = adapter.get_jobs(filter_status)
st.sidebar.markdown(f"**Queue Size:** {len(jobs)}")

if not jobs:
    st.info("Queue is empty. Great job!")
    st.stop()

# Navigation State
if "idx" not in st.session_state:
    st.session_state.idx = 0


def next_doc():
    st.session_state.idx = min(len(jobs) - 1, st.session_state.idx + 1)


def prev_doc():
    st.session_state.idx = max(0, st.session_state.idx - 1)


# Navigation bar
nav_cols = st.columns([1, 1, 4, 2])
with nav_cols[0]:
    st.button("Previous (P)", on_click=prev_doc, key="btn_prev")
with nav_cols[1]:
    st.button("Next (N)", on_click=next_doc, key="btn_next")
with nav_cols[2]:
    st.markdown(
        f"**Document {st.session_state.idx + 1} of {len(jobs)}**"
    )

if st.session_state.idx >= len(jobs):
    st.session_state.idx = 0

job = jobs[st.session_state.idx]
result_data = get_result_data(job.id)

# --- Workspace Layout ---
col_img, col_data = st.columns([1, 1])

with col_img:
    st.subheader("Document")
    img = load_image(job.image_path)
    if img:
        if show_bboxes and job.extraction:
            img = draw_bboxes(img, job.extraction)
        st.image(img, caption=job.id, use_container_width=True)
    else:
        st.error(f"Image not found: {job.image_path}")

with col_data:
    st.subheader("Extraction Data")

    # Status Banner
    if job.status == "VALID":
        st.success(f"VALID ({job.document_type.upper()})")
    elif job.status == "REJECTED":
        st.warning(f"REJECTED: {job.validation_error}")
    else:
        st.error(f"INVALID: {job.validation_error}")

    # Tabbed panels for deep inspection
    tab_fields, tab_forensics, tab_calibration, tab_crossdoc, tab_policy = st.tabs(
        ["Fields", "Forensics", "Calibration", "Cross-Doc", "Policy"]
    )

    with tab_fields:
        if job.extraction:
            for field_name, field_data in job.extraction.items():
                if isinstance(field_data, dict):
                    value = field_data.get("value", "")
                    det_c = field_data.get("det_conf", 0)
                    ocr_c = field_data.get("ocr_conf", 0)
                    color = FIELD_COLORS.get(field_name, DEFAULT_COLOR)
                    st.markdown(
                        f"**{field_name.replace('_', ' ').title()}**: `{value}` "
                        f"(det: {det_c:.0%}, ocr: {ocr_c:.0%})"
                    )
                else:
                    st.markdown(f"**{field_name.replace('_', ' ').title()}**: `{field_data}`")
        else:
            st.info("No extraction data.")

    with tab_forensics:
        render_forensics_panel(result_data.get("forensics", {}))

    with tab_calibration:
        render_calibration_panel(
            result_data.get("calibration", {}),
            result_data,
        )

    with tab_crossdoc:
        render_cross_doc_panel(result_data.get("cross_doc", {}))

    with tab_policy:
        render_policy_panel(result_data.get("policy", {}))

# --- Correction Form ---
st.markdown("---")
st.subheader("Review & Correct")

with st.form(key=f"form_{job.id}"):
    fields = []
    if job.document_type == "pan":
        fields = ["pan_number", "date_of_birth", "name", "father_name"]
    elif job.document_type == "aadhaar":
        fields = ["aadhaar_number", "date_of_birth", "gender", "name"]
    elif job.extraction:
        fields = list(job.extraction.keys())

    if not fields:
        st.info("No structured fields to edit.")

    corrected = {}
    field_cols = st.columns(2)
    for i, field in enumerate(fields):
        val_obj = job.extraction.get(field, {})
        curr_val = val_obj.get("value", "") if isinstance(val_obj, dict) else str(val_obj or "")
        label = field.replace("_", " ").title()
        with field_cols[i % 2]:
            corrected[field] = st.text_input(label, value=str(curr_val), key=f"field_{job.id}_{field}")

    notes = st.text_area("Reviewer Notes", height=80, key=f"notes_{job.id}")

    btn_cols = st.columns(3)
    with btn_cols[0]:
        submitted = st.form_submit_button("Save to Ground Truth (A)", type="primary")
    with btn_cols[1]:
        reject = st.form_submit_button("Reject Document (R)")

    if submitted or reject:
        review_result = ReviewResult(
            job_id=job.id,
            reviewer=reviewer_name,
            corrected_data=corrected if submitted else {},
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            notes=f"[REJECTED] {notes}" if reject else notes,
        )
        saved_path = adapter.save_review(review_result)
        st.toast(f"Saved: {saved_path}")
        next_doc()
        st.rerun()
