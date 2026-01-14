# apps/review_ui/main.py
import streamlit as st
import os
from datetime import datetime, timezone
from PIL import Image

# --- 1. Page Config ---
st.set_page_config(layout="wide", page_title="KYC Review Console")

# Clean Architecture Imports
from apps.review_ui.adapters import FileSystemAdapter
from apps.review_ui.domain import ReviewResult
from apps.common.settings import load_settings

SETTINGS = load_settings()
EVAL_PATH = str(SETTINGS.eval_results_path)
IMG_ROOTS = [str(p) for p in SETTINGS.image_roots]

# --- Helper Functions ---
@st.cache_resource
def get_adapter():
    return FileSystemAdapter(EVAL_PATH, IMG_ROOTS)

def load_image(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return None

# --- Main App ---
adapter = get_adapter()
st.title("🛡️ KYC Review Console")

# Sidebar
st.sidebar.header("Controls")
# UPDATED FILTER: Added "REJECTED"
filter_status = st.sidebar.radio("Status Filter", ["INVALID", "REJECTED", "VALID", "ALL"], key="status_filter")
reviewer_name = st.sidebar.text_input("Reviewer", value=os.getenv("USER", "analyst"), key="reviewer_name")

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Data Loading
jobs = adapter.get_jobs(filter_status)
st.sidebar.markdown(f"**Queue Size:** {len(jobs)}")

if not jobs:
    st.balloons()
    st.info("Queue is empty. Great job!")
    st.stop()

# Navigation State
if "idx" not in st.session_state: 
    st.session_state.idx = 0

def next_doc():
    st.session_state.idx = min(len(jobs)-1, st.session_state.idx + 1)
def prev_doc():
    st.session_state.idx = max(0, st.session_state.idx - 1)

col_prev, col_next, _ = st.columns([1, 1, 6])
with col_prev: st.button("⬅️ Previous", on_click=prev_doc)
with col_next: st.button("Next ➡️", on_click=next_doc)

# Index Safety Check
if st.session_state.idx >= len(jobs):
    st.session_state.idx = 0
    
job = jobs[st.session_state.idx]

# --- Workspace Layout ---
col_img, col_data = st.columns([1, 1])

with col_img:
    st.subheader("Document")
    img = load_image(job.image_path)
    if img:
        st.image(img, caption=job.id, width="stretch")
    else:
        st.error(f"Image not found: {job.image_path}")

with col_data:
    st.subheader("Extraction Data")
    
    # Status Banner
    if job.status == "VALID":
        st.success(f"✅ VALID ({job.document_type.upper()})")
    elif job.status == "REJECTED":
        st.warning(f"⛔ REJECTED: {job.validation_error}")
    else:
        st.error(f"❌ INVALID: {job.validation_error}")

    # Correction Form
    with st.form(key=f"form_{job.id}"):
        st.markdown("### Correct Fields")
        
        fields = []
        if job.document_type == "pan":
             fields = ["pan_number", "date_of_birth", "name", "father_name"]
        elif job.document_type == "aadhaar":
             fields = ["aadhaar_number", "date_of_birth", "gender", "name"]
        elif job.extraction:
             fields = list(job.extraction.keys())
        
        if not fields:
             st.info("No structured fields to edit (Rejected).")

        corrected = {}
        for field in fields:
            val_obj = job.extraction.get(field, {})
            curr_val = val_obj.get("value", "") if isinstance(val_obj, dict) else str(val_obj or "")
            
            label = field.replace("_", " ").title()
            corrected[field] = st.text_input(label, value=str(curr_val))
            
        st.markdown("---")
        notes = st.text_area("Reviewer Notes", height=100)
        
        submitted = st.form_submit_button("💾 Save to Ground Truth", type="primary")
        
        if submitted:
            result = ReviewResult(
                job_id=job.id,
                reviewer=reviewer_name,
                corrected_data=corrected,
                reviewed_at=datetime.now(timezone.utc).isoformat(),
                notes=notes
            )
            saved_path = adapter.save_review(result)
            st.toast(f"Saved: {saved_path}", icon="✅")
            st.rerun()
