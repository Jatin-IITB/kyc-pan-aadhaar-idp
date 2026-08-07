# ADR-016: Evidence-Based Review UI with Tabbed Deep Inspection

## Status
Accepted

## Context
The original review UI was a basic Streamlit form showing extracted field values and a correction interface. Reviewers had no visibility into *why* the system made its decision — no forensics evidence, no calibration breakdown, no cross-document analysis, no detection bounding boxes. This made it impossible to make informed review decisions or to provide quality corrections for the active learning loop.

## Decision
Enhance the Review UI with five capabilities:

1. **Bbox overlays** — Draw YOLO detection bounding boxes on document images with field-specific colors and confidence labels. Togglable via sidebar checkbox.

2. **Tabbed deep inspection** — Five tabs (Fields, Forensics, Calibration, Cross-Doc, Policy) replace the flat extraction display. Each tab renders the corresponding pipeline output with visual indicators.

3. **Forensics evidence panel** — Component-score progress bars (ELA, copy-move, font, metadata, screen recapture), risk-level color coding, and evidence flag listing with severity icons.

4. **Calibration display** — Shows calibrated confidence, weighted signal breakdown (extraction 0.35, forensics 0.25, policy 0.25, cross-doc 0.15), auto-clear recommendation, and any override rules applied.

5. **Cross-doc comparison** — Consistency score, contradiction listing with field-level detail (values from each doc side by side), entity resolution match score, and canonical name.

6. **Two-column correction form** — Fields laid out in 2 columns for density. Save and Reject buttons with auto-advance to next document.

## Consequences
- Reviewers see the full evidence chain for every decision
- Corrections are more accurate because reviewers understand what the system saw
- Active learning quality improves via better ground truth
- UI requires no additional backend changes — all data comes from existing result.json
- Keyboard shortcut hints displayed (N/P/A/R) — Streamlit doesn't support true keyboard binding without JS, so these are button labels
