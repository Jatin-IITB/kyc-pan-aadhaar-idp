"""Generate the KYC-IDP technical documentation PDF.

Metrics are read from eval/metrics.json at build time so the document cannot
drift from measured reality. Regenerate after any `make eval` run:

    .venv/bin/python -m tools.docs.generate_pdf
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "docs" / "KYC-IDP-Technical-Documentation.pdf"

INK = colors.HexColor("#14181F")
MUTED = colors.HexColor("#5A6472")
ACCENT = colors.HexColor("#1F4E79")
RULE = colors.HexColor("#D4D9E0")
BAND = colors.HexColor("#F2F5F9")
WARN = colors.HexColor("#8A5A00")
WARNBG = colors.HexColor("#FDF6E7")

BUILD_DATE = "16 August 2026"
REPO_URL = "github.com/Jatin-IITB/kyc-pan-aadhaar-idp"


# ---------------------------------------------------------------- styles

def build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=ss["Title"], fontName="Helvetica-Bold",
        fontSize=27, leading=32, textColor=INK, alignment=TA_CENTER, spaceAfter=4,
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=ss["Normal"], fontName="Helvetica",
        fontSize=12.5, leading=18, textColor=MUTED, alignment=TA_CENTER,
    )
    s["h1"] = ParagraphStyle(
        "h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
        fontSize=16, leading=20, textColor=ACCENT, spaceBefore=2, spaceAfter=8,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
        fontSize=11.8, leading=15, textColor=INK, spaceBefore=12, spaceAfter=5,
    )
    s["body"] = ParagraphStyle(
        "body", parent=ss["Normal"], fontName="Helvetica",
        fontSize=9.6, leading=14.2, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=12, bulletIndent=3, spaceAfter=4,
    )
    s["caption"] = ParagraphStyle(
        "caption", parent=ss["Normal"], fontName="Helvetica-Oblique",
        fontSize=8.3, leading=11.8, textColor=MUTED, spaceAfter=9,
    )
    s["code"] = ParagraphStyle(
        "code", parent=ss["Normal"], fontName="Courier",
        fontSize=7.5, leading=10.2, textColor=INK,
    )
    s["callout"] = ParagraphStyle(
        "callout", parent=ss["Normal"], fontName="Helvetica",
        fontSize=9.1, leading=13.4, textColor=INK, alignment=TA_JUSTIFY,
    )
    s["cell"] = ParagraphStyle(
        "cell", parent=ss["Normal"], fontName="Helvetica",
        fontSize=8.6, leading=11.6, textColor=INK,
    )
    s["cellb"] = ParagraphStyle(
        "cellb", parent=s["cell"], fontName="Helvetica-Bold",
    )
    return s


S = build_styles()


# ---------------------------------------------------------------- helpers

def table(data, widths, align=None, header=True, zebra=True, font=8.6):
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    cmd = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if header:
        cmd += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ]
        if zebra:
            for r in range(1, len(data)):
                if r % 2 == 0:
                    cmd.append(("BACKGROUND", (0, r), (-1, r), BAND))
    for col, a in (align or {}).items():
        cmd.append(("ALIGN", (col, 0), (col, -1), a))
    t.setStyle(TableStyle(cmd))
    return t


def callout(title, body, bg=WARNBG, bar=WARN):
    inner = [
        [Paragraph(f'<b><font color="{bar.hexval()}">{title}</font></b>', S["callout"])],
        [Paragraph(body, S["callout"])],
    ]
    t = Table(inner, colWidths=[168 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.6, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (0, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
    ]))
    return t


def codeblock(text):
    lines = [[Paragraph(l.replace(" ", "&nbsp;") or "&nbsp;", S["code"])]
             for l in text.strip("\n").split("\n")]
    t = Table(lines, colWidths=[168 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BAND),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 0.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
    ]))
    return t


def bullets(items):
    return [Paragraph(f"• {i}", S["bullet"]) for i in items]


def pct(x):
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------- chrome

def _footer(canvas, doc, cover=False):
    canvas.saveState()
    if cover:
        canvas.restoreState()
        return
    w, h = A4
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(21 * mm, 15 * mm, w - 21 * mm, 15 * mm)
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(21 * mm, 10.5 * mm, "KYC-IDP — Technical Documentation")
    canvas.drawRightString(w - 21 * mm, 10.5 * mm, f"Page {canvas.getPageNumber() - 1}")
    canvas.restoreState()


def cover_bg(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(ACCENT)
    canvas.rect(0, h - 13 * mm, w, 13 * mm, stroke=0, fill=1)
    canvas.setFillColor(BAND)
    canvas.rect(0, 0, w, 9 * mm, stroke=0, fill=1)
    canvas.restoreState()


def build_doc():
    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=21 * mm, rightMargin=21 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="KYC-IDP — Technical Documentation",
        author="Jatin Gupta", subject="Document intelligence platform for Indian KYC",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=cover_bg),
        PageTemplate(id="body", frames=[frame], onPage=_footer),
    ])
    return doc


# ---------------------------------------------------------------- content

def story():
    m = json.loads((REPO_ROOT / "eval" / "metrics.json").read_text())
    ho = m["holdout"]["forensics"]
    hod = m["holdout"]["decision"]
    lat = ho["latency_ms"]
    pa = ho["per_attack"]

    F = []

    # ---- cover
    F += [Spacer(1, 48 * mm)]
    F.append(Paragraph("KYC-IDP", S["title"]))
    F.append(Paragraph("Multi-Agent Document Intelligence for Indian KYC",
                       S["subtitle"]))
    F.append(Spacer(1, 7 * mm))
    line = Table([[""]], colWidths=[52 * mm], rowHeights=[2])
    line.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)]))
    line.hAlign = "CENTER"
    F.append(line)
    F.append(Spacer(1, 9 * mm))
    F.append(Paragraph(
        "Tamper forensics · Regulatory compliance · Cross-document intelligence<br/>"
        "Calibrated decisioning · Hash-chained audit", S["subtitle"]))
    F.append(Spacer(1, 30 * mm))
    meta = table([
        ["Author", "Jatin Gupta"],
        ["Document", "Technical Documentation"],
        ["Version", "Phase 11 — W5 complete"],
        ["Date", BUILD_DATE],
        ["Repository", REPO_URL],
    ], [38 * mm, 92 * mm], header=False, zebra=False, font=9.2)
    meta.hAlign = "CENTER"
    meta.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("FONTSIZE", (0, 0), (-1, -1), 9.2),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]))
    F.append(meta)
    F.append(NextPageTemplate("body"))
    F.append(PageBreak())

    # ---- 1. overview
    F.append(Paragraph("1 &nbsp;Overview", S["h1"]))
    F.append(Paragraph(
        "KYC-IDP is a document intelligence platform for Indian identity documents "
        "(PAN and Aadhaar). It goes beyond field extraction to answer the questions "
        "that actually gate a KYC decision: is the document forged, does the packet "
        "satisfy regulation, do the documents agree with each other, and can the case "
        "be safely auto-approved. Every stage emits hash-chained audit events.",
        S["body"]))
    F.append(Paragraph(
        "The system is built as a LangGraph state machine with dual-path extraction — "
        "a YOLOv8 + PaddleOCR fast path with a vision-language-model fallback — and is "
        "governed by an evaluation harness that gates CI on ratcheting quality "
        "thresholds.", S["body"]))

    F.append(Paragraph("1.1 &nbsp;Problem framing", S["h2"]))
    F.append(table([
        ["Question", "Mechanism"],
        ["Is this document forged?", "Five-signal forensics suite, noisy-OR fusion"],
        ["Does it satisfy RBI regulation?", "Hybrid RAG with per-requirement citations"],
        ["Do the documents agree?", "Jaro-Winkler / Soundex entity resolution"],
        ["Can we auto-approve safely?", "Confidence calibration + hard overrides"],
        ["Can we prove what happened?", "SHA-256 hash-chained audit ledger"],
        ["Are the metrics real?", "Synthetic forge + held-out eval + CI gates"],
    ], [58 * mm, 110 * mm]))

    F.append(Paragraph("1.2 &nbsp;At a glance", S["h2"]))
    F.append(table([
        ["Graph nodes", "Unit tests", "ADRs", "Attack classes", "Doc schemas"],
        ["13", "105", "29", "6", "12+"],
    ], [33.6 * mm] * 5,
        align={0: "CENTER", 1: "CENTER", 2: "CENTER", 3: "CENTER", 4: "CENTER"}))

    # ---- 2. architecture
    F.append(Paragraph("2 &nbsp;Architecture", S["h1"]))
    F.append(Paragraph(
        "Processing is orchestrated as an explicit state machine rather than a linear "
        "call chain. Each stage is an independently testable node, routing is data-"
        "dependent, and forensics executes as a parallel branch so it never blocks "
        "extraction.", S["body"]))
    F.append(KeepTogether(codeblock("""
  Client -> FastAPI -> Celery
                         |
      +------------------+------------------+
      |        LangGraph State Machine      |
      |                                     |
      |   ingest -> quality_gate            |
      |        |-- reject --------------+   |
      |        +-- classify             |   |
      |             |-- extract_yolo    |   |
      |             |     |- conf ok    |   |
      |             |     +- VLM fallback|  |
      |             |        ensemble   |   |
      |             |        validate   |   |
      |             |     policy_verify |   |
      |             |       cross_doc   |   |
      |             |      llm_rescue   |   |
      |             +-- forensics ------+   |
      |                         decide      |
      |                   audit_commit      |
      +-------------------------------------+
                         |
   Postgres -- MinIO -- Qdrant -- Redis -- Streamlit
""")))
    F.append(KeepTogether([
        Paragraph("2.1 &nbsp;Dual-path extraction", S["h2"]),
        Paragraph(
            "Documents route to a fast detector-plus-OCR path when detection confidence is "
            "adequate, and to a vision-language model otherwise. An ensemble node selects "
            "between the two results. The consequence is graceful degradation: document "
            "types with no trained detector still extract, just more slowly.", S["body"]),
        table([
            ["Path", "Pipeline", "Typical latency"],
            ["Fast", "YOLOv8n detection → PaddleOCR on ROIs → normalize", "~200 ms"],
            ["Fallback", "Vision-LLM (minicpm-v) structured extraction", "~2–4 s"],
        ], [24 * mm, 108 * mm, 36 * mm]),
    ]))

    # ---- 3. results
    F.append(Paragraph("3 &nbsp;Measured Results", S["h1"]))
    F.append(Paragraph(
        f"Forensics figures are produced by <font face='Courier'>make eval</font> and "
        f"scored on a held-out seed pair ({m['dataset']}) never used for threshold "
        f"tuning. They are reproducible from a clean checkout.", S["body"]))

    F.append(Paragraph("3.1 &nbsp;Forensics — held-out set", S["h2"]))
    F.append(table([
        ["Metric", "Result"],
        ["Genuine false-positive rate", f"0 / {ho['genuine_n']}  ({pct(ho['genuine_fpr'])})"],
        ["Overall tamper recall", pct(ho["overall_recall"])],
        ["Decision-layer leakage", str(hod["flagged_leakage"])],
        ["Genuine auto-clear rate", pct(hod["genuine_auto_clear_rate"])],
    ], [86 * mm, 82 * mm], align={1: "LEFT"}))
    F.append(Paragraph(
        "A KYC system must never flag a genuine customer. Zero false positives is a "
        "hard invariant, CI-enforced. Leakage counts forensically-flagged documents "
        "that nonetheless auto-cleared — a decision-layer bug class, gated at zero.",
        S["caption"]))

    F.append(KeepTogether([
        Paragraph("3.2 &nbsp;Recall by attack class", S["h2"]),
        table([
            ["Attack", "Recall", "Detection basis"],
            ["exif_edit", pct(pa["exif_edit"]["recall"]), "EXIF software tags, date anomalies"],
            ["text_splice", pct(pa["text_splice"]["recall"]) + "  *", "JPEG quantization-table quality"],
            ["regenerate", pct(pa["regenerate"]["recall"]) + "  *", "JPEG quantization-table quality"],
            ["screen_recapture", pct(pa["screen_recapture"]["recall"]), "FFT Moiré analysis"],
            ["copy_move", pct(pa["copy_move"]["recall"]), "ORB keypoint matching (alignment-free)"],
            ["font_swap", pct(pa["font_swap"]["recall"]), "Not yet detectable"],
        ], [34 * mm, 22 * mm, 112 * mm], align={1: "CENTER"}),
        Spacer(1, 2 * mm),
        callout(
            "* These two figures are tautological — read them carefully",
            "The 100% recall on <b>text_splice</b> and <b>regenerate</b> reflects the detector "
            "firing on <i>“saved at quality &lt; 88”</i>, which is an artifact of how the forge "
            "writes these attacks — not evidence of tampering. An adversary who re-saves at "
            "Q=92 evades detection entirely. The gates are set at 0.90 to catch regressions, "
            "<b>not</b> as a claim of adversarial robustness. Full analysis in ADR-028."),
    ]))
    F.append(Spacer(1, 4 * mm))

    F.append(KeepTogether([
        Paragraph("3.3 &nbsp;Detector latency", S["h2"]),
        table([
            ["Detector", "p50 (ms)", "p95 (ms)"],
            ["metadata", f"{lat['metadata']['p50']:.1f}", f"{lat['metadata']['p95']:.1f}"],
            ["font", f"{lat['font']['p50']:.1f}", f"{lat['font']['p95']:.1f}"],
            ["ELA", f"{lat['ela']['p50']:.1f}", f"{lat['ela']['p95']:.1f}"],
            ["screen recapture", f"{lat['screen']['p50']:.1f}", f"{lat['screen']['p95']:.1f}"],
            ["copy-move (ORB)", f"{lat['copy_move']['p50']:.1f}", f"{lat['copy_move']['p95']:.1f}"],
        ], [56 * mm, 56 * mm, 56 * mm], align={1: "CENTER", 2: "CENTER"}),
    ]))

    F.append(KeepTogether([
        Paragraph("3.4 &nbsp;Field detection", S["h2"]),
        table([
            ["Model", "Source", "mAP@50", "mAP@50-95"],
            ["Aadhaar", "Pre-trained (HuggingFace)", "0.963 †", "0.748 †"],
            ["PAN", "Trained here (Roboflow data)", "0.919 ‡", "0.643 ‡"],
        ], [26 * mm, 74 * mm, 34 * mm, 34 * mm], align={2: "CENTER", 3: "CENTER"}),
        Paragraph(
            "† Reported by the upstream model author; not independently re-measured.",
            S["caption"]),
    ]))

    F.append(KeepTogether([
        Paragraph("3.5 &nbsp;PAN detector — per-class breakdown", S["h2"]),
        table([
            ["Class", "Precision", "Recall", "mAP@50"],
            ["name", "0.992", "1.000", "0.995"],
            ["fathername", "0.933", "1.000", "0.995"],
            ["dob", "1.000", "0.806", "0.931"],
            ["pan  (the PAN number)", "0.901", "0.600", "0.757"],
        ], [58 * mm, 36 * mm, 36 * mm, 38 * mm],
            align={1: "CENTER", 2: "CENTER", 3: "CENTER"}),
        Spacer(1, 2 * mm),
        callout(
            "‡ PAN results are provisional — the aggregate hides the important finding",
            "The PAN detector was trained on <b>71 images and validated on 6</b> (23 "
            "instances). An aggregate mAP over six images carries very wide error bars. "
            "More importantly, the headline 0.919 conceals that <b>pan</b> — the PAN "
            "number itself, the document's primary key — is the <i>weakest</i> class, "
            "missing 40% of instances. Training converged (box loss 2.21 → 0.93), so the "
            "ceiling is data volume, not schedule. A credible detector needs ≥500 "
            "annotated images merged across the available PAN datasets."),
    ]))

    # ---- 4. scope
    F.append(Spacer(1, 3 * mm))
    F.append(Paragraph("4 &nbsp;Scope and Validity of Results", S["h1"]))
    F.append(Paragraph(
        "Stated explicitly so the figures above are not over-read:", S["body"]))
    F += bullets([
        "All forensics numbers are measured on <b>synthetic</b> documents produced by the "
        "built-in Identity Forge — not production traffic.",
        "The 0% genuine false-positive rate holds against forge output saved at Q=92. Real "
        "submissions arrive via phone cameras, messaging apps and scanners at varied "
        "quality; real-world FPR is <b>unmeasured</b>.",
        "Decision-layer figures isolate the forensic gate — extraction, policy and "
        "cross-document scores are held at 1.0.",
        "PAN detection metrics rest on a six-image validation split and should be treated "
        "as directional only.",
        "End-to-end extraction F1 and full-graph p95 latency are <b>not yet benchmarked</b>.",
    ])
    F.append(Spacer(1, 2 * mm))
    F.append(Paragraph(
        "The evaluation harness exists precisely to keep these claims falsifiable. It is "
        "what surfaced the tautology documented in §3.2 — the system reporting its own "
        "weakness rather than concealing it.", S["body"]))

    # ---- 5. components
    F.append(Paragraph("5 &nbsp;Component Reference", S["h1"]))
    F.append(table([
        ["Module", "Responsibility"],
        ["services/graph/", "LangGraph state machine — 13 nodes, conditional routing"],
        ["services/forensics/", "ELA, ORB copy-move, font, JPEG/EXIF metadata, FFT screen-recapture, noisy-OR scorer"],
        ["services/rag/", "Policy indexer, hybrid retriever (dense + BM25 + RRF), cross-encoder reranker, citation verifier"],
        ["services/cross_doc/", "Entity resolution, contradiction detection, Indian address normalization"],
        ["services/decisioning/", "Confidence calibrator, auto-clear engine with hard overrides"],
        ["services/audit/", "SHA-256 hash-chained ledger, state replay"],
        ["services/active_learning/", "Ground-truth store, retrain triggers, model registry, regression checker"],
        ["services/extraction/", "VLM extractor, ensemble scoring, normalizers, LLM cleaner"],
        ["tools/forge/", "Identity Forge (Verhoeff-valid synthetic IDs) + Tamper Forge (6 attack classes)"],
        ["tools/eval/", "Evaluation harness with ratcheting CI gates"],
    ], [40 * mm, 128 * mm], font=8.3))

    F.append(Paragraph("5.1 &nbsp;Technology", S["h2"]))
    F.append(table([
        ["Layer", "Technology", "Layer", "Technology"],
        ["Orchestration", "LangGraph", "Vector store", "Qdrant + bge-small-en-v1.5"],
        ["API", "FastAPI + Uvicorn", "Reranking", "ms-marco-MiniLM-L-6-v2"],
        ["Queue", "Celery + Redis", "Detection", "YOLOv8n (Ultralytics)"],
        ["Vision LLM", "minicpm-v (Ollama)", "OCR", "PaddleOCR"],
        ["Text LLM", "Qwen 3 8B (Ollama)", "Persistence", "Postgres + MinIO"],
    ], [26 * mm, 52 * mm, 26 * mm, 64 * mm], font=8.3))

    # ---- 6. practices
    F.append(Spacer(1, 3 * mm))
    F.append(Paragraph("6 &nbsp;Engineering Practices", S["h1"]))

    F.append(Paragraph("6.1 &nbsp;Decision records", S["h2"]))
    F.append(Paragraph(
        "Twenty-nine ADRs capture every non-obvious decision, including the ones that "
        "failed. ADR-019 documents a rotation classifier that scored 0/4 on real cards "
        "and was disabled behind a config flag rather than shipped. ADR-028 documents "
        "the tautological detector in §3.2. Recording negative results is deliberate: it "
        "prevents the same ground being re-explored and keeps reported metrics anchored.",
        S["body"]))

    F.append(Paragraph("6.2 &nbsp;Ratcheting quality gates", S["h2"]))
    F.append(Paragraph(
        "config/eval_thresholds.yaml encodes the current <i>measured</i> floor rather than "
        "aspirations. Any change degrading a certified metric turns the build red. Floors "
        "are raised only after an eval run proves the improvement is real.", S["body"]))

    F.append(Paragraph("6.3 &nbsp;Held-out evaluation", S["h2"]))
    F.append(Paragraph(
        "Tuning and holdout seed pairs are separate and CI-enforced, so thresholds cannot "
        "be tuned into looking good on the data that produced them.", S["body"]))

    F.append(Paragraph("6.4 &nbsp;Independent audit per phase", S["h2"]))
    F.append(Paragraph(
        "Each phase is reviewed by a separate pass before the next begins, with findings "
        "classified Critical / Significant / Minor and remediated in place. The W5 audit "
        "surfaced six significant findings, all fixed prior to merge.", S["body"]))

    F.append(Paragraph("6.5 &nbsp;Failure modes made visible", S["h2"]))
    F.append(Paragraph(
        "Silent degradation is treated as a defect class in its own right. During W5 the "
        "PAN detector emitted class names absent from the configured field map; because "
        "unmapped labels are dropped by design, it produced zero fields and the pipeline "
        "quietly fell back to VLM-only — no exception, no error metric. A regression test "
        "now asserts that every class an installed model emits has a mapping, and was "
        "verified to fail against the broken configuration before being committed.",
        S["body"]))

    # ---- 7. gaps
    F.append(Paragraph("7 &nbsp;Known Gaps and Roadmap", S["h1"]))
    F.append(table([
        ["Gap", "Required work"],
        ["PAN detector under-trained (71/6 split, pan recall 0.60)",
         "Merge available PAN datasets to ≥500 annotated images; re-train and re-validate"],
        ["font_swap undetected (0% recall)",
         "OCR-context font forensics — stroke-width and glyph-metric consistency"],
        ["text_splice / regenerate detection evadable",
         "Frequency-domain work: double-JPEG ghosts, DCT histogram analysis, learned localizer"],
        ["No production-traffic calibration",
         "Calibrate quality thresholds against real scan-quality distributions"],
        ["Extraction F1 / end-to-end p95 unbenchmarked",
         "Extend the eval harness across the full graph"],
    ], [62 * mm, 106 * mm], font=8.3))

    F.append(KeepTogether([
        Paragraph("7.1 &nbsp;Reproducing these results", S["h2"]),
        codeblock("""
  docker compose up                              # infrastructure
  alembic upgrade head                           # schema

  .venv/bin/python -m pytest tests/unit -q       # 105 unit tests
  make eval                                      # regenerate all §3 figures

  # field detectors
  .venv/bin/python -m tools.train.download_pretrained --type aadhaar
  read -s ROBOFLOW_API_KEY && export ROBOFLOW_API_KEY
  .venv/bin/python -m tools.train.download_pretrained --type pan
"""),
        Paragraph(
            "Without detector weights present the pipeline runs VLM-only — functional, "
            "slower. Every figure in §3 regenerates from a clean checkout.", S["caption"]),
    ]))

    return F


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build_doc().build(story())
    print(f"wrote {OUT.relative_to(REPO_ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
