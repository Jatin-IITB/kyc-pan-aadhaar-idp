# Phase 11 Execution Plan — The Truth Engine

**Goal:** turn every resume metric into a measurement produced by `make eval` from a
clean checkout. Five workstreams, ordered by dependency.

## W1 — Identity Forge (synthetic document renderer)

`tools/forge/identity_forge.py` + `tools/forge/templates/`

- Identity generation: Faker(`en_IN`) names/addresses; PAN numbers matching
  `[A-Z]{5}[0-9]{4}[A-Z]` with correct 4th-char entity code; Aadhaar numbers with a
  **valid Verhoeff check digit** (the real algorithm — interviewers notice);
  DL numbers per state RTO format; EPIC numbers for Voter ID.
- Rendering: PIL-based templates per doc type — background pattern, emblem, labeled
  fields, synthetic photo block, QR stub. Fonts: 2–3 free fonts to vary.
- Augmentation stack (applied post-render): Gaussian blur, rotation (±15°, plus 90°
  multiples for the rotation classifier), perspective warp, JPEG quality sweep,
  brightness/shadow gradients, sensor noise.
- **Output per sample:** `image.jpg`, `truth.json` (all field values), `boxes.txt`
  (YOLO-format bboxes — Phase 15 training data for free).
- CLI: `python -m tools.forge.identity_forge --type pan --n 500 --out data/synthetic/`

## W2 — Tamper Forge (red team)

`tools/forge/tamper_forge.py` — takes genuine synthetic docs, emits forged variants:

| Attack | Method | Should trip |
|---|---|---|
| `text_splice` | re-render one field in a different font/size, paste over | ELA, font |
| `copy_move` | duplicate a region within the image | copy-move |
| `font_swap` | re-render all fields in a mismatched font | font |
| `screen_recapture` | perspective warp + Moiré overlay + specular glare | screen |
| `exif_edit` | inject Photoshop markers / impossible dates | metadata |
| `regenerate` | render same identity on subtly-wrong template | ELA, font |

Output: `image.jpg`, `truth.json` + `attack.json` (type, region, params).
Every attack parameterized by severity so we can plot recall-vs-subtlety curves.

## W3 — Eval Harness

`tools/eval/run_eval.py` + `Makefile` target `eval`

- **Extraction:** per-field precision/recall/F1, exact and fuzzy (Jaro-Winkler ≥ 0.9);
  aggregate micro/macro F1.
- **Forensics:** per-attack recall matrix, per-detector ROC/AUC over genuine+forged,
  FPR on the genuine set (the copy-move lesson, institutionalized).
- **Decisioning:** auto-clear rate, spoof-leakage count (must be 0), review-queue rate,
  false-reject rate vs a rules-only baseline (the −45% claim).
- **Latency:** per-node and end-to-end p50/p95 across the run.
- Outputs `eval/report.html` + `eval/metrics.json`; CI job fails if any metric drops
  below `config/eval_thresholds.yaml`.

## W4 — Forensic precision pass

- ✅ Copy-move v2: DC removal + repetition filter + shift-vector clustering (ADR-022)
- ✅ Noisy-OR aggregation replacing weighted-average dilution (ADR-024)
- ✅ Screen detector rescored by peak prominence; conservative threshold
- Data-driven backlog surfaced by the tamper forge (recall still under target):
  - **text_splice** — block-wise ELA to localize a recompression seam; global
    mean ELA cannot (0% recall today)
  - **regenerate** — clean re-render defeats ELA/font; needs no-capture-noise /
    PRNU sensor analysis (0% recall today)
  - **font_swap** — stroke-width consistency has no separating power on
    multi-font cards; needs OCR field-region context (blocked on W5 PaddleOCR)
  - **screen** — recall traded down (67%) for real-capture safety; retune once
    a real phone-capture validation set exists

## W5 — Dual path restored

- Install `paddleocr` + `paddlepaddle` in `.venv`; verify `ROIOCR` loads
- Train YOLOv8n field detectors on W1 output (2k PAN + 2k Aadhaar), promote through
  the model registry; ensemble node ranks YOLO vs VLM live again

## Exit criteria

1. `make eval` prints the north-star table (F1, tamper recall matrix, auto-clear,
   spoof leakage, latency) from a clean checkout
2. Copy-move FPR on 200 genuine synthetic docs: 0; recall on `copy_move` attacks ≥ 95%
3. CI gate live: a PR that degrades F1 or recall fails
4. YOLO weights trained from synthetic data serving in the ensemble

## Order of attack

W4 (started) → W1 → W2 → W3 → W5. Audit by separate agent after W3, then W5.
