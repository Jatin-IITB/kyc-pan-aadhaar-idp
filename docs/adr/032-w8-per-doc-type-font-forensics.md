# ADR-032: Per-doc-type font forensics (W8)

**Status:** Accepted  
**Date:** 2026-08-20  
**Supersedes:** — (extends ADR-030)

## Context

W6 (ADR-030) introduced template-conformance font forensics with a global
margin/vote setting (margin=8.0, vote=1). Font_swap recall reached 46.7% on
the eval holdout. Aadhaar font_swap recall remained at 6.2%—the largest
remaining gap.

W8 set out to improve Aadhaar font_swap detection by adding ID-line-specific
features.

## Investigation and negative results

1. **id_adv_cv (advance CV)**: Direction inverted on Aadhaar—inter-group
   spaces in "XXXX XXXX XXXX" inflate monospace advance CV MORE than
   proportional, so genuine > swapped (opposite of expected).

2. **id_width_cv on Aadhaar**: Genuine width CV (0.17–0.22) was HIGHER than
   swapped (0.08–0.18) due to specific digit font characteristics. The
   "high" direction check cannot work.

3. **Value-line-only corner/mod**: Excluding the ID line from ink-ranked
   features helped Aadhaar marginally but hurt PAN stability.

4. **Fundamental Aadhaar barrier**: At LOW severity font_swap, the Aadhaar
   ID number stays in mono (unchanged). Only ~15–20 characters of value
   text change. The ID number at font size 46 dominates ink-ranked
   features, diluting the serif signal from smaller value lines.

**Aadhaar font_swap at low severity is below the detection floor of
per-line typographic features.** This is a documented known gap.

## Decision

### D1: id_width_cv for DL only (per-doc-type feature selection)

id_width_cv measures glyph width uniformity on the ID-number line. DL has a
pure-digit ID with genuine CV 0.02–0.05 vs swapped 0.16+. PAN's
alphanumeric ID (ABCDE1234F) has varying letter widths even in monospace,
making the feature noisy. Aadhaar's digits show the wrong direction.

Feature selection is per-doc-type: `BASE_FEATURES` (corner_top, mod_top,
adv_cv_min) apply everywhere; `DT_EXTRA` adds id_width_cv for DL only.

### D2: Per-doc-type vote/margin

A global margin=5.0/vote=2 maximizes calibration holdout recall (62.5%) but
regressed the eval: screen_recapture tuning dropped from 33.3% to 20%
(gate failure), and font_swap holdout dropped from 46.7% to 43.3%.

Root cause: vote=2 requires 2 features to breach. PAN/Aadhaar have 3
features—2/3 is a high bar. DL has 4 features—2/4 is more achievable.
And the tighter margin shifted which documents the font_template signal
contributed to, removing incidental screen_recapture detections.

Solution: per-doc-type margin and vote stored as `_vote` in each profile
entry. PAN/Aadhaar use margin=8.0/vote=1 (the proven W6 setting). DL uses
margin=5.0/vote=2 (id_width_cv provides the stable second feature).

### D3: Merged-CC filtering in width CV

Adjacent glyphs in monospace can touch at the pixel level, forming a single
connected component much wider than any individual glyph. These are filtered
(width > 2.5× median) before computing CV.

### D4: Extractor version bump to tf-2

EXTRACTOR_VERSION changed from "tf-1" to "tf-2" because signature() now
returns id_width_cv and uses _id_line_features(). Profiles carrying "tf-1"
are refused at load.

## Measured results

Eval (independent seeds 42/123 and 777/888, 10 docs/type):

| Metric | Pre-W8 | Post-W8 | Delta |
|---|---|---|---|
| font_swap tuning | 46.7% | 53.3% | **+6.7pp** |
| font_swap holdout | 46.7% | 50.0% | **+3.3pp** |
| overall tuning | 73.3% | 74.4% | +1.1pp |
| overall holdout | 75.6% | 76.1% | +0.5pp |
| genuine FPR | 0% | 0% | unchanged |
| screen_recapture | 33.3%/43.3% | 33.3%/43.3% | unchanged |
| undetected_autoclear | 48/44 | 46/43 | -2/-1 |

Font_swap gate ratcheted from 0.40 to 0.47.

## Known gaps (unchanged)

- Aadhaar font_swap at low severity: undetectable with current features
- Needs fundamentally different approach (frequency-domain, more calibration
  data, or OCR-context font analysis)
