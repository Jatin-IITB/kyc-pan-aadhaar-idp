# ADR-014: Consolidate Spoof Score Thresholds

## Status
Accepted

## Context
Spoof score thresholds were fragmented across three files with three different values:
- `decide.py`: >0.7 for hard reject, >0.4 for review escalation
- `auto_clear.py`: >0.5 for auto-clear override to reject

These are intentionally different thresholds at different decision points in the pipeline, but having them as raw literals made it impossible to audit the full decisioning policy in one place.

## Decision
Create `services/decisioning/thresholds.py` as a single source of truth for all spoof-related thresholds:
- `SPOOF_REJECT_THRESHOLD = 0.7` — hard reject in decide node (pre-calibration)
- `SPOOF_AUTO_CLEAR_OVERRIDE = 0.5` — override calibration to reject in auto-clear engine
- `SPOOF_REVIEW_THRESHOLD = 0.4` — escalate to human review in decide node (post-calibration)

Both `decide.py` and `auto_clear.py` import from this module.

## Consequences
- All spoof thresholds visible and tunable in one file
- Threshold semantics documented by constant names
- Future tuning requires changing one file, not hunting through multiple modules
