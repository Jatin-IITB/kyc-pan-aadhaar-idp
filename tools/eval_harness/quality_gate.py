import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Path to metrics.json from an eval run.")
    ap.add_argument("--baseline", required=True, help="Path to baseline metrics.json committed in repo.")
    ap.add_argument("--min-ok-rate", type=float, default=0.99)
    ap.add_argument("--min-valid-rate", type=float, default=0.90)
    ap.add_argument(
        "--allow-valid-drop",
        type=float,
        default=0.005,
        help="Allowed absolute drop in valid_rate vs baseline (e.g., 0.005 = 0.5%).",
    )
    args = ap.parse_args()

    m = _load_json(Path(args.metrics))
    b = _load_json(Path(args.baseline))

    ok_rate = float(m.get("ok_rate", 0.0))
    valid_rate = float(m.get("valid_rate", 0.0))

    b_valid = float(b.get("valid_rate", 0.0))

    failures = []
    if ok_rate < args.min_ok_rate:
        failures.append(f"ok_rate {ok_rate:.4f} < {args.min_ok_rate:.4f}")
    if valid_rate < args.min_valid_rate:
        failures.append(f"valid_rate {valid_rate:.4f} < {args.min_valid_rate:.4f}")
    if (b_valid - valid_rate) > args.allow_valid_drop:
        failures.append(
            f"valid_rate dropped {b_valid - valid_rate:.4f} > {args.allow_valid_drop:.4f} "
            f"(baseline={b_valid:.4f}, current={valid_rate:.4f})"
        )

    if failures:
        print("QUALITY GATE FAILED")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)

    print("QUALITY GATE PASSED")
    print(f"ok_rate={ok_rate:.4f}, valid_rate={valid_rate:.4f} (baseline={b_valid:.4f})")


if __name__ == "__main__":
    main()
