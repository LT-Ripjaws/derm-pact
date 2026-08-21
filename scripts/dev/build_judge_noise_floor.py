"""Measure the RAGAS judge's test-retest floor from the byte-identical arms, then
re-read every real judge-based comparison against it.  Offline; zero paid calls."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medguard.eval.judge_noise_floor import (  # noqa: E402
    DEFAULT_IDENTICAL_ARMS,
    apply_floor,
    measure_floor,
    verify_identical,
)
from medguard.eval.paired_endpoints import (  # noqa: E402
    load_case_series,
    paired_endpoint_difference,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=",".join(DEFAULT_IDENTICAL_ARMS))
    ap.add_argument("--comparators", default="B1,B4,B5,B6")
    ap.add_argument("--nboot", type=int, default=2000)
    args = ap.parse_args()

    from medguard.eval.h2h_production import load_development_truth

    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    scored = json.loads(Path(args.scored).read_text(encoding="utf-8"))
    truth = load_development_truth()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    identity = verify_identical(raw, arms)
    print(f"identity proven: {identity['arms']} over {identity['cases']} cases")

    series = {a: load_case_series(raw, scored, a, truth) for a in arms}
    floor = measure_floor(series, n_boot=args.nboot)

    print("\n=== judge test-retest floor (differences between IDENTICAL systems) ===")
    for endpoint, block in floor["endpoints"].items():
        print(f"{endpoint:>40}  max|diff|={block['max_abs_null_difference']:.4f}  "
              f"false-positives={block['false_positives']}/{block['comparisons_run']}")
        for null in block["null_comparisons"]:
            flag = "  <-- FALSE POSITIVE" if null["false_positive"] else ""
            print(f"{'':>42}{null['pair']}: {null['estimate']:+.4f} "
                  f"[{null['low']:+.4f}, {null['high']:+.4f}] p={null['p_value']:.4f}{flag}")

    # Re-read the real comparisons against the floor.
    reread = {}
    for comparator in [c.strip() for c in args.comparators.split(",") if c.strip()]:
        if comparator not in scored["per_arm"]:
            continue
        other = load_case_series(raw, scored, comparator, truth)
        rows = []
        for endpoint in floor["endpoints"]:
            if endpoint not in series["M"] or endpoint not in other:
                continue
            result = paired_endpoint_difference(
                series["M"][endpoint], other[endpoint], n_boot=args.nboot)
            if result is not None:
                rows.append(apply_floor(result, floor))
        reread[comparator] = rows

    print("\n=== real comparisons re-read against the floor ===")
    for comparator, rows in reread.items():
        print(f"\nM vs {comparator}:")
        for r in rows:
            verdict = "SURVIVES" if r["above_judge_noise_floor"] else "BELOW FLOOR"
            ratio = r.get("effect_to_noise_ratio")
            print(f"  {r['metric']:>40} {r['estimate']:+.4f}  floor={r['judge_noise_floor']:.4f}  "
                  f"ratio={ratio}x  {verdict}")

    payload = {"identity": identity, "floor": floor, "reread_against_floor": reread}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
