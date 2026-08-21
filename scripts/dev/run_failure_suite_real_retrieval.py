"""Fix 5: genuine F1-F4 recovery rate, measured with REAL re-retrieval over the
frozen indexes (not the fixture tools in eval.failure_typed_suite_v1.run_failure_suite).

Builds the real production dependencies (real frozen text/image/guideline retrieval)
but a generator that must never be called -- this measurement is retrieval + contract
verification only, never generation. ZERO paid calls, regardless of --n.

Usage:
  python run_failure_suite_real_retrieval.py --cache <dir> --out <path> [--n 160]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path("E:/Studies/Research/MedGuard-RAG-Journal")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _NeverCalledClient:
    """A generator client that must never be invoked -- this pass is retrieval-only."""

    model = "unused/never-called"
    call_log: list = []

    def vision(self, *_args, **_kwargs):
        raise AssertionError(
            "fix-5 real-retrieval recovery measurement must never call the generator "
            "(it scores contract verification only, before any generation step)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="frozen-runtime cache dir (offline load)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=160, help="number of dev lesions (default: all)")
    args = ap.parse_args()

    from medguard.eval.failure_typed_suite_v1 import run_failure_suite_real_retrieval
    from medguard.eval.h2h_production import (
        build_production_dependencies,
        load_development_run_cases,
    )
    from medguard.pipeline.generator import VLMGenerator

    cases = load_development_run_cases()[: args.n]
    print(f"loaded {len(cases)} real development lesions")
    print("building production dependencies (heavy offline load, zero paid calls)...")
    deps = build_production_dependencies(
        args.cache, generator=VLMGenerator(_NeverCalledClient(), max_schema_repairs=0))

    report = run_failure_suite_real_retrieval(cases, deps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"wrote {args.out}")
    for stratum, block in report["per_stratum"].items():
        print(stratum, block)


if __name__ == "__main__":
    main()
