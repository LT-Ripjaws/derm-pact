"""Create or verify the pre-sealed runtime lock.

  python build_runtime_lock.py --build    # once, before the sealed run
  python build_runtime_lock.py --verify   # before the run, and again when scoring

Offline; zero paid calls; authorizes nothing.
"""
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

from medguard.eval.ragas_judge import judge_specification_sha256  # noqa: E402
from medguard.eval.runtime_lock import (  # noqa: E402
    build_lock,
    verify_judge_specification,
    verify_lock,
)

LOCK_PATH = "research/RUNTIME_LOCK_V1.json"
GATE_PATH = "research/evidence/dev_h2h_full/dev_h2h_full_160x11_v4.abstention_gate.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--lock", default=LOCK_PATH)
    ap.add_argument("--gate", default=GATE_PATH)
    args = ap.parse_args()
    if args.build == args.verify:
        ap.error("choose exactly one of --build or --verify")

    lock_file = PROJECT_ROOT / args.lock

    if args.build:
        lock = build_lock(
            PROJECT_ROOT,
            abstention_gate_path=args.gate,
            judge_specification_sha256=judge_specification_sha256(),
            spec_id="medguard-derm-evidence-contract-h2h-v6",
            identical_arms_for_noise_floor=["M", "A0"],
            created="2026-08-08",
        )
        lock_file.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        print(f"locked {lock['source_tree']['n_files']} source files")
        print(f"  source digest    {lock['source_tree']['digest']}")
        print(f"  artifact digest  {lock['frozen_artifacts']['digest']}")
        print(f"  card digest      {lock['implementation_cards']['digest']}")
        print(f"  gate             {lock['abstention_gate']['sha256'][:16]} "
              f"({lock['abstention_gate']['path']})")
        print(f"  judge spec       {lock['scorers']['ragas_judge_specification_sha256'][:16]}")
        print(f"wrote {args.lock}")
        return

    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    result = verify_lock(PROJECT_ROOT, lock)
    verify_judge_specification(lock, judge_specification_sha256())
    print(f"RUNTIME LOCK VERIFIED: {result['source_files']} source files, "
          f"digest {result['source_digest'][:16]}")
    print("judge specification unchanged")
    print("\nThe sealed run remains a separate, explicit, single-use authorization.")


if __name__ == "__main__":
    main()
