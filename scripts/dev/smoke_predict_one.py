"""Identity smoke test for the single-image entry point.

Feeds a valid-split case's RAW dermoscopy JPG through predict_one (fresh encode, no
feature caches) and compares the resulting contract against the promoted cohort
pipeline's serialized contract for the same case (cdss_contracts_valid_promoted.json).

PASS = same diagnosis, same criteria values, same score range, confidence within
float tolerance. That equality is the end-to-end claim: the deployable path and the
evaluated path are the same system.

Also asserts the domain gate does NOT fire on these in-domain valid cases (gated
must be False and the domain block must carry the bank hash): the gate exists to
downgrade out-of-domain inputs, and a gate that fired in-domain would silently
replace the evaluated dual-view contract this smoke exists to verify.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from medguard_v2.data.derm7pt import load_cases
from medguard_v2.eval.agent_eval import _load_promoted_map
from medguard_v2.pipeline.predict_one import predict_one

N_CASES = 3


def main() -> None:
    emap = _load_promoted_map()

    valid = [c for c in load_cases() if c.split == "valid"]
    picked = [valid[i] for i in (0, len(valid) // 2, len(valid) - 1)][:N_CASES]

    failures = []
    for case in picked:
        cid = f"derm7pt-{case.case_num}"
        ref = emap[cid]
        pred = predict_one(
            case.derm_path,
            sex=str(getattr(case, "sex", "unknown") or "unknown"),
            location=str(getattr(case, "location", "unknown") or "unknown"),
            elevation=str(getattr(case, "elevation", "unknown") or "unknown"),
            case_id=cid,
        )
        ref_criteria = {c["name"]: c["value"] for c in ref["criteria"]}
        checks = {
            "diagnosis": (pred.diagnosis, ref["classifier"]["diagnosis"]),
            "confidence": (round(pred.confidence, 3), round(ref["classifier"]["confidence"], 3)),
            "criteria": (pred.criteria, ref_criteria),
            "score_min": (pred.contract.score.score_min, ref["seven_point_score"]["min"]),
            "score_max": (pred.contract.score.score_max, ref["seven_point_score"]["max"]),
            # In-domain valid cases must NOT be domain-gated; the block must be present + hashed.
            "domain_not_gated": (pred.domain.get("gated"), False),
            "domain_bank_hash_present": (bool(pred.domain.get("bank_hash")), True),
        }
        bad = {k: v for k, v in checks.items() if v[0] != v[1]}
        status = "PASS" if not bad else f"FAIL {bad}"
        print(f"{cid}: {status}")
        if bad:
            failures.append((cid, bad))

    if failures:
        raise SystemExit(f"IDENTITY SMOKE FAILED on {len(failures)}/{len(picked)} cases")
    print(f"\nIDENTITY SMOKE PASSED ({len(picked)}/{len(picked)}): the single-image path "
          "reproduces the evaluated pipeline's contracts from raw JPGs.")
    print("\n--- sample rendered report (first case, first 40 lines) ---")
    print("\n".join(predict_one(
        picked[0].derm_path, case_id=f"derm7pt-{picked[0].case_num}"
    ).report_markdown.splitlines()[:40]))


if __name__ == "__main__":
    main()
