"""Sealed-partition case and truth loaders for the confirmatory H2H run.

Mirrors ``h2h_production.load_development_run_cases`` / ``load_development_truth`` for
the HAM10000 sealed test partition (1,991 images over 1,495 lesions), with three
differences that exist because this partition may be evaluated once:

1. **Every entry point goes through the one-shot guard.** Loading sealed cases logs a
   ``sealed_h2h_run`` access; loading sealed truth logs ``sealed_scoring``. A second
   attempt raises rather than quietly re-running.
2. **Truth is separated from cases by construction.** ``load_sealed_run_cases``
   physically cannot return a label -- it builds ``H2HRunCase`` objects that have no
   label field -- so an arm cannot see an outcome even by accident.
3. **The perception artifact must be a sealed one.** The development binding loader
   asserts ``test_records_loaded == 0``; this one requires the opposite, so the two
   can never be swapped by mistake.

Nothing here runs the probe. The sealed perception artifact must already have been
frozen by ``scripts/dev/freeze_sealed_perception.py``; if it is absent this module
raises rather than falling back to development perception, because silently scoring the
sealed set with development commitments would be undetectable in the outputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from medguard.eval.matched_full_h2h_run import H2HRunCase
from medguard.eval.sealed_access import open_sealed_access
from medguard.evidence.perception import PerceptionBinding, binding_from_prediction_row

ROOT = Path(__file__).resolve().parents[2]
SEALED_MANIFEST = ROOT / "research" / "manifests" / "ham10000_split_v2.json"
SEALED_PERCEPTION = (ROOT / "research" / "experiments" / "medguard_h2h_v1" /
                     "perception" / "dermlip_sealed_v1")
SEALED_PREDICTIONS = SEALED_PERCEPTION / "predictions.jsonl"
SEALED_RECEIPT = SEALED_PERCEPTION / "receipt.json"

SEALED_SPLIT = "test"
EXPECTED_LESIONS = 1495
EXPECTED_IMAGES = 1991


class SealedArtifactError(RuntimeError):
    """Raised when the sealed inputs are missing, mismatched, or the wrong partition."""


def _manifest_rows(split: str = SEALED_SPLIT) -> List[dict]:
    if not SEALED_MANIFEST.is_file():
        raise SealedArtifactError(f"sealed manifest missing: {SEALED_MANIFEST}")
    data = json.loads(SEALED_MANIFEST.read_text(encoding="utf-8"))
    return [row for row in data["records"] if row.get("split") == split]


def sealed_partition_summary(split: str = SEALED_SPLIT) -> Dict[str, int]:
    """Shape of the partition, readable WITHOUT opening sealed access.

    Counting rows is not evaluating them, and a run cannot be planned without knowing
    how big it is. No label is read here.
    """
    rows = _manifest_rows(split)
    return {"images": len(rows), "lesions": len({str(r["cluster_id"]) for r in rows})}


def sealed_perception_status() -> Dict[str, Any]:
    """Is a usable sealed perception artifact present? Checked WITHOUT opening access.

    Exists so a dry run can verify the precondition most likely to be missing. Reading
    a receipt is not evaluating a partition, and discovering the artifact is absent
    only after the one-shot guard has been opened would waste the single run.
    """
    if not SEALED_PREDICTIONS.is_file():
        return {"ready": False, "reason": f"missing {SEALED_PREDICTIONS}"}
    if not SEALED_RECEIPT.is_file():
        return {"ready": False, "reason": f"missing {SEALED_RECEIPT}"}
    receipt = json.loads(SEALED_RECEIPT.read_text(encoding="utf-8"))
    if int(receipt.get("test_records_loaded", 0)) <= 0:
        return {"ready": False, "reason": "receipt reports a development artifact"}
    if receipt.get("truth_fields_present") is not False:
        return {"ready": False, "reason": "artifact is not truth-redacted"}
    rows = sum(1 for line in SEALED_PREDICTIONS.read_text(encoding="utf-8").splitlines()
               if line.strip())
    if rows != EXPECTED_LESIONS:
        return {"ready": False,
                "reason": f"{rows} commitments, expected {EXPECTED_LESIONS}"}
    return {"ready": True, "lesions": rows,
            "images_covered": int(receipt.get("test_records_loaded", 0))}


def _load_sealed_perception() -> Dict[str, PerceptionBinding]:
    if not SEALED_PREDICTIONS.is_file():
        raise SealedArtifactError(
            f"sealed perception artifact missing: {SEALED_PREDICTIONS}. Freeze it first "
            f"with scripts/dev/freeze_sealed_perception.py. This module will NOT fall "
            f"back to development perception -- scoring the sealed set with development "
            f"commitments would not be visible in the outputs."
        )
    receipt = json.loads(SEALED_RECEIPT.read_text(encoding="utf-8"))
    if int(receipt.get("test_records_loaded", 0)) <= 0:
        raise SealedArtifactError(
            "sealed perception receipt reports test_records_loaded <= 0; this is a "
            "development artifact, not a sealed one"
        )
    if receipt.get("truth_fields_present") is not False:
        raise SealedArtifactError(
            "sealed perception artifact claims truth fields are present; a perception "
            "binding must be truth-redacted"
        )
    # View image hashes come from the manifest, joined by case id, and are ordered the
    # same way the perception freeze ordered them -- the binding records the exact
    # images the commitment was computed from, so a swapped or re-encoded image is
    # detectable afterwards.
    views: Dict[str, List[dict]] = {}
    for row in _manifest_rows():
        views.setdefault(str(row["cluster_id"]), []).append(row)
    for rows in views.values():
        rows.sort(key=lambda r: str(r["case_id"]))

    raw = SEALED_PREDICTIONS.read_bytes()
    artifact_sha = hashlib.sha256(raw).hexdigest()
    # Repo-relative when inside the tree, absolute otherwise: a test (or an operator
    # pointing at a copy) may hold the artifact outside the repo, and refusing to name
    # it then would be a crash rather than a check. Third place this has bitten.
    try:
        artifact_id = str(SEALED_PREDICTIONS.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        artifact_id = str(SEALED_PREDICTIONS).replace("\\", "/")

    bindings: Dict[str, PerceptionBinding] = {}
    for raw_line in raw.splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line.decode("utf-8"))
        cluster_id = str(row["cluster_id"])
        source_rows = views.get(cluster_id)
        if not source_rows:
            raise SealedArtifactError(
                f"perception record has no sealed-manifest cluster: {cluster_id}")
        expected = [str(item["case_id"]) for item in source_rows]
        if list(row["view_case_ids"]) != expected or int(row["view_count"]) != len(source_rows):
            raise SealedArtifactError(f"perception view lineage mismatch: {cluster_id}")
        bindings[str(row["case_id"])] = binding_from_prediction_row(
            row, raw_line=raw_line, artifact_id=artifact_id,
            artifact_sha256=artifact_sha, model_id=str(receipt["model_id"]),
            class_order=receipt["class_order"],
            view_image_sha256s=[str(item["image_sha256"]) for item in source_rows],
        )
    if len(bindings) != EXPECTED_LESIONS:
        raise SealedArtifactError(
            f"loaded {len(bindings)} sealed bindings, expected {EXPECTED_LESIONS}")
    return bindings


#: The confirmatory purpose. A smoke test MUST NOT consume it, or the real run would
#: need a logged override to proceed -- which would make a routine dry run look like a
#: protocol breach in the audit trail.
CONFIRMATORY_PURPOSE = "sealed_h2h_run"
SMOKE_PURPOSE = "sealed_smoke"


def load_sealed_run_cases(
    *, detail: str, override_token: Optional[str] = None,
    purpose: str = CONFIRMATORY_PURPOSE, limit: Optional[int] = None,
) -> Tuple[List[H2HRunCase], dict]:
    """The sealed lesions as truth-free run cases. Opens and logs sealed access.

    Returns ``(cases, grant)``. The grant is written into the run report so the
    manuscript can quote the access-log entry that authorised the number it prints.

    ``purpose`` exists so a pipeline smoke test can touch sealed data under its OWN
    logged purpose instead of spending the confirmatory one. ``limit`` restricts the
    number of lesions and is only permitted for a non-confirmatory purpose -- a
    partial confirmatory run is not a confirmatory run, and allowing one would let a
    truncated evaluation be reported as the real thing.
    """
    if limit is not None and purpose == CONFIRMATORY_PURPOSE:
        raise SealedArtifactError(
            "refusing to limit the confirmatory run: a partial sealed evaluation is not "
            f"a sealed evaluation. Use purpose={SMOKE_PURPOSE!r} for a pipeline check."
        )
    if limit is not None and limit <= 0:
        raise SealedArtifactError("limit must be positive")
    summary = sealed_partition_summary()
    if summary["lesions"] != EXPECTED_LESIONS or summary["images"] != EXPECTED_IMAGES:
        raise SealedArtifactError(
            f"sealed partition is {summary}, expected {EXPECTED_IMAGES} images over "
            f"{EXPECTED_LESIONS} lesions; the manifest is not the registered one"
        )
    bindings = _load_sealed_perception()
    try:
        artifact = str(SEALED_PREDICTIONS.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        artifact = str(SEALED_PREDICTIONS).replace("\\", "/")
    grant = open_sealed_access(
        purpose, detail=detail, override_token=override_token,
        extra={"lesions": summary["lesions"], "images": summary["images"],
               "lesions_used": limit if limit is not None else summary["lesions"],
               "confirmatory": purpose == CONFIRMATORY_PURPOSE,
               "perception_artifact": artifact},
    )
    grouped: Dict[str, List[dict]] = {}
    for row in _manifest_rows():
        grouped.setdefault(str(row["cluster_id"]), []).append(row)

    cases: List[H2HRunCase] = []
    for cluster_id in sorted(grouped):
        rows = sorted(grouped[cluster_id], key=lambda r: str(r["case_id"]))
        case_id = f"lesion:{cluster_id}"
        binding = bindings.get(case_id)
        if binding is None:
            raise SealedArtifactError(
                f"no sealed perception commitment for {case_id}; the perception freeze "
                f"does not cover the registered partition"
            )
        cases.append(H2HRunCase(
            case_id=case_id,
            cluster_id=cluster_id,
            perception=binding,
            ordered_view_case_ids=tuple(str(r["case_id"]) for r in rows),
            ordered_image_paths=tuple(str(r["image_path"]) for r in rows),
        ))
    if len(cases) != EXPECTED_LESIONS:
        raise SealedArtifactError(
            f"built {len(cases)} sealed cases, expected {EXPECTED_LESIONS}")
    if limit is not None:
        cases = cases[:limit]
    return cases, grant.as_dict()


def load_sealed_truth(*, detail: str, override_token: Optional[str] = None,
                      purpose: str = "sealed_scoring") -> Dict[str, str]:
    """Sealed labels keyed by cluster id, for the out-of-band scorer ONLY.

    Logged under a separate purpose from the run, so the access log distinguishes
    "we generated on the sealed set" from "we looked at its labels" -- and shows the
    order in which those happened.
    """
    open_sealed_access(
        purpose, detail=detail, override_token=override_token,
        extra={"partition": SEALED_SPLIT},
    )
    truth: Dict[str, str] = {}
    for row in _manifest_rows():
        truth[str(row["cluster_id"])] = str(row["label"])
    if len(truth) != EXPECTED_LESIONS:
        raise SealedArtifactError(
            f"joined {len(truth)} sealed labels, expected {EXPECTED_LESIONS}")
    return truth
