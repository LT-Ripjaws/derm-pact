"""Phase 4.5: image-backed applicability benchmark with controlled fact variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from dermpact.guidelines.derivation import DERM7PT_INSTRUMENT, derive_case_facts
from dermpact.eval import dermpact_phase4_reference_table as ref
from dermpact.eval import dermpact_phase2_4b_inputs as inputs_mod
from dermpact.eval import dermpact_phase2_baselines as baselines


ROOT = Path(__file__).resolve().parents[2]
OOF_RAW = ROOT / "research/experiments/dermpact_phase2_4b/paired_photo_v1/canonical_oof_raw.json"
OUT_DIR = ROOT / "research/manifests/dermpact_phase4/applicability_benchmark_v1"
EVAL_DATE = date(2026, 9, 1)
INDETERMINATE_BAND = 0.15   # |p - 0.5| < band -> criterion indeterminate (benchmark convention)
GLOBAL_CLASSES = (
    "melanoma", "nevus", "basal_cell_carcinoma", "squamous_cell_carcinoma",
    "actinic_keratosis_or_intraepithelial_carcinoma", "seborrheic_keratosis_or_benign_keratosis",
    "dermatofibroma", "vascular_lesion", "other",
)
CRITERIA = tuple(baselines.CRITERIA)
POS_VALUE = {"pigment_network": "atypical", "blue_whitish_veil": "present", "vascular_structures": "dotted",
             "streaks": "irregular", "pigmentation": "diffuse irregular", "dots_and_globules": "irregular",
             "regression_structures": "blue areas"}
NEG_VALUE = {"pigment_network": "typical", "blue_whitish_veil": "absent", "vascular_structures": "absent",
             "streaks": "absent", "pigmentation": "absent", "dots_and_globules": "regular",
             "regression_structures": "absent"}
EXEC_RULES = tuple(ref.RULE_PREREQS)
OUT_OF_SCOPE_QUERY_RULE = "eado-mel-p1-2024:R1"

#: Fixed variant list applied to every base case (exact denominators).
VARIANTS: Sequence[Dict[str, Any]] = (
    {"variant": "base", "family": "base", "overrides": {}},
    {"variant": "clinician_suspected_true", "family": "conflicting_or_assertion", "overrides": {"clinician_melanoma_suspected": True}},
    {"variant": "clinician_suspected_false", "family": "negated", "overrides": {"clinician_melanoma_suspected": False}},
    {"variant": "rcm_available", "family": "fact_supplied", "overrides": {"rcm_available": True}},
    {"variant": "rcm_unavailable", "family": "negated", "overrides": {"rcm_available": False}},
    {"variant": "high_risk_true", "family": "fact_supplied", "overrides": {"high_risk_status": True}},
    {"variant": "high_risk_false", "family": "negated", "overrides": {"high_risk_status": False}},
    {"variant": "high_risk_nevus_100", "family": "boundary", "overrides": {"high_risk_status": True, "total_nevus_count": 100}},
    {"variant": "high_risk_nevus_99", "family": "boundary", "overrides": {"high_risk_status": True, "total_nevus_count": 99}},
    {"variant": "iqa_fail", "family": "reliability", "overrides": {"iqa_state": "fail"}},
    {"variant": "no_image", "family": "missing", "overrides": {"dermoscopic_present": False}},
    {"variant": "wrong_instrument", "family": "wrong_instrument", "overrides": {"instrument_id": "nice_ng12_weighted_7point"}},
    {"variant": "stale_2027", "family": "temporal", "overrides": {"evaluation_date": date(2027, 1, 1)}},
    {"variant": "jurisdiction_us", "family": "wrong_jurisdiction", "overrides": {"jurisdiction": "US"}},
    {"variant": "wrong_entity", "family": "wrong_entity", "overrides": {}},
    {"variant": "scope_query_staging", "family": "out_of_scope", "overrides": {}},
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(text, encoding="utf-8")
    with tmp.open("r+b") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    digest = _sha256_file(path)
    Path(str(path) + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def criterion_values_from_logits(logits: Sequence[float]) -> Dict[str, str]:
    values = {}
    for name, logit in zip(CRITERIA, logits):
        p = _sigmoid(float(logit))
        if abs(p - 0.5) < INDETERMINATE_BAND:
            values[name] = "indeterminate"
        else:
            values[name] = POS_VALUE[name] if p >= 0.5 else NEG_VALUE[name]
    return values


def criterion_values_from_raw_labels(raw: Mapping[str, str]) -> Dict[str, str]:
    return {name: str(raw[name]) for name in CRITERIA}


def load_locked_perception() -> Dict[str, Dict[str, Any]]:
    raw = json.loads(OOF_RAW.read_text(encoding="utf-8"))
    truth = raw["truth"]
    tp = raw["conditions"]["true_pair"]
    ids = list(truth["ids"])
    if tp["ids"] != ids:
        raise AssertionError("true_pair ids are not aligned with truth ids")
    out: Dict[str, Dict[str, Any]] = {}
    for i, cid in enumerate(ids):
        diag_logits = list(tp["diagnosis_logits"][i])
        if len(diag_logits) != len(GLOBAL_CLASSES):
            raise AssertionError("diagnosis logit width changed")
        out[cid] = {
            "provisional_class": GLOBAL_CLASSES[int(np.argmax(diag_logits))],
            "melanoma_probability": _sigmoid(float(tp["melanoma_logits"][i])),
            "criterion_values": criterion_values_from_logits(tp["criterion_logits"][i]),
            "true_diagnosis": str(truth["diagnosis_targets"][i]),
            "true_melanoma": bool(truth["melanoma_targets"][i]),
        }
    return out


def _base_kwargs(case: inputs_mod.Phase24BCaseInput, perception: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(
        jurisdiction="EU", evaluation_date=EVAL_DATE, dermoscopic_present=True, iqa_state="pass",
        ood_state="in_distribution",
        provisional_class=perception["provisional_class"],
        melanoma_probability=perception["melanoma_probability"],
        criterion_values=perception["criterion_values"],
        instrument_id=DERM7PT_INSTRUMENT,
        sex=case.facts.sex, site=case.facts.site, elevation=case.facts.elevation, age_years=case.facts.age_years,
    )


def _ref_input(case_id: str, kw: Mapping[str, Any]) -> ref.RefInput:
    return ref.RefInput(
        case_id=case_id, jurisdiction=kw["jurisdiction"], evaluation_date=kw["evaluation_date"],
        dermoscopic_present=kw.get("dermoscopic_present", True), iqa_state=kw.get("iqa_state"),
        provisional_class=kw.get("provisional_class"), criterion_values=kw.get("criterion_values"),
        instrument_id=kw.get("instrument_id", DERM7PT_INSTRUMENT),
        clinician_melanoma_suspected=kw.get("clinician_melanoma_suspected"),
        high_risk_status=kw.get("high_risk_status"), total_nevus_count=kw.get("total_nevus_count"),
        high_total_nevus_count=kw.get("high_total_nevus_count"), rcm_available=kw.get("rcm_available"),
    )


def _serialise_kwargs(kw: Mapping[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in kw.items():
        out[k] = v.isoformat() if isinstance(v, date) else v
    return out


def build_benchmark(*, out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    inputs = inputs_mod.resolve_development_inputs()
    perception = load_locked_perception()
    records = baselines._derm7pt_records()
    raw_by_id = {str(r["group_id"]): r for r in records}
    if len(inputs.cases) != 616:
        raise AssertionError("expected 616 development cases")
    lines: List[str] = []
    n_runs = 0
    family_counts: Dict[str, int] = {}
    for case in inputs.cases:
        cid = case.group_id
        if cid not in perception:
            raise AssertionError(f"no locked perception for {cid}")
        base = _base_kwargs(case, perception[cid])
        true_view = dict(base, provisional_class=perception[cid]["true_diagnosis"],
                         criterion_values=criterion_values_from_raw_labels(raw_by_id[cid]["criteria_raw"]))
        for spec in VARIANTS:
            kw = dict(base)
            kw.update(spec["overrides"])
            facts = derive_case_facts(cid, **kw)
            gold = ref.evaluate_all(_ref_input(cid, kw))
            gold_true = ref.evaluate_all(_ref_input(cid, dict(true_view, **spec["overrides"])))
            expected_refusal = spec["variant"] == "wrong_entity"
            query_rule_ids = [OUT_OF_SCOPE_QUERY_RULE] if spec["variant"] == "scope_query_staging" else list(EXEC_RULES)
            if spec["variant"] == "scope_query_staging":
                gold = {OUT_OF_SCOPE_QUERY_RULE: {"state": "no_catalogue_coverage", "reason": "rule_status_out_of_scope",
                                                 "action": None, "missing": None}}
                gold_true = dict(gold)
            row = {
                "run_id": f"{cid}::{spec['variant']}",
                "case_id": cid,
                "fold": case.fold,
                "variant": spec["variant"],
                "family": spec["family"],
                "query_rule_ids": query_rule_ids,
                "facts": facts.to_dict(),
                "facts_kwargs": _serialise_kwargs(kw),
                "expected_case_id": "derm7pt-case-000000" if expected_refusal else cid,
                "expected_refusal": expected_refusal,
                "gold": gold,
                "gold_true_label_view": gold_true,
                "perception_context": {
                    "provisional_class": perception[cid]["provisional_class"],
                    "true_diagnosis": perception[cid]["true_diagnosis"],
                    "melanoma_probability": round(perception[cid]["melanoma_probability"], 6),
                },
            }
            lines.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
            n_runs += 1
            family_counts[spec["family"]] = family_counts.get(spec["family"], 0) + 1
    bench_path = out_dir / "benchmark.jsonl"
    bench_sha = _atomic_write(bench_path, "\n".join(lines) + "\n")
    gold_counts: Dict[str, int] = {}
    for ln in lines:
        row = json.loads(ln)
        for rid, g in row["gold"].items():
            gold_counts[g["state"]] = gold_counts.get(g["state"], 0) + 1
    manifest = {
        "schema_version": "dermpact-phase4-applicability-benchmark-manifest-v1",
        "status": "PASS",
        "n_base_cases": len(inputs.cases),
        "n_variants_per_case": len(VARIANTS),
        "n_runs": n_runs,
        "family_counts": family_counts,
        "gold_state_counts": gold_counts,
        "conventions": {
            "indeterminate_band": INDETERMINATE_BAND,
            "criterion_positive_representatives": POS_VALUE,
            "criterion_negative_representatives": NEG_VALUE,
            "provisional_class": "argmax over the 9 global diagnosis logits (frozen Phase 2.4B true_pair OOF)",
            "evaluation_date": EVAL_DATE.isoformat(),
            "gold_source": "dermpact/eval/dermpact_phase4_reference_table.py (independent of runtime engine)",
            "true_label_view": "report-only: gold recomputed from raw criterion labels + true diagnosis",
        },
        "sources": {
            "locked_oof_raw": {"path": str(OOF_RAW.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256_file(OOF_RAW)},
            "inventory_sha256": inputs.inventory_sha256,
            "fold_manifest_sha256": inputs.fold_manifest_sha256,
        },
        "artifact": {"path": str(bench_path.relative_to(ROOT)).replace("\\", "/"), "sha256": bench_sha},
        "hard_scope_checks": {
            "derm7pt_development_rows": len(inputs.cases),
            "derm7pt_official_test_rows_loaded": 0,
            "ham_rows_loaded": 0, "bcn_rows_loaded": 0, "ph2_rows_loaded": 0,
            "images_opened": 0, "models_loaded": 0,
        },
        "builder": {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
                    "sha256": _sha256_file(Path(__file__).resolve())},
    }
    manifest_sha = _atomic_write(out_dir / "benchmark_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "manifest_sha256": manifest_sha}


def load_benchmark(*, out_dir: Path = OUT_DIR) -> List[Dict[str, Any]]:
    manifest = json.loads((out_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    path = ROOT / manifest["artifact"]["path"]
    if _sha256_file(path) != manifest["artifact"]["sha256"]:
        raise RuntimeError("benchmark file hash mismatch")
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    if not args.build:
        parser.error("use --build")
    result = build_benchmark()
    print(json.dumps({k: result[k] for k in ("status", "n_base_cases", "n_variants_per_case", "n_runs",
                                              "family_counts", "gold_state_counts", "hard_scope_checks", "manifest_sha256")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
