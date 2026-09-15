"""WP-B: image-backed applicability benchmark rebuilt under reviewed EADO v2 semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dermpact.core import versions as V
from dermpact.eval import dermpact_phase2_4b_inputs as inputs_mod
from dermpact.eval import dermpact_phase4_arms as arms_mod
from dermpact.eval import dermpact_phase4_evaluation as eval_mod
from dermpact.eval import dermpact_phase4_reference_v2 as ref2
from dermpact.eval import dermpact_phase4_rule_catalog as catalog_mod
from dermpact.guidelines.derivation import DERM7PT_INSTRUMENT
from dermpact.guidelines.engine import EntityMismatchError
from dermpact.guidelines.facts import CaseFacts, TV
from dermpact.guidelines.reviewed_v2 import ReviewedApplicabilityEngine
from dermpact.guidelines.reviewed_v2 import derive_case_facts as derive_case_facts_v2

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research/manifests/dermpact_phase6/applicability_benchmark_v2"
FRIEND_META = ROOT / "friend-cuda/research/manifests"
STAGE2_OOF_PATH = ROOT / "research/experiments/dermpact_phase2_5/stage2_cefix_v1/stage2_oof.npz"
EVAL_DATE = date(2026, 9, 6)
N_CASES = 616
N_BOOT = 10000
BOOT_SEED = 20260906
WRONG_ENTITY_SENTINEL = "derm7pt-case-000000"
VALID_FROM = "2024-11-28"
VALID_UNTIL = "2026-12-31"
INDETERMINATE_BAND = 0.15

GLOBAL_CLASSES = (
    "melanoma", "nevus", "basal_cell_carcinoma", "squamous_cell_carcinoma",
    "actinic_keratosis_or_intraepithelial_carcinoma", "seborrheic_keratosis_or_benign_keratosis",
    "dermatofibroma", "vascular_lesion", "other",
)
CRITERIA = (
    "pigment_network", "streaks", "pigmentation", "regression_structures",
    "dots_and_globules", "blue_whitish_veil", "vascular_structures",
)
POS_VALUE = {"pigment_network": "atypical", "blue_whitish_veil": "present", "vascular_structures": "dotted",
             "streaks": "irregular", "pigmentation": "diffuse irregular", "dots_and_globules": "irregular",
             "regression_structures": "blue areas"}
NEG_VALUE = {"pigment_network": "typical", "blue_whitish_veil": "absent", "vascular_structures": "absent",
             "streaks": "absent", "pigmentation": "absent", "dots_and_globules": "regular",
             "regression_structures": "absent"}
ALL_POSITIVE_CRITERIA = {name: POS_VALUE[name] for name in CRITERIA}

RULE_FULL_ID = {short: f"eado-mel-p1-2024:{short}" for short in ("R1", "R3", "R4", "R5", "R6", "R7")}
ACTION_CODE = {
    "R3": "histopathological_confirmation",
    "R4": "dermatoscopic_assessment",
    "R5": "whole_body_photography_surveillance",
    "R6": "sequential_digital_dermatoscopy",
    "R7": "reflectance_confocal_microscopy",
}
NEXT_STEP_PRIORITY = ("R3", "R7", "R4", "R6", "R5")
QUERY_RULE_SHORTS = ("R3", "R7", "R4", "R6", "R5")
DECISIVE_ORDER: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    "R3": (("clinician_melanoma_suspected", "clinician.melanoma_suspected"),),
    "R4": (("lesion_under_assessment", "lesion.under_assessment"),),
    "R5": (("high_risk_status", "patient.high_risk_status"),),
    "R6": (("high_risk_status", "patient.high_risk_status"),
           ("high_total_nevus_count", "patient.high_total_nevus_count")),
    "R7": (("clinician_equivocal", "clinician.equivocal"),
           ("rcm_available", "site.rcm_available")),
}
_DECISIVE_DEFAULTS = {"lesion_under_assessment": True}

#: Fixed variant list applied to every one of the 616 base cases (exact denominators).
VARIANTS: Tuple[Dict[str, Any], ...] = (
    {"variant": "base", "family": "base", "overrides": {}},
    {"variant": "suspected_true", "family": "assertion", "overrides": {"clinician_melanoma_suspected": True}},
    {"variant": "suspected_false", "family": "negated", "overrides": {"clinician_melanoma_suspected": False}},
    {"variant": "equivocal_true_rcm_unknown", "family": "r7_combo",
     "overrides": {"clinician_melanoma_suspected": False, "clinician_equivocal": True}},
    {"variant": "equivocal_true_rcm_true", "family": "r7_combo",
     "overrides": {"clinician_melanoma_suspected": False, "clinician_equivocal": True, "rcm_available": True}},
    {"variant": "equivocal_true_rcm_false", "family": "r7_combo",
     "overrides": {"clinician_melanoma_suspected": False, "clinician_equivocal": True, "rcm_available": False}},
    {"variant": "equivocal_false_rcm_true", "family": "r7_combo",
     "overrides": {"clinician_melanoma_suspected": False, "clinician_equivocal": False, "rcm_available": True}},
    {"variant": "equivocal_unknown_rcm_true", "family": "r7_combo",
     "overrides": {"clinician_melanoma_suspected": False, "rcm_available": True}},
    {"variant": "under_assessment_false", "family": "negated", "overrides": {"lesion_under_assessment": False}},
    {"variant": "high_risk_true_count_unknown", "family": "r6_combo", "overrides": {"high_risk_status": True}},
    {"variant": "high_risk_true_high_count_true", "family": "r6_combo",
     "overrides": {"high_risk_status": True, "high_total_nevus_count": True}},
    {"variant": "high_risk_true_count_250_no_assertion", "family": "boundary",
     "overrides": {"high_risk_status": True, "total_nevus_count": 250}},
    {"variant": "high_risk_false", "family": "negated", "overrides": {"high_risk_status": False}},
    {"variant": "iqa_review_required_with_suspected_true", "family": "reliability",
     "overrides": {"clinician_melanoma_suspected": True, "iqa_state": "fail"}},
    {"variant": "no_image_no_assertions", "family": "missing", "overrides": {"dermoscopic_present": False}},
    {"variant": "no_image_suspected_true", "family": "missing",
     "overrides": {"dermoscopic_present": False, "clinician_melanoma_suspected": True}},
    {"variant": "model_melanoma_no_assertion", "family": "distractor",
     "overrides": {"provisional_class": "melanoma", "melanoma_probability": 0.97}},
    {"variant": "checklist_high_no_assertion", "family": "distractor",
     "overrides": {"criterion_values": ALL_POSITIVE_CRITERIA}},
    {"variant": "stale_2027", "family": "temporal", "overrides": {"evaluation_date": date(2027, 1, 1)}},
    {"variant": "jurisdiction_us", "family": "wrong_jurisdiction", "overrides": {"jurisdiction": "US"}},
    {"variant": "wrong_entity", "family": "wrong_entity", "overrides": {}},
    {"variant": "scope_query_staging", "family": "out_of_scope", "overrides": {}},
)




def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def criterion_values_from_logits(logits: Sequence[float]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for name, logit in zip(CRITERIA, logits):
        p = _sigmoid(float(logit))
        if abs(p - 0.5) < INDETERMINATE_BAND:
            values[name] = "indeterminate"
        else:
            values[name] = POS_VALUE[name] if p >= 0.5 else NEG_VALUE[name]
    return values


def _serialise_kwargs(kw: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in kw.items():
        out[k] = v.isoformat() if isinstance(v, date) else v
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _atomic_json(path: Path, payload: Any) -> str:
    return _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _relative_or_absolute(path: Path) -> str:
    """Repository-relative path when possible; an absolute path otherwise (e.g. a test's ``tmp_path`` output directory outside the repository)."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _load_checked_npz(path: Path) -> Mapping[str, np.ndarray]:
    sidecar = Path(str(path) + ".sha256")
    if _sha256_file(path) != sidecar.read_text(encoding="utf-8").strip():
        raise AssertionError(f"changed artifact: {path}")
    return np.load(path, allow_pickle=False)


def resolve_inputs() -> inputs_mod.Phase24BInputSet:
    """Resolve the 616 locked development cases exactly as ``dermpact/eval/dermpact_phase3_assessed_records.py`` does: friend-cuda manifest paths, ``require_files=False`` (no images opened)."""
    return inputs_mod.resolve_development_inputs(
        inventory_path=FRIEND_META / "dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json",
        fold_manifest_path=FRIEND_META / "dermpact_phase2_1/dermpact_phase2_1_development_folds_v1.json",
        clinical_photo_audit_path=FRIEND_META / "dermpact_phase2_3/clinical_photo_features_v1/clinical_photo_feature_protocol.json",
        structured_fact_support_path=FRIEND_META / "dermpact_phase2_3/structured_facts_v1/structured_fact_support.json",
        require_files=False,
    )




def decisive_fact_for(short: str, kw: Mapping[str, Any]) -> Optional[str]:
    """First UNKNOWN fact (in the rule's own required-fact order) for a rule whose reference state is ``applicability_unknown``."""
    for kwname, factname in DECISIVE_ORDER[short]:
        value = kw.get(kwname, _DECISIVE_DEFAULTS.get(kwname))
        if value is None:
            return factname
    return None


def compute_gold_for_rules(kw: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Gold four-state applicability for R3/R4/R5/R6/R7, from the independent reference transcription (``dermpact_phase4_reference_v2.reference_states``)."""
    ref_kw = dict(
        suspected=kw.get("clinician_melanoma_suspected"),
        equivocal=kw.get("clinician_equivocal"),
        high_risk=kw.get("high_risk_status"),
        high_count=kw.get("high_total_nevus_count"),
        rcm_available=kw.get("rcm_available"),
        lesion_under_assessment=kw.get("lesion_under_assessment", True),
        jurisdiction=kw.get("jurisdiction", "EU"),
        date_iso=kw.get("evaluation_date", EVAL_DATE).isoformat(),
    )
    states = ref2.reference_states(**ref_kw)
    gold: Dict[str, Dict[str, Any]] = {}
    for short, state in states.items():
        full_id = RULE_FULL_ID[short]
        entry: Dict[str, Any] = {"state": state, "action_code": None, "decisive_missing_fact": None}
        if state == "applicable":
            entry["action_code"] = ACTION_CODE[short]
        elif state == "applicability_unknown":
            entry["decisive_missing_fact"] = decisive_fact_for(short, kw)
        gold[full_id] = entry
    return gold


def next_step_walk(per_rule: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """v2 next-step priority walk: the first APPLICABLE rule anywhere in ``NEXT_STEP_PRIORITY`` wins (``recommended``); else the first UNKNOWN-with-question wins (``conditional``); else ``withheld_no_coverage``."""
    for short in NEXT_STEP_PRIORITY:
        entry = per_rule.get(RULE_FULL_ID[short])
        if entry is not None and entry.get("state") == "applicable":
            return {"status": "recommended", "rule_id": RULE_FULL_ID[short],
                     "action_code": entry.get("action_code"), "decisive_missing_fact": None}
    for short in NEXT_STEP_PRIORITY:
        entry = per_rule.get(RULE_FULL_ID[short])
        if entry is not None and entry.get("state") == "applicability_unknown":
            return {"status": "conditional", "rule_id": RULE_FULL_ID[short],
                     "action_code": None, "decisive_missing_fact": entry.get("decisive_missing_fact")}
    return {"status": "withheld_no_coverage", "rule_id": None, "action_code": None, "decisive_missing_fact": None}




@dataclass
class BuiltBenchmark:
    runs: List[Dict[str, Any]]
    inputs_meta: inputs_mod.Phase24BInputSet
    oof_sha256: str


def build_runs() -> BuiltBenchmark:
    data = resolve_inputs()
    if len(data.cases) != N_CASES:
        raise AssertionError(f"expected {N_CASES} development cases, got {len(data.cases)}")
    oof = _load_checked_npz(STAGE2_OOF_PATH)
    oof_sha256 = _sha256_file(STAGE2_OOF_PATH)
    if oof["derm_diagnosis_logits"].shape[0] != N_CASES:
        raise AssertionError("stage2 OOF row count does not match the locked development set")

    runs: List[Dict[str, Any]] = []
    for index, case in enumerate(data.cases):
        cid = case.group_id
        diag_logits = oof["derm_diagnosis_logits"][index]
        provisional_class = GLOBAL_CLASSES[int(np.argmax(diag_logits))]
        melanoma_probability = _sigmoid(float(oof["derm_melanoma_logits"][index]))
        criterion_values = criterion_values_from_logits(oof["derm_criteria_logits"][index])
        base_kw: Dict[str, Any] = dict(
            jurisdiction="EU", evaluation_date=EVAL_DATE, dermoscopic_present=True, iqa_state="pass",
            provisional_class=provisional_class, melanoma_probability=melanoma_probability,
            criterion_values=criterion_values, instrument_id=DERM7PT_INSTRUMENT,
            sex=case.facts.sex, site=case.facts.site, elevation=case.facts.elevation,
        )
        for spec in VARIANTS:
            variant = spec["variant"]
            kw = dict(base_kw)
            kw.update(spec["overrides"])
            facts = derive_case_facts_v2(cid, **kw)

            if variant == "wrong_entity":
                query_rule_ids = [RULE_FULL_ID[s] for s in QUERY_RULE_SHORTS]
                gold = compute_gold_for_rules(kw)
                gold_next_step = next_step_walk(gold)
                expected_case_id = WRONG_ENTITY_SENTINEL
                expected_refusal = True
            elif variant == "scope_query_staging":
                query_rule_ids = [RULE_FULL_ID["R1"]]
                gold = {RULE_FULL_ID["R1"]: {"state": "no_catalogue_coverage", "action_code": None,
                                              "decisive_missing_fact": None}}
                gold_next_step = {"status": "withheld_no_coverage", "rule_id": None,
                                   "action_code": None, "decisive_missing_fact": None}
                expected_case_id = cid
                expected_refusal = False
            else:
                query_rule_ids = [RULE_FULL_ID[s] for s in QUERY_RULE_SHORTS]
                gold = compute_gold_for_rules(kw)
                gold_next_step = next_step_walk(gold)
                expected_case_id = cid
                expected_refusal = False

            row = {
                "run_id": f"{cid}::{variant}",
                "case_id": cid,
                "fold": case.fold,
                "variant": variant,
                "family": spec["family"],
                "query_rule_ids": query_rule_ids,
                "facts": facts.to_dict(),
                "facts_kwargs": _serialise_kwargs(kw),
                "expected_case_id": expected_case_id,
                "expected_refusal": expected_refusal,
                "gold": gold,
                "gold_next_step": gold_next_step,
                "perception_context": {
                    "provisional_class": provisional_class,
                    "melanoma_probability": round(melanoma_probability, 6),
                },
            }
            runs.append(row)
    return BuiltBenchmark(runs=runs, inputs_meta=data, oof_sha256=oof_sha256)




def run_deterministic_reviewed_v2(engine: ReviewedApplicabilityEngine, run: Mapping[str, Any]) -> Dict[str, Any]:
    facts = CaseFacts.from_dict(run["facts"])
    try:
        case = engine.evaluate_case(facts, expected_case_id=run["expected_case_id"], rule_ids=run["query_rule_ids"])
    except EntityMismatchError:
        return {"arm": "deterministic_reviewed_v2", "run_id": run["run_id"], "refused": True, "per_rule": {},
                "next_step": {"status": "refused", "rule_id": None, "action_code": None, "decisive_missing_fact": None}}
    per_rule = {r.rule_id: {"state": r.state.value, "action_code": r.action_code,
                             "decisive_missing_fact": r.decisive_missing_fact} for r in case.results}
    if run["variant"] == "scope_query_staging":
        next_step = {"status": "withheld_no_coverage", "rule_id": None, "action_code": None, "decisive_missing_fact": None}
    else:
        next_step = next_step_walk(per_rule)
    return {"arm": "deterministic_reviewed_v2", "run_id": run["run_id"], "refused": False,
            "per_rule": per_rule, "next_step": next_step}


def run_deterministic_forms_template(run: Mapping[str, Any]) -> Dict[str, Any]:
    """A second, independent implementation of the same six-fact boolean logic (no ``ApplicabilityEngine``/``RuleCatalogue`` objects): the same-capability control the brief asks for."""
    facts = CaseFacts.from_dict(run["facts"])
    if facts.case_id != run["expected_case_id"]:
        return {"arm": "deterministic_forms_template", "run_id": run["run_id"], "refused": True, "per_rule": {},
                "next_step": {"status": "refused", "rule_id": None, "action_code": None, "decisive_missing_fact": None}}
    if run["variant"] == "scope_query_staging":
        per_rule = {RULE_FULL_ID["R1"]: {"state": "no_catalogue_coverage", "action_code": None,
                                          "decisive_missing_fact": None}}
        return {"arm": "deterministic_forms_template", "run_id": run["run_id"], "refused": False,
                "per_rule": per_rule,
                "next_step": {"status": "withheld_no_coverage", "rule_id": None, "action_code": None,
                              "decisive_missing_fact": None}}

    covered = facts.jurisdiction == "EU" and VALID_FROM <= facts.evaluation_date.isoformat() <= VALID_UNTIL
    if not covered:
        per_rule = {RULE_FULL_ID[s]: {"state": "no_catalogue_coverage", "action_code": None,
                                       "decisive_missing_fact": None} for s in QUERY_RULE_SHORTS}
        return {"arm": "deterministic_forms_template", "run_id": run["run_id"], "refused": False,
                "per_rule": per_rule,
                "next_step": {"status": "withheld_no_coverage", "rule_id": None, "action_code": None,
                              "decisive_missing_fact": None}}

    def _tv_state(name: str) -> str:
        v = facts.tv(name)
        if v is TV.TRUE:
            return "applicable"
        if v is TV.FALSE:
            return "not_applicable"
        return "applicability_unknown"

    def _and_state(name_a: str, name_b: str) -> str:
        a, b = facts.tv(name_a), facts.tv(name_b)
        if a is TV.FALSE or b is TV.FALSE:
            return "not_applicable"
        if a is TV.UNKNOWN or b is TV.UNKNOWN:
            return "applicability_unknown"
        return "applicable"

    def _decisive(names: Sequence[str]) -> Optional[str]:
        for n in names:
            if not facts.is_known(n):
                return n
        return None

    states = {
        "R3": _tv_state("clinician.melanoma_suspected"),
        "R4": _tv_state("lesion.under_assessment"),
        "R5": _tv_state("patient.high_risk_status"),
        "R6": _and_state("patient.high_risk_status", "patient.high_total_nevus_count"),
        "R7": _and_state("clinician.equivocal", "site.rcm_available"),
    }
    fact_names = {
        "R3": ("clinician.melanoma_suspected",),
        "R4": ("lesion.under_assessment",),
        "R5": ("patient.high_risk_status",),
        "R6": ("patient.high_risk_status", "patient.high_total_nevus_count"),
        "R7": ("clinician.equivocal", "site.rcm_available"),
    }
    per_rule: Dict[str, Dict[str, Any]] = {}
    for short, state in states.items():
        full_id = RULE_FULL_ID[short]
        entry: Dict[str, Any] = {"state": state, "action_code": None, "decisive_missing_fact": None}
        if state == "applicable":
            entry["action_code"] = ACTION_CODE[short]
        elif state == "applicability_unknown":
            entry["decisive_missing_fact"] = _decisive(fact_names[short])
        per_rule[full_id] = entry
    next_step = next_step_walk(per_rule)
    return {"arm": "deterministic_forms_template", "run_id": run["run_id"], "refused": False,
            "per_rule": per_rule, "next_step": next_step}




def score_run(run: Mapping[str, Any], out: Mapping[str, Any]) -> Dict[str, Any]:
    gold = run["gold"]
    gold_next = run["gold_next_step"]
    per_rule = out.get("per_rule", {})
    refused = bool(out.get("refused"))
    expected_refusal = bool(run.get("expected_refusal"))

    state_triples: List[Tuple[str, str, str]] = []
    wrong_release = 0
    n_release_opportunities = 0
    if not refused:
        for rid, g in gold.items():
            p = per_rule.get(rid, {"state": "missing"})
            arm_state = p.get("state", "missing")
            state_triples.append((rid, g["state"], arm_state))
            if g["state"] != "applicable":
                n_release_opportunities += 1
                if arm_state == "applicable":
                    wrong_release += 1

    missing_hit: Optional[int] = None
    if gold_next["status"] == "conditional":
        if refused:
            missing_hit = 0
        else:
            arm_next = out.get("next_step", {})
            missing_hit = int(arm_next.get("status") == "conditional"
                               and arm_next.get("decisive_missing_fact") == gold_next["decisive_missing_fact"])

    entity_refusal_correct: Optional[int] = int(refused) if expected_refusal else None

    no_coverage_correct: Optional[int] = None
    source_expiry_correct: Optional[int] = None
    if run["variant"] in ("stale_2027", "jurisdiction_us", "scope_query_staging"):
        if refused:
            no_coverage_correct = 0
        else:
            no_coverage_correct = int(all(per_rule.get(rid, {}).get("state") == "no_catalogue_coverage" for rid in gold))
        if run["variant"] == "stale_2027":
            source_expiry_correct = no_coverage_correct

    if expected_refusal:
        wrong_next_step = int(not refused)
    elif refused:
        wrong_next_step = 1
    else:
        arm_next = out.get("next_step", {})
        if gold_next["status"] == "recommended":
            wrong_next_step = int(not (arm_next.get("status") == "recommended"
                                        and arm_next.get("rule_id") == gold_next["rule_id"]
                                        and arm_next.get("action_code") == gold_next["action_code"]))
        else:
            wrong_next_step = int(arm_next.get("status") == "recommended")

    return {
        "run_id": run["run_id"], "case_id": run["case_id"], "variant": run["variant"], "family": run["family"],
        "state_triples": state_triples, "wrong_release": wrong_release,
        "n_release_opportunities": n_release_opportunities,
        "missing_hit": missing_hit, "entity_refusal_correct": entity_refusal_correct,
        "no_coverage_correct": no_coverage_correct, "source_expiry_correct": source_expiry_correct,
        "wrong_or_inapplicable_next_step": wrong_next_step, "refused": int(refused),
    }


def _rate(values: Sequence[Optional[int]]) -> Tuple[float, int]:
    vals = [v for v in values if v is not None]
    return (float(sum(vals)) / len(vals) if vals else float("nan")), len(vals)


def summarise_arm(scored: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    all_triples = [t for s in scored for t in s["state_triples"]]
    exact = [g == a for _, g, a in all_triples]
    confusion: Dict[str, Counter] = defaultdict(Counter)
    for rid, g, a in all_triples:
        confusion[rid][f"{g}->{a}"] += 1
    total_release_opps = sum(s["n_release_opportunities"] for s in scored)
    total_wrong_release = sum(s["wrong_release"] for s in scored)
    primary, _ = _rate([s["wrong_or_inapplicable_next_step"] for s in scored])
    missing_rate, n_missing = _rate([s["missing_hit"] for s in scored])
    entity_rate, n_entity = _rate([s["entity_refusal_correct"] for s in scored])
    coverage_rate, n_coverage = _rate([s["no_coverage_correct"] for s in scored])
    expiry_rate, n_expiry = _rate([s["source_expiry_correct"] for s in scored])
    return {
        "n_runs": len(scored),
        "four_state_exact_match_rate": float(sum(exact) / len(exact)) if exact else float("nan"),
        "n_state_pairs": len(exact),
        "per_rule_confusion": {rid: dict(c) for rid, c in sorted(confusion.items())},
        "wrong_or_inapplicable_next_step_rate": primary,
        "wrong_or_inapplicable_next_step_count": int(sum(s["wrong_or_inapplicable_next_step"] for s in scored)),
        "release_when_not_applicable_rate": (float(total_wrong_release) / total_release_opps) if total_release_opps else float("nan"),
        "n_release_opportunities": total_release_opps,
        "decisive_missing_fact_hit_rate": missing_rate, "n_missing_fact_opportunities": n_missing,
        "entity_refusal_rate": entity_rate, "n_entity_refusal_opportunities": n_entity,
        "no_catalogue_coverage_correctness_rate": coverage_rate, "n_no_coverage_opportunities": n_coverage,
        "source_expiry_correctness_rate": expiry_rate, "n_source_expiry_opportunities": n_expiry,
        "refused_count": int(sum(s["refused"] for s in scored)),
    }




def _sources_manifest() -> List[Dict[str, str]]:
    files = [
        Path(__file__).resolve(),
        ROOT / "dermpact/guidelines/reviewed_v2.py",
        ROOT / "dermpact/eval/dermpact_phase4_reference_v2.py",
        ROOT / "dermpact/data/corpus/eado_melanoma_part1_rules_v2.jsonl",
        ROOT / "research/manifests/dermpact_phase4/rule_catalog_v2/catalog_manifest.json",
        STAGE2_OOF_PATH,
        FRIEND_META / "dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json",
        FRIEND_META / "dermpact_phase2_1/dermpact_phase2_1_development_folds_v1.json",
        FRIEND_META / "dermpact_phase2_3/clinical_photo_features_v1/clinical_photo_feature_protocol.json",
        FRIEND_META / "dermpact_phase2_3/structured_facts_v1/structured_fact_support.json",
    ]
    return [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256_file(p)} for p in files]


def freeze_protocol(out_path: Path, *, llm_mode: str, limit: Optional[int], llm_model: str) -> Dict[str, Any]:
    protocol = {
        "schema_version": "dermpact-phase6-applicability-benchmark-v2-protocol-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_base_cases": N_CASES,
        "n_variants_per_case": len(VARIANTS),
        "variants": [{"variant": v["variant"], "family": v["family"],
                      "overrides": _serialise_kwargs(v["overrides"])} for v in VARIANTS],
        "arms": ["deterministic_reviewed_v2", "deterministic_forms_template",
                 "direct_llm", "naive_rag", "llm_traversal"],
        "llm_mode": llm_mode, "llm_limit": limit, "llm_model": llm_model,
        "next_step_priority": list(NEXT_STEP_PRIORITY),
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "cluster": "case_id"},
        "sources": _sources_manifest(),
        "hard_scope": {"protected_or_final_rows_loaded": 0, "derm7pt_official_test_rows_loaded": 0,
                       "ham_rows_loaded": 0, "bcn_rows_loaded": 0, "ph2_rows_loaded": 0, "images_opened": 0},
    }
    _atomic_json(out_path, protocol)
    return protocol


def write_result_markdown(path: Path, protocol: Mapping[str, Any], gold_summary: Mapping[str, Any],
                           results: Mapping[str, Any]) -> None:
    lines = [
        "# DermPact WP-B -- applicability benchmark v2 (reviewed EADO Part 1 semantics)", "",
        f"Schema: `{results['schema_version']}`  ",
        f"n base cases: {gold_summary['n_base_cases']}  n variants: {gold_summary['n_variants']}  "
        f"n runs: {gold_summary['n_runs']}  ",
        f"LLM mode: `{results['llm_mode']}`  limit: {results['limit']}", "",
        "## Deterministic arms (reportable)", "",
        "| Arm | n runs | wrong/inapplicable-next-step rate [95% CI] | four-state exact match | "
        "missing-fact hit | entity refusal | no-coverage correctness | source-expiry correctness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("deterministic_reviewed_v2", "deterministic_forms_template"):
        s = results["arms"][arm]
        ci = s["primary_ci"]
        boundary = ci.get("boundary_case_bound_one_sided_95")
        boundary_text = (f"; one-sided case-level 95% {boundary['kind']} bound {boundary['value']:.4f}"
                         if boundary else "")
        lines.append(
            f"| {arm} | {s['n_runs']} | {s['wrong_or_inapplicable_next_step_rate']:.4f} "
            f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]{boundary_text} | {s['four_state_exact_match_rate']:.4f} | "
            f"{s['decisive_missing_fact_hit_rate']:.4f} | {s['entity_refusal_rate']:.4f} | "
            f"{s['no_catalogue_coverage_correctness_rate']:.4f} | {s['source_expiry_correctness_rate']:.4f} |"
        )
    lines += ["", "## LLM arms -- plumbing checks only (FakeLLMClient), NOT results", "",
              "| Arm | n runs scored | LLM calls | prompt tokens | completion tokens |",
              "|---|---:|---:|---:|---:|"]
    for arm in ("direct_llm", "naive_rag", "llm_traversal"):
        s = results["arms"].get(arm)
        if s is None:
            continue
        lines.append(f"| {arm} | {s['n_runs']} | {s['llm']['calls']} | {s['llm']['prompt_tokens']} | "
                     f"{s['llm']['completion_tokens']} |")
    lines += ["", "## Deferred by user: live LLM comparison", "",
              "```", str(results["deferred_live_llm_command"]), "```", ""]
    if results.get("llm_cost_estimate"):
        ce = results["llm_cost_estimate"]
        lines += [
            f"Fake-run plumbing sample: {ce['fake_run_n_calls']} calls, {ce['fake_run_prompt_tokens']} prompt + "
            f"{ce['fake_run_completion_tokens']} completion tokens ({ce['avg_tokens_per_call']:.1f} tokens/call average).",
            f"Full-corpus extrapolation ({ce['full_corpus_estimated_calls']} calls across 3 LLM arms x "
            f"{gold_summary['n_runs']} runs): ~{ce['full_corpus_estimated_tokens']:.0f} tokens.",
            f"Pricing: {ce['pricing']}.", "",
        ]
    lines += ["## Notes", ""]
    for note in results["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-mode", choices=["none", "fake", "live"], default="none")
    parser.add_argument("--model", default=V.PRIMARY_LIVE_AGENT_MODEL,
                        help="model locked to the previous-paper generator by default")
    parser.add_argument("--limit", type=int, default=None,
                        help="limit the number of runs scored by the LLM plumbing arms (cost control)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-suffix", type=str, default="",
                        help="suffix inserted before each output file's extension (byte-identity reruns)")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    suffix = args.out_suffix

    protocol_path = out_dir / f"protocol{suffix}.json"
    runs_path = out_dir / f"runs{suffix}.jsonl"
    gold_summary_path = out_dir / f"gold_summary{suffix}.json"
    results_path = out_dir / f"results{suffix}.json"
    result_md_path = out_dir / f"RESULT{suffix}.md"

    print("[1/6] protocol freeze", flush=True)
    protocol = freeze_protocol(protocol_path, llm_mode=args.llm_mode, limit=args.limit, llm_model=args.model)

    print("[2/6] build benchmark (616 development cases x variants)", flush=True)
    built = build_runs()
    runs = built.runs
    if len({r["case_id"] for r in runs}) != N_CASES:
        raise AssertionError("case coverage changed")
    runs_text = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in runs) + "\n"
    runs_sha = _atomic_write(runs_path, runs_text)

    print("[3/6] gold summary", flush=True)
    family_counts: Counter = Counter(r["family"] for r in runs)
    gold_state_counts: Dict[str, Counter] = defaultdict(Counter)
    next_step_status_counts: Counter = Counter()
    for r in runs:
        for rid, g in r["gold"].items():
            gold_state_counts[rid][g["state"]] += 1
        next_step_status_counts[r["gold_next_step"]["status"]] += 1
    gold_summary = {
        "n_base_cases": N_CASES, "n_variants": len(VARIANTS), "n_runs": len(runs),
        "family_counts": dict(sorted(family_counts.items())),
        "gold_state_counts_per_rule": {rid: dict(c) for rid, c in sorted(gold_state_counts.items())},
        "gold_next_step_status_counts": dict(next_step_status_counts),
        "protected_or_final_rows_loaded": 0,
        "artifact": {"path": _relative_or_absolute(runs_path), "sha256": runs_sha},
        "oof_sha256": built.oof_sha256,
        "inventory_sha256": built.inputs_meta.inventory_sha256,
        "fold_manifest_sha256": built.inputs_meta.fold_manifest_sha256,
    }
    _atomic_json(gold_summary_path, gold_summary)

    print("[4/6] deterministic arms (full 616 x variants set)", flush=True)
    engine = ReviewedApplicabilityEngine()
    det_outputs: Dict[str, List[Dict[str, Any]]] = {"deterministic_reviewed_v2": [], "deterministic_forms_template": []}
    for r in runs:
        det_outputs["deterministic_reviewed_v2"].append(run_deterministic_reviewed_v2(engine, r))
        det_outputs["deterministic_forms_template"].append(run_deterministic_forms_template(r))

    print("[5/6] LLM plumbing arms (FakeLLMClient only)", flush=True)
    llm_outputs: Dict[str, List[Dict[str, Any]]] = {}
    llm_cost_estimate: Optional[Dict[str, Any]] = None
    if args.llm_mode != "none":
        pages = catalog_mod.load_source_pages()["pages"]
        llm = arms_mod.FakeLLMClient() if args.llm_mode == "fake" else arms_mod.LiveGatewayClient(model=args.model)
        arms = arms_mod.Arms(engine.catalogue, source_pages=pages, llm=llm)
        subset = runs[: args.limit] if args.limit else runs
        for name in ("direct_llm", "naive_rag", "llm_traversal"):
            llm_outputs[name] = [arms.run_arm(name, r).to_dict() for r in subset]
        total_pt = sum(o.get("prompt_tokens", 0) for outs in llm_outputs.values() for o in outs)
        total_ct = sum(o.get("completion_tokens", 0) for outs in llm_outputs.values() for o in outs)
        n_scored = sum(len(v) for v in llm_outputs.values())
        avg_tokens_per_call = (total_pt + total_ct) / n_scored if n_scored else 0.0
        est_total_calls = len(runs) * 3
        llm_cost_estimate = {
            "fake_run_n_calls": n_scored, "fake_run_prompt_tokens": total_pt, "fake_run_completion_tokens": total_ct,
            "avg_tokens_per_call": avg_tokens_per_call,
            "full_corpus_estimated_calls": est_total_calls,
            "full_corpus_estimated_tokens": avg_tokens_per_call * est_total_calls,
            "pricing": "PLACEHOLDER_RATE_USD_PER_1K_TOKENS -- user must fill",
        }

    deferred_command = (
        f"./venv_p2/Scripts/python.exe -m dermpact.core.eval.applicability_benchmark_v2 "
        f"--llm-mode live --limit {args.limit or 5} --model {args.model}  "
        f"# model explicitly frozen; refuses unless "
        f"DERMPACT_ALLOW_LIVE_LLM=1 and a funded gateway key are present; NOT run in this delegation"
    )

    print("[6/6] scoring + receipts", flush=True)
    results: Dict[str, Any] = {
        "schema_version": "dermpact-phase6-applicability-benchmark-v2-results-v1",
        "llm_mode": args.llm_mode, "limit": args.limit,
        "n_runs_total": len(runs),
        "arms": {},
        "deferred_live_llm_command": deferred_command,
        "llm_cost_estimate": llm_cost_estimate,
        "notes": [
            "v1 results (research/manifests/dermpact_phase4/applicability_benchmark_v1) are not v2 results.",
            "Derm7pt 616 stage2 OOF rows are each case's single held-out member, not the five-model deployment ensemble.",
            "deterministic_reviewed_v2 and deterministic_forms_template are the only reportable scientific arms; "
            "direct_llm/naive_rag/llm_traversal are plumbing checks only (FakeLLMClient), never results.",
        ],
    }
    for arm_name, outs in det_outputs.items():
        scored = [score_run(r, o) for r, o in zip(runs, outs)]
        summary = summarise_arm(scored)
        summary["primary_ci"] = eval_mod.clustered_bootstrap_rate(scored, n_boot=N_BOOT, seed=BOOT_SEED)
        results["arms"][arm_name] = summary
    for arm_name, outs in llm_outputs.items():
        subset_runs = runs[: len(outs)]
        scored = [score_run(r, o) for r, o in zip(subset_runs, outs)]
        summary = summarise_arm(scored)
        summary["label"] = "plumbing_check -- not a result"
        summary["llm"] = {"calls": sum(o.get("n_llm_calls", 0) for o in outs),
                          "prompt_tokens": sum(o.get("prompt_tokens", 0) for o in outs),
                          "completion_tokens": sum(o.get("completion_tokens", 0) for o in outs)}
        results["arms"][arm_name] = summary

    _atomic_json(results_path, results)
    write_result_markdown(result_md_path, protocol, gold_summary, results)
    print(json.dumps({
        "n_runs": len(runs),
        "deterministic_reviewed_v2_primary": results["arms"]["deterministic_reviewed_v2"]["wrong_or_inapplicable_next_step_rate"],
        "deterministic_forms_template_primary": results["arms"]["deterministic_forms_template"]["wrong_or_inapplicable_next_step_rate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
