"""Independent reference decision table for the EADO Part 1 executable rules."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

_MAJOR = {"pigment_network": ({"atypical"}, 2), "blue_whitish_veil": ({"present"}, 2),
          "vascular_structures": ({"dotted", "linear irregular"}, 2)}
_MINOR = {"streaks": ({"irregular"}, 1),
          "pigmentation": ({"diffuse irregular", "localized irregular"}, 1),
          "dots_and_globules": ({"irregular"}, 1),
          "regression_structures": ({"blue areas", "white areas", "combinations"}, 1)}
_THRESHOLD = 3
_INDET = "indeterminate"
NEVUS_HIGH = 100
VALID_FROM = date(2024, 11, 28)
VALID_UNTIL = date(2026, 12, 31)
COVERED_JURISDICTIONS = ("EU",)
DERM7PT_INSTRUMENT = "derm7pt_dermoscopic_7point_argenziano"

U = "unknown"


def score_range(values: Mapping[str, str]) -> Tuple[int, int, Tuple[str, ...]]:
    lo = hi = 0
    indet = []
    for name, (scoring, pts) in {**_MAJOR, **_MINOR}.items():
        v = values[name]
        if v == _INDET:
            hi += pts
            indet.append(name)
        elif v in scoring:
            lo += pts
            hi += pts
    return lo, hi, tuple(indet)


@dataclass
class RefInput:
    case_id: str
    jurisdiction: str = "EU"
    evaluation_date: date = date(2026, 9, 1)
    dermoscopic_present: Optional[bool] = True
    iqa_state: Optional[str] = None
    provisional_class: Optional[str] = None
    criterion_values: Optional[Mapping[str, str]] = None
    instrument_id: Optional[str] = DERM7PT_INSTRUMENT
    clinician_melanoma_suspected: Optional[bool] = None
    high_risk_status: Optional[bool] = None
    total_nevus_count: Optional[int] = None
    high_total_nevus_count: Optional[bool] = None
    rcm_available: Optional[bool] = None


def _tv(b: Optional[bool]) -> str:
    return U if b is None else ("true" if b else "false")


def derive(inp: RefInput) -> Dict[str, str]:
    """Reference fact derivation (mirror of guidelines/derivation.py policy)."""
    usable = inp.iqa_state != "fail" and inp.dermoscopic_present is not False and (
        inp.provisional_class is not None or inp.criterion_values is not None)
    f: Dict[str, str] = {}
    f["image.dermoscopic_present"] = _tv(inp.dermoscopic_present)
    f["lesion.under_dermoscopic_assessment"] = _tv(inp.dermoscopic_present)
    decidable = U
    meets = U
    if inp.criterion_values is not None and usable:
        lo, hi, _ = score_range(inp.criterion_values)
        dec = lo >= _THRESHOLD or hi < _THRESHOLD
        decidable = "true" if dec else "false"
        if dec:
            meets = "true" if lo >= _THRESHOLD else "false"
    prov_is_mel = U
    if inp.provisional_class is not None and usable and inp.provisional_class != "abstained":
        prov_is_mel = "true" if inp.provisional_class == "melanoma" else "false"
    disagreement = U if (prov_is_mel == U or meets == U) else ("true" if prov_is_mel != meets else "false")
    if inp.clinician_melanoma_suspected is not None:
        suspected = _tv(inp.clinician_melanoma_suspected)
    elif not usable:
        suspected = U
    elif prov_is_mel == "true":
        suspected = "true"
    elif decidable == "true":
        suspected = meets
    else:
        suspected = U
    if not usable:
        equivocal = U
    elif inp.provisional_class == "abstained":
        equivocal = "true"
    elif decidable == "false":
        equivocal = "true"
    elif disagreement == "true":
        equivocal = "true"
    elif decidable == "true" and disagreement == "false":
        equivocal = "false"
    else:
        equivocal = U
    if inp.high_total_nevus_count is not None:
        high_nevus = _tv(inp.high_total_nevus_count)
    elif inp.total_nevus_count is not None:
        high_nevus = "true" if inp.total_nevus_count >= NEVUS_HIGH else "false"
    else:
        high_nevus = U
    f.update({
        "lesion.melanoma_suspected": suspected,
        "lesion.equivocal": equivocal,
        "patient.high_risk_status": _tv(inp.high_risk_status),
        "patient.high_total_nevus_count": high_nevus,
        "site.rcm_available": _tv(inp.rcm_available),
        "checklist.instrument_id": inp.instrument_id or U,
    })
    return f


# Decision tables: ordered prerequisite fact names per rule.
RULE_PREREQS: Mapping[str, Tuple[str, ...]] = {
    "eado-mel-p1-2024:R3": ("image.dermoscopic_present", "lesion.melanoma_suspected"),
    "eado-mel-p1-2024:R4": ("lesion.under_dermoscopic_assessment",),
    "eado-mel-p1-2024:R5": ("patient.high_risk_status",),
    "eado-mel-p1-2024:R6": ("patient.high_risk_status", "patient.high_total_nevus_count"),
    "eado-mel-p1-2024:R7": ("image.dermoscopic_present", "lesion.equivocal", "site.rcm_available"),
}
RULE_ACTION: Mapping[str, str] = {
    "eado-mel-p1-2024:R3": "histopathological_confirmation",
    "eado-mel-p1-2024:R4": "dermatoscopic_assessment",
    "eado-mel-p1-2024:R5": "whole_body_photography_surveillance",
    "eado-mel-p1-2024:R6": "sequential_digital_dermatoscopy",
    "eado-mel-p1-2024:R7": "reflectance_confocal_microscopy",
}
RULE_INSTRUMENT_BOUND = {"eado-mel-p1-2024:R3", "eado-mel-p1-2024:R7"}
CHECKLIST_DERIVED = ("lesion.melanoma_suspected", "lesion.equivocal")


def evaluate_rule(rule_id: str, inp: RefInput) -> Dict[str, Any]:
    if not (VALID_FROM <= inp.evaluation_date <= VALID_UNTIL):
        return {"state": "no_catalogue_coverage", "reason": "source_stale_or_superseded", "action": None, "missing": None}
    if inp.jurisdiction not in COVERED_JURISDICTIONS:
        return {"state": "no_catalogue_coverage", "reason": "jurisdiction_not_covered", "action": None, "missing": None}
    facts = derive(inp)
    instrument_mismatch = rule_id in RULE_INSTRUMENT_BOUND and facts["checklist.instrument_id"] != DERM7PT_INSTRUMENT
    if instrument_mismatch:
        facts = dict(facts)
        for name in CHECKLIST_DERIVED:
            # the clinician assertion for suspicion survives blanking (it is not checklist-derived)
            if name == "lesion.melanoma_suspected" and inp.clinician_melanoma_suspected is not None:
                continue
            facts[name] = U
    prereqs = RULE_PREREQS[rule_id]
    vals = [facts[p] for p in prereqs]
    if any(v == "false" for v in vals):
        return {"state": "not_applicable", "reason": "prerequisite_false", "action": None, "missing": None}
    if all(v == "true" for v in vals):
        return {"state": "applicable", "reason": "prerequisites_satisfied", "action": RULE_ACTION[rule_id], "missing": None}
    missing = next(p for p, v in zip(prereqs, vals) if v == U)
    reason = "instrument_mismatch_blanked_facts" if instrument_mismatch else "decisive_fact_unknown"
    return {"state": "applicability_unknown", "reason": reason, "action": RULE_ACTION[rule_id], "missing": missing}


def evaluate_all(inp: RefInput) -> Dict[str, Dict[str, Any]]:
    return {rid: evaluate_rule(rid, inp) for rid in RULE_PREREQS}


__all__ = ["RefInput", "RULE_ACTION", "RULE_PREREQS", "derive", "evaluate_all", "evaluate_rule", "score_range"]
