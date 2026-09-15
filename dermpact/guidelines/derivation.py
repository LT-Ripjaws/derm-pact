"""Deterministic derivation of rule-consumable facts from a locked case record."""
from __future__ import annotations

from datetime import date
from typing import Mapping, Optional, Sequence

from dermpact.guidelines.facts import CaseFacts, FactProvenance, TV

try:  # the audited Phase 2 checklist arithmetic
    from dermpact.criteria.aggregate import score_criteria
    from dermpact.criteria.schema import INDETERMINATE, MELANOMA_SUSPICION_THRESHOLD
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError("dermpact.criteria is required for checklist derivation") from exc


DERM7PT_INSTRUMENT = "derm7pt_dermoscopic_7point_argenziano"
HIGH_TOTAL_NEVUS_COUNT_THRESHOLD = 100  # EADO Part 1 §3: "100 nevi and more"
DERIVATION_POLICY_VERSION = "dermpact-phase4-fact-derivation-v1"


def _prov(source: str, detail: str = "") -> FactProvenance:
    return FactProvenance(source, detail)


def derive_case_facts(
    case_id: str,
    *,
    jurisdiction: str = "EU",
    evaluation_date: Optional[date] = None,
    dermoscopic_present: Optional[bool] = True,
    iqa_state: Optional[str] = None,
    ood_state: Optional[str] = None,
    provisional_class: Optional[str] = None,
    melanoma_probability: Optional[float] = None,
    criterion_values: Optional[Mapping[str, str]] = None,
    instrument_id: Optional[str] = DERM7PT_INSTRUMENT,
    sex: Optional[str] = None,
    site: Optional[str] = None,
    elevation: Optional[str] = None,
    age_years: Optional[float] = None,
    clinician_melanoma_suspected: Optional[bool] = None,
    high_risk_status: Optional[bool] = None,
    total_nevus_count: Optional[int] = None,
    high_total_nevus_count: Optional[bool] = None,
    personal_melanoma_history: Optional[bool] = None,
    genetic_predisposition: Optional[bool] = None,
    rcm_available: Optional[bool] = None,
) -> CaseFacts:
    facts = CaseFacts(case_id=case_id, jurisdiction=jurisdiction,
                      evaluation_date=evaluation_date or date.today())

    if dermoscopic_present is not None:
        facts.set("image.dermoscopic_present", dermoscopic_present, _prov("record", "input mask"))
    if iqa_state is not None:
        facts.set("image.iqa_state", iqa_state, _prov("record", "Phase 3 IQA state"))
    if ood_state is not None:
        facts.set("image.ood_state", ood_state, _prov("record", "Phase 3 OOD state"))
    facts.set(
        "lesion.under_dermoscopic_assessment",
        bool(dermoscopic_present) if dermoscopic_present is not None else None,
        _prov("derived", "intended use: one dermoscopic image per case"),
    )
    if instrument_id is not None:
        facts.set("checklist.instrument_id", instrument_id, _prov("record", "checklist instrument"))

    if sex is not None:
        facts.set("patient.sex", sex, _prov("observed"))
    if site is not None:
        facts.set("lesion.site", site, _prov("observed"))
    if elevation is not None:
        facts.set("lesion.elevation", elevation, _prov("observed"))
    if age_years is not None:
        facts.set("patient.age_years", float(age_years), _prov("observed"))

    if clinician_melanoma_suspected is not None:
        facts.set("clinician.melanoma_suspected", clinician_melanoma_suspected, _prov("asserted", "clinician statement"))
    if high_risk_status is not None:
        facts.set("patient.high_risk_status", high_risk_status, _prov("asserted", "clinician statement"))
    if total_nevus_count is not None:
        facts.set("patient.total_nevus_count", int(total_nevus_count), _prov("asserted", "clinician statement"))
    if personal_melanoma_history is not None:
        facts.set("patient.personal_melanoma_history", personal_melanoma_history, _prov("asserted"))
    if genetic_predisposition is not None:
        facts.set("patient.genetic_predisposition", genetic_predisposition, _prov("asserted"))
    if rcm_available is not None:
        facts.set("site.rcm_available", rcm_available, _prov("asserted", "site capability"))

    # high total nevus count: assertion, else derived from a known count
    if high_total_nevus_count is not None:
        facts.set("patient.high_total_nevus_count", high_total_nevus_count, _prov("asserted"))
    elif total_nevus_count is not None:
        facts.set(
            "patient.high_total_nevus_count",
            int(total_nevus_count) >= HIGH_TOTAL_NEVUS_COUNT_THRESHOLD,
            _prov("derived", f"total_nevus_count >= {HIGH_TOTAL_NEVUS_COUNT_THRESHOLD} (EADO Part 1 §3)"),
        )

    perception_usable = (
        iqa_state != "fail"
        and dermoscopic_present is not False
        and (provisional_class is not None or criterion_values is not None)
    )
    blanked_reason = None
    if iqa_state == "fail":
        blanked_reason = "image quality failed: perception-derived facts withheld"
    elif dermoscopic_present is False:
        blanked_reason = "no dermoscopic image: perception-derived facts withheld"

    if provisional_class is not None and perception_usable:
        facts.set("diagnosis.provisional_class", provisional_class, _prov("record", "locked direct diagnosis"))
    if melanoma_probability is not None and perception_usable:
        facts.set("diagnosis.melanoma_probability", float(melanoma_probability), _prov("record"))

    decidable: TV = TV.UNKNOWN
    meets: TV = TV.UNKNOWN
    if criterion_values is not None and perception_usable:
        scored = score_criteria(dict(criterion_values))
        facts.set("checklist.score_min", int(scored.score_min), _prov("derived", "7-point arithmetic"))
        facts.set("checklist.score_max", int(scored.score_max), _prov("derived", "7-point arithmetic"))
        facts.set("checklist.indeterminate_criteria", tuple(scored.indeterminate), _prov("derived"))
        decidable = TV.TRUE if scored.decidable else TV.FALSE
        facts.set("checklist.decidable", decidable, _prov("derived"))
        if scored.decidable:
            meets = TV.TRUE if scored.meets_threshold else TV.FALSE
        facts.set("checklist.meets_threshold", meets, _prov("derived", f"score_min >= {MELANOMA_SUSPICION_THRESHOLD}"))

    prov_is_mel: TV = TV.UNKNOWN
    if provisional_class is not None and perception_usable:
        if provisional_class == "abstained":
            prov_is_mel = TV.UNKNOWN
        else:
            prov_is_mel = TV.TRUE if provisional_class == "melanoma" else TV.FALSE

    # lesion.direct_checklist_disagreement
    if prov_is_mel is TV.UNKNOWN or meets is TV.UNKNOWN:
        disagreement = TV.UNKNOWN
    else:
        disagreement = TV.TRUE if (prov_is_mel is not meets) else TV.FALSE
    facts.set("lesion.direct_checklist_disagreement", disagreement,
              _prov("derived", blanked_reason or "provisional-melanoma XOR checklist-meets"))

    # lesion.melanoma_suspected: clinician precedence, else derivation
    if clinician_melanoma_suspected is not None:
        facts.set("lesion.melanoma_suspected", clinician_melanoma_suspected,
                  _prov("asserted", "clinician assertion takes precedence over derivation"))
    else:
        if not perception_usable:
            derived = TV.UNKNOWN
        elif prov_is_mel is TV.TRUE:
            derived = TV.TRUE
        elif decidable is TV.TRUE:
            derived = meets
        else:
            derived = TV.UNKNOWN
        facts.set("lesion.melanoma_suspected", derived,
                  _prov("derived", blanked_reason or "provisional==melanoma OR decidable checklist meets threshold"))

    # lesion.equivocal
    if not perception_usable:
        equivocal = TV.UNKNOWN
    elif provisional_class == "abstained":
        equivocal = TV.TRUE
    elif decidable is TV.FALSE:
        equivocal = TV.TRUE
    elif disagreement is TV.TRUE:
        equivocal = TV.TRUE
    elif decidable is TV.TRUE and disagreement is TV.FALSE:
        equivocal = TV.FALSE
    else:
        equivocal = TV.UNKNOWN
    facts.set("lesion.equivocal", equivocal,
              _prov("derived", blanked_reason or "undecidable checklist OR direct/checklist disagreement OR abstained"))

    return facts


__all__ = [
    "DERIVATION_POLICY_VERSION",
    "DERM7PT_INSTRUMENT",
    "HIGH_TOTAL_NEVUS_COUNT_THRESHOLD",
    "derive_case_facts",
]
