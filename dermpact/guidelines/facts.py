"""Three-valued case facts for guideline applicability."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union


class TV(str, Enum):
    """Kleene three-valued truth."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    @classmethod
    def of(cls, value: Any) -> "TV":
        if isinstance(value, TV):
            return value
        if value is None:
            return cls.UNKNOWN
        if value is True:
            return cls.TRUE
        if value is False:
            return cls.FALSE
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "present"):
                return cls.TRUE
            if lowered in ("false", "no", "absent"):
                return cls.FALSE
            if lowered in ("unknown", "", "none", "indeterminate"):
                return cls.UNKNOWN
        raise ValueError(f"cannot interpret {value!r} as a three-valued truth")


def tv_and(*values: TV) -> TV:
    if any(v is TV.FALSE for v in values):
        return TV.FALSE
    if any(v is TV.UNKNOWN for v in values):
        return TV.UNKNOWN
    return TV.TRUE


def tv_or(*values: TV) -> TV:
    if any(v is TV.TRUE for v in values):
        return TV.TRUE
    if any(v is TV.UNKNOWN for v in values):
        return TV.UNKNOWN
    return TV.FALSE


def tv_not(value: TV) -> TV:
    if value is TV.TRUE:
        return TV.FALSE
    if value is TV.FALSE:
        return TV.TRUE
    return TV.UNKNOWN


FACT_SCHEMA: Mapping[str, Mapping[str, Any]] = {
    "image.dermoscopic_present": {"kind": "tv", "source": "record"},
    "image.iqa_state": {"kind": "enum", "values": ("pass", "fail", "unknown"), "source": "record"},
    "image.ood_state": {"kind": "enum", "values": ("in_distribution", "out_of_distribution", "unknown"), "source": "record"},
    "diagnosis.provisional_class": {
        "kind": "enum",
        "values": (
            "melanoma",
            "nevus",
            "basal_cell_carcinoma",
            "squamous_cell_carcinoma",
            "actinic_keratosis_or_intraepithelial_carcinoma",
            "seborrheic_keratosis_or_benign_keratosis",
            "dermatofibroma",
            "vascular_lesion",
            "other",
            "abstained",
        ),
        "source": "record",
    },
    "diagnosis.melanoma_probability": {"kind": "number", "source": "record"},
    "checklist.instrument_id": {"kind": "enum", "values": ("derm7pt_dermoscopic_7point_argenziano", "nice_ng12_weighted_7point", "unknown"), "source": "record"},
    "checklist.score_min": {"kind": "number", "source": "record"},
    "checklist.score_max": {"kind": "number", "source": "record"},
    "checklist.decidable": {"kind": "tv", "source": "record"},
    "checklist.meets_threshold": {"kind": "tv", "source": "record"},
    "checklist.indeterminate_criteria": {"kind": "list", "source": "record"},
    "lesion.under_dermoscopic_assessment": {"kind": "tv", "source": "derived"},
    "lesion.melanoma_suspected": {"kind": "tv", "source": "derived"},
    "lesion.equivocal": {"kind": "tv", "source": "derived"},
    "lesion.direct_checklist_disagreement": {"kind": "tv", "source": "derived"},
    "patient.age_years": {"kind": "number", "source": "observed"},
    "patient.sex": {"kind": "enum", "values": ("female", "male"), "source": "observed"},
    "lesion.site": {
        "kind": "enum",
        "values": (
            "head_neck", "upper_extremity", "lower_extremity", "acral",
            "anterior_torso", "posterior_torso", "trunk_other", "genital_or_mucosal",
        ),
        "source": "observed",
    },
    "lesion.elevation": {"kind": "enum", "values": ("flat", "palpable", "nodular"), "source": "observed"},
    "clinician.melanoma_suspected": {"kind": "tv", "source": "asserted"},
    "clinician.equivocal": {"kind": "tv", "source": "asserted"},
    "lesion.under_assessment": {"kind": "tv", "source": "record"},
    "patient.high_risk_status": {"kind": "tv", "source": "asserted"},
    "patient.total_nevus_count": {"kind": "number", "source": "asserted"},
    "patient.high_total_nevus_count": {"kind": "tv", "source": "asserted_or_derived"},
    "patient.personal_melanoma_history": {"kind": "tv", "source": "asserted"},
    "patient.genetic_predisposition": {"kind": "tv", "source": "asserted"},
    "site.rcm_available": {"kind": "tv", "source": "asserted"},
}

FACT_NAMES: Tuple[str, ...] = tuple(FACT_SCHEMA)


@dataclass(frozen=True)
class FactProvenance:
    """Where a fact value came from.  ``source`` mirrors FACT_SCHEMA ``source``."""

    source: str
    detail: str = ""


FactValue = Union[TV, float, int, str, Tuple[str, ...], None]


@dataclass
class CaseFacts:
    """Facts for exactly one case, keyed by the closed vocabulary."""

    case_id: str
    jurisdiction: str = "EU"
    evaluation_date: date = field(default_factory=date.today)
    values: Dict[str, FactValue] = field(default_factory=dict)
    provenance: Dict[str, FactProvenance] = field(default_factory=dict)

    # -- construction / mutation -----------------------------------------

    def set(self, name: str, value: FactValue, provenance: FactProvenance) -> None:
        spec = FACT_SCHEMA.get(name)
        if spec is None:
            raise KeyError(f"unknown fact name {name!r}")
        kind = spec["kind"]
        if kind == "tv":
            value = TV.of(value)
        elif kind == "number":
            if value is not None and not isinstance(value, (int, float)):
                raise TypeError(f"{name} expects a number, got {value!r}")
        elif kind == "enum":
            if value is not None and value not in spec["values"]:
                raise ValueError(f"{name}: {value!r} not in {spec['values']}")
        elif kind == "list":
            if value is not None:
                value = tuple(str(v) for v in value)
        self.values[name] = value
        self.provenance[name] = provenance

    # -- lookup ---------------------------------------------------------

    def tv(self, name: str) -> TV:
        """Three-valued view of any fact: TV facts as-is; scalars are TRUE if present."""
        if name not in FACT_SCHEMA:
            raise KeyError(f"unknown fact name {name!r}")
        value = self.values.get(name)
        kind = FACT_SCHEMA[name]["kind"]
        if kind == "tv":
            return TV.UNKNOWN if value is None else TV.of(value)
        return TV.UNKNOWN if value is None else TV.TRUE

    def scalar(self, name: str) -> FactValue:
        if name not in FACT_SCHEMA:
            raise KeyError(f"unknown fact name {name!r}")
        return self.values.get(name)

    def is_known(self, name: str) -> bool:
        return self.tv(name) is not TV.UNKNOWN

    def known_names(self) -> Tuple[str, ...]:
        return tuple(n for n in FACT_NAMES if self.is_known(n))

    def unknown_names(self) -> Tuple[str, ...]:
        return tuple(n for n in FACT_NAMES if not self.is_known(n))

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        out_values: Dict[str, Any] = {}
        for name, value in self.values.items():
            if isinstance(value, TV):
                out_values[name] = value.value
            elif isinstance(value, tuple):
                out_values[name] = list(value)
            else:
                out_values[name] = value
        return {
            "case_id": self.case_id,
            "jurisdiction": self.jurisdiction,
            "evaluation_date": self.evaluation_date.isoformat(),
            "values": out_values,
            "provenance": {
                name: {"source": p.source, "detail": p.detail}
                for name, p in self.provenance.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaseFacts":
        facts = cls(
            case_id=str(payload["case_id"]),
            jurisdiction=str(payload.get("jurisdiction", "EU")),
            evaluation_date=date.fromisoformat(str(payload["evaluation_date"])),
        )
        prov = payload.get("provenance", {})
        for name, value in payload.get("values", {}).items():
            p = prov.get(name, {"source": "unspecified", "detail": ""})
            facts.set(name, value, FactProvenance(str(p.get("source", "")), str(p.get("detail", ""))))
        return facts

    def copy(self) -> "CaseFacts":
        return CaseFacts.from_dict(self.to_dict())


__all__ = [
    "FACT_NAMES",
    "FACT_SCHEMA",
    "CaseFacts",
    "FactProvenance",
    "FactValue",
    "TV",
    "tv_and",
    "tv_not",
    "tv_or",
]
