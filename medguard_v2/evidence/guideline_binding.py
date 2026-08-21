"""Bind a committed condition to an authoritative recommendation, or say there isn't one.

The catalog is the one carried over from v1: 66 recommendations extracted verbatim from current
clinical practice guidelines, each pinned to a source, version, citation and content hash.  Nothing
here paraphrases, summarises or infers -- a recommendation is either reproduced exactly and
verifiably, or it is reported as absent.

**Coverage is the honest part.**  The catalog holds recommendations for basal cell carcinoma,
melanoma and actinic keratosis.  Derm7pt contains no actinic keratosis, so against this dataset the
catalog reaches **BCC and melanoma only** -- 294 of 1011 cases (117 of the 395-case test
split).  Naevi, seborrhoeic
keratoses and the miscellaneous group have no formal guideline backing, and the correct behaviour
for those is to say so.

That is not a defect to engineer around.  v1's central finding was that formal dermatology
guidelines concentrate on malignancy, and that a system claiming guideline grounding must publish
the denominator rather than quietly generating management text where no guidance exists.  The same
wall appears here on a second dataset, which is a replication, not a surprise.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO / "medguard" / "data" / "corpus" / "derm_guideline_recommendations_v1.jsonl"

#: Derm7pt diagnosis group -> catalog condition. Absence is deliberate and reported.
GROUP_TO_CONDITION: Mapping[str, Optional[str]] = {
    "BCC": "basal_cell_carcinoma",
    "MEL": "melanoma",
    "NEV": None,
    "SK": None,
    "MISC": None,
}

#: Human-readable names for the committed diagnosis groups. Lives here -- next to
#: GROUP_TO_CONDITION, the module that owns the group vocabulary -- because both the render
#: layer and the single-view action derivation need it, and two copies would drift.
GROUP_FULL_NAME: Mapping[str, str] = {
    "BCC": "basal cell carcinoma",
    "MEL": "melanoma",
    "NEV": "melanocytic naevus",
    "SK": "seborrhoeic keratosis",
    "MISC": "other / miscellaneous",
}

#: Question types that answer "what should be done about this lesion", in priority order.
MANAGEMENT_QUESTION_TYPES: Tuple[str, ...] = (
    "initial_local_management",
    "referral_triage",
    "follow_up_surveillance",
)


class GuidelineIntegrityError(RuntimeError):
    """A catalog record failed its own content hash. Fail closed; never serve it."""


@dataclass(frozen=True)
class Recommendation:
    """One verbatim recommendation with everything needed to check it independently."""

    record_id: str
    condition: str
    question_types: Tuple[str, ...]
    text: str
    citation_id: str
    source_version: str
    authority_class: str
    content_sha256: str
    license: str
    distribution_status: str

    def verify(self) -> bool:
        """Recompute the content hash over the recommendation text."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest() == self.content_sha256


@dataclass(frozen=True)
class BindingResult:
    """What the host can say about management for a committed condition."""

    condition_group: str
    catalog_condition: Optional[str]
    covered: bool
    recommendations: Tuple[Recommendation, ...]
    absence_reason: Optional[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "condition_group": self.condition_group,
            "catalog_condition": self.catalog_condition,
            "covered": self.covered,
            "n_recommendations": len(self.recommendations),
            "recommendations": [
                {
                    "id": r.record_id,
                    "citation_id": r.citation_id,
                    "source_version": r.source_version,
                    "authority_class": r.authority_class,
                    "content_sha256": r.content_sha256,
                    "question_types": list(r.question_types),
                    "text": r.text,
                }
                for r in self.recommendations
            ],
            "absence_reason": self.absence_reason,
        }


class GuidelineCatalog:
    """Hash-verified, read-only access to the frozen recommendation corpus."""

    def __init__(self, records: Sequence[Mapping[str, object]]):
        self._by_condition: Dict[str, List[Recommendation]] = {}
        self._tampered: List[str] = []
        for raw in records:
            rec = Recommendation(
                record_id=str(raw["id"]),
                condition=str(raw["condition"]),
                question_types=tuple(str(q) for q in (raw.get("question_types") or [])),
                text=str(raw["text"]),
                citation_id=str(raw.get("citation_id", "")),
                source_version=str(raw.get("source_version", "")),
                authority_class=str(raw.get("authority_class", "")),
                content_sha256=str(raw.get("content_sha256", "")),
                license=str(raw.get("license", "")),
                distribution_status=str(raw.get("distribution_status", "")),
            )
            if not rec.verify():
                self._tampered.append(rec.record_id)
                continue
            self._by_condition.setdefault(rec.condition, []).append(rec)

        if self._tampered:
            raise GuidelineIntegrityError(
                f"{len(self._tampered)} catalog record(s) failed their content hash and were "
                f"refused: {self._tampered[:5]}"
            )

    @classmethod
    def load(cls, path: Path = CATALOG_PATH) -> "GuidelineCatalog":
        with path.open(encoding="utf-8") as handle:
            return cls([json.loads(line) for line in handle if line.strip()])

    @property
    def conditions(self) -> Tuple[str, ...]:
        return tuple(sorted(self._by_condition))

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_condition.values())

    def has_record(self, record_id: str) -> bool:
        return any(
            rec.record_id == record_id
            for recs in self._by_condition.values()
            for rec in recs
        )

    def bind(self, diagnosis_group: str, *, question_types: Sequence[str] = MANAGEMENT_QUESTION_TYPES,
             limit: int = 3) -> BindingResult:
        """Return the recommendations that apply, or an explicit statement that none do."""
        if diagnosis_group not in GROUP_TO_CONDITION:
            raise KeyError(f"unknown diagnosis group {diagnosis_group!r}")

        condition = GROUP_TO_CONDITION[diagnosis_group]
        if condition is None:
            return BindingResult(
                condition_group=diagnosis_group,
                catalog_condition=None,
                covered=False,
                recommendations=(),
                absence_reason=(
                    f"The licensed guideline corpus holds no recommendation for "
                    f"'{diagnosis_group}'. Formal dermatology guidelines concentrate on "
                    f"malignant and pre-malignant lesions; this is a coverage boundary of the "
                    f"corpus, not a retrieval failure."
                ),
            )

        available = self._by_condition.get(condition, [])
        if not available:
            return BindingResult(
                condition_group=diagnosis_group,
                catalog_condition=condition,
                covered=False,
                recommendations=(),
                absence_reason=f"No records for condition '{condition}' in the catalog.",
            )

        # Rank by the priority order of the requested question types; a recommendation matching
        # 'initial_local_management' outranks one matching only 'follow_up_surveillance'.
        def rank(rec: Recommendation) -> int:
            for i, qt in enumerate(question_types):
                if qt in rec.question_types:
                    return i
            return len(question_types)

        selected = sorted(
            (r for r in available if rank(r) < len(question_types)), key=lambda r: (rank(r), r.record_id)
        )[:limit]

        if not selected:
            return BindingResult(
                condition_group=diagnosis_group,
                catalog_condition=condition,
                covered=False,
                recommendations=(),
                absence_reason=(
                    f"Records exist for '{condition}' but none answer any of "
                    f"{list(question_types)}."
                ),
            )

        return BindingResult(
            condition_group=diagnosis_group,
            catalog_condition=condition,
            covered=True,
            recommendations=tuple(selected),
            absence_reason=None,
        )

    def coverage_report(self, groups: Sequence[str]) -> Dict[str, object]:
        """Denominators, so the coverage boundary is published rather than implied."""
        counts: Dict[str, int] = {}
        covered = 0
        for group in groups:
            counts[group] = counts.get(group, 0) + 1
            if GROUP_TO_CONDITION.get(group) is not None:
                covered += 1
        return {
            "n_cases": len(groups),
            "n_guideline_eligible": covered,
            "eligible_fraction": round(covered / len(groups), 4) if groups else 0.0,
            "by_group": counts,
            "covered_groups": sorted(g for g, c in GROUP_TO_CONDITION.items() if c is not None),
            "uncovered_groups": sorted(g for g, c in GROUP_TO_CONDITION.items() if c is None),
        }
