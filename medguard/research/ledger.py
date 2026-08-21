"""Machine-checkable competitive evidence ledger.

Numbers are not authoritative merely because they appear in a note.  A headline
metric is admissible only when its evidence points to full text or an official
artifact and records the exact evaluation protocol.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


class EvidenceLevel(str, Enum):
    VERIFIED = "verified"
    INFERRED = "inferred"
    GUESS = "guess"


COMPARISON_TYPES = {"official-code-h2h", "style-reimplementation", "reference-only", "pending"}


@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    locator: str
    detail: str
    sha256: str = ""


@dataclass
class RivalRecord:
    rival_id: str
    name: str
    architecture: str
    training_objective: str
    backbones: List[str]
    datasets: List[str]
    retrieval_corpus: str
    compute_budget: str
    code_url: str
    checkpoint_url: str
    metrics: List[str]
    evaluation_protocol: str
    strongest_reported_result: str
    known_failure_modes: List[str]
    missing_evaluations: List[str]
    medguard_component_to_beat: str
    comparison_type: str = "pending"
    evidence_level: EvidenceLevel = EvidenceLevel.GUESS
    evidence: List[EvidenceRef] = field(default_factory=list)
    last_verified: str = ""

    def validate(self, *, require_verified_h2h: bool = False) -> List[str]:
        errors: List[str] = []
        if not self.rival_id or not self.name:
            errors.append("rival_id and name are required")
        if self.comparison_type not in COMPARISON_TYPES:
            errors.append(f"unknown comparison_type {self.comparison_type!r}")
        if self.comparison_type != "pending" and not self.evidence:
            errors.append("non-pending comparison requires evidence")
        if require_verified_h2h and self.comparison_type in {
                "official-code-h2h", "style-reimplementation"}:
            if self.evidence_level is not EvidenceLevel.VERIFIED:
                errors.append("H2H record is not verified")
            required = (self.architecture, self.evaluation_protocol, self.metrics,
                        self.medguard_component_to_beat)
            if any(not x for x in required):
                errors.append("H2H record lacks a load-bearing method field")
        return errors


class RivalLedger:
    def __init__(self, records: Sequence[RivalRecord], *, scan_date: str = ""):
        self.records = list(records)
        self.scan_date = scan_date

    def validate(self, *, require_verified_h2h: bool = False) -> Dict[str, List[str]]:
        ids = [r.rival_id for r in self.records]
        errors = {r.rival_id: r.validate(require_verified_h2h=require_verified_h2h)
                  for r in self.records}
        if len(ids) != len(set(ids)):
            errors.setdefault("__ledger__", []).append("duplicate rival_id")
        return {k: v for k, v in errors.items() if v}

    def require_valid(self, *, require_verified_h2h: bool = False) -> "RivalLedger":
        errors = self.validate(require_verified_h2h=require_verified_h2h)
        if errors:
            raise ValueError(f"invalid rival ledger: {errors}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        records = []
        for record in self.records:
            value = asdict(record)
            value["evidence_level"] = record.evidence_level.value
            records.append(value)
        return {"schema_version": 1, "scan_date": self.scan_date, "records": records}

    def dump(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RivalLedger":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records: List[RivalRecord] = []
        for raw in data.get("records", []):
            value = dict(raw)
            value["evidence_level"] = EvidenceLevel(value.get("evidence_level", "guess"))
            value["evidence"] = [EvidenceRef(**e) for e in value.get("evidence", [])]
            records.append(RivalRecord(**value))
        return cls(records, scan_date=str(data.get("scan_date", "")))


def full_text_evidence(path: str | Path, detail: str) -> EvidenceRef:
    target = Path(path)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return EvidenceRef("full-text", str(target), detail, digest)
