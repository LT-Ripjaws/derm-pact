"""Typed guideline rule records and the predicate language they use."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dermpact.guidelines.facts import FACT_SCHEMA, CaseFacts, TV, tv_and, tv_not, tv_or


class ApplicabilityState(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "applicability_unknown"
    NO_COVERAGE = "no_catalogue_coverage"


class RuleStatus(str, Enum):
    EXECUTABLE = "executable"
    PROSE_ONLY = "prose_only"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS_EXCLUDED = "ambiguous_excluded"


class CatalogueIntegrityError(RuntimeError):
    """A catalogue record failed its content hash or source binding. Never serve it."""




@dataclass(frozen=True)
class PredicateOutcome:
    predicate: str
    outcome: TV
    facts_used: Tuple[str, ...]


class Predicate:
    """A serialisable three-valued predicate over ``CaseFacts``."""

    def __init__(self, spec: Mapping[str, Any]):
        if not isinstance(spec, Mapping) or len(spec) != 1:
            raise ValueError(f"predicate must be a one-key mapping, got {spec!r}")
        (self.op, self.arg), = spec.items()
        if self.op in ("all_of", "any_of"):
            if not isinstance(self.arg, Sequence) or isinstance(self.arg, (str, bytes)):
                raise ValueError(f"{self.op} expects a list of predicates")
            self.children = tuple(Predicate(p) for p in self.arg)
        elif self.op == "not":
            self.children = (Predicate(self.arg),)
        elif self.op in ("fact_is", "fact_ge", "fact_le", "fact_in"):
            if not isinstance(self.arg, Sequence) or len(self.arg) != 2:
                raise ValueError(f"{self.op} expects [fact, value]")
            fact = str(self.arg[0])
            if fact not in FACT_SCHEMA:
                raise ValueError(f"predicate references unknown fact {fact!r}")
            self.children = ()
        else:
            raise ValueError(f"unknown predicate op {self.op!r}")

    # -- introspection --------------------------------------------------

    def facts(self) -> Tuple[str, ...]:
        if self.children:
            out: List[str] = []
            for c in self.children:
                for f in c.facts():
                    if f not in out:
                        out.append(f)
            return tuple(out)
        return (str(self.arg[0]),)

    def to_spec(self) -> Dict[str, Any]:
        if self.op in ("all_of", "any_of"):
            return {self.op: [c.to_spec() for c in self.children]}
        if self.op == "not":
            return {"not": self.children[0].to_spec()}
        return {self.op: [self.arg[0], self.arg[1]]}

    def describe(self) -> str:
        if self.op == "all_of":
            return "(" + " AND ".join(c.describe() for c in self.children) + ")"
        if self.op == "any_of":
            return "(" + " OR ".join(c.describe() for c in self.children) + ")"
        if self.op == "not":
            return "NOT " + self.children[0].describe()
        fact, value = self.arg
        sym = {"fact_is": "==", "fact_ge": ">=", "fact_le": "<=", "fact_in": "in"}[self.op]
        return f"{fact} {sym} {value!r}"

    # -- evaluation -----------------------------------------------------

    def evaluate(self, facts: CaseFacts, trace: Optional[List[PredicateOutcome]] = None) -> TV:
        if self.op == "all_of":
            result = tv_and(*[c.evaluate(facts, trace) for c in self.children])
        elif self.op == "any_of":
            result = tv_or(*[c.evaluate(facts, trace) for c in self.children])
        elif self.op == "not":
            result = tv_not(self.children[0].evaluate(facts, trace))
        else:
            result = self._evaluate_leaf(facts)
        if trace is not None and self.op not in ("all_of", "any_of", "not"):
            trace.append(PredicateOutcome(self.describe(), result, self.facts()))
        return result

    def _evaluate_leaf(self, facts: CaseFacts) -> TV:
        fact, value = self.arg
        kind = FACT_SCHEMA[fact]["kind"]
        if self.op == "fact_is":
            if kind == "tv":
                current = facts.tv(fact)
                if current is TV.UNKNOWN:
                    return TV.UNKNOWN
                return TV.TRUE if current is TV.of(value) else TV.FALSE
            current_scalar = facts.scalar(fact)
            if current_scalar is None:
                return TV.UNKNOWN
            return TV.TRUE if current_scalar == value else TV.FALSE
        if self.op in ("fact_ge", "fact_le"):
            if kind != "number":
                raise ValueError(f"{self.op} on non-numeric fact {fact!r}")
            current_num = facts.scalar(fact)
            if current_num is None:
                return TV.UNKNOWN
            ok = current_num >= value if self.op == "fact_ge" else current_num <= value
            return TV.TRUE if ok else TV.FALSE
        if self.op == "fact_in":
            current_any = facts.scalar(fact)
            if current_any is None:
                return TV.UNKNOWN
            return TV.TRUE if current_any in tuple(value) else TV.FALSE
        raise AssertionError(self.op)




def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceBinding:
    document_id: str
    doi: str
    title: str
    issuer: str
    pdf_sha256: str
    pdf_page: int
    page_text_sha256: str
    section: str
    span_text: str
    span_sha256: str
    span_char_start: int
    span_char_end: int

    def verify_span(self) -> bool:
        return sha256_text(self.span_text) == self.span_sha256

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Supersession:
    status: str  # "current" | "superseded" | "withdrawn"
    valid_from: str
    valid_until: str
    superseded_by: Optional[str] = None
    checked_on: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Action:
    code: str
    text: str
    conditional_on: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "text": self.text, "conditional_on": list(self.conditional_on)}


@dataclass(frozen=True)
class RuleRecord:
    rule_id: str
    catalogue_id: str
    catalogue_version: str
    status: RuleStatus
    scope: str
    recommendation_number: Optional[int]
    title: str
    source: SourceBinding
    authority: str
    jurisdiction: str
    guideline_version: str
    supersession: Supersession
    instrument_id: Optional[str]
    target_population: str
    prerequisites: Optional[Predicate]
    exclusions: Optional[Predicate]
    required_facts: Tuple[str, ...]
    action: Optional[Action]
    recommendation_strength: Optional[str]
    evidence_level: Optional[str]
    consensus_rate: Optional[str]
    basis: str
    encoding_rationale: str
    interpretation_points: Tuple[str, ...]
    record_sha256: str = ""

    # -- hashing ----------------------------------------------------------

    def unsigned(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "catalogue_id": self.catalogue_id,
            "catalogue_version": self.catalogue_version,
            "status": self.status.value,
            "scope": self.scope,
            "recommendation_number": self.recommendation_number,
            "title": self.title,
            "source": self.source.as_dict(),
            "authority": self.authority,
            "jurisdiction": self.jurisdiction,
            "guideline_version": self.guideline_version,
            "supersession": self.supersession.as_dict(),
            "instrument_id": self.instrument_id,
            "target_population": self.target_population,
            "prerequisites": self.prerequisites.to_spec() if self.prerequisites else None,
            "exclusions": self.exclusions.to_spec() if self.exclusions else None,
            "required_facts": list(self.required_facts),
            "action": self.action.as_dict() if self.action else None,
            "recommendation_strength": self.recommendation_strength,
            "evidence_level": self.evidence_level,
            "consensus_rate": self.consensus_rate,
            "basis": self.basis,
            "encoding_rationale": self.encoding_rationale,
            "interpretation_points": list(self.interpretation_points),
        }

    def compute_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.unsigned())).hexdigest()

    def verify(self) -> bool:
        return bool(self.record_sha256) and self.compute_sha256() == self.record_sha256 and self.source.verify_span()

    def signed(self) -> "RuleRecord":
        return RuleRecord(**{**self.__dict__, "record_sha256": self.compute_sha256()})

    def to_dict(self) -> Dict[str, Any]:
        return {**self.unsigned(), "record_sha256": self.record_sha256}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuleRecord":
        src = raw["source"]
        sup = raw["supersession"]
        act = raw.get("action")
        return cls(
            rule_id=str(raw["rule_id"]),
            catalogue_id=str(raw["catalogue_id"]),
            catalogue_version=str(raw["catalogue_version"]),
            status=RuleStatus(raw["status"]),
            scope=str(raw["scope"]),
            recommendation_number=raw.get("recommendation_number"),
            title=str(raw["title"]),
            source=SourceBinding(
                document_id=str(src["document_id"]),
                doi=str(src["doi"]),
                title=str(src["title"]),
                issuer=str(src["issuer"]),
                pdf_sha256=str(src["pdf_sha256"]),
                pdf_page=int(src["pdf_page"]),
                page_text_sha256=str(src["page_text_sha256"]),
                section=str(src["section"]),
                span_text=str(src["span_text"]),
                span_sha256=str(src["span_sha256"]),
                span_char_start=int(src["span_char_start"]),
                span_char_end=int(src["span_char_end"]),
            ),
            authority=str(raw["authority"]),
            jurisdiction=str(raw["jurisdiction"]),
            guideline_version=str(raw["guideline_version"]),
            supersession=Supersession(
                status=str(sup["status"]),
                valid_from=str(sup["valid_from"]),
                valid_until=str(sup["valid_until"]),
                superseded_by=sup.get("superseded_by"),
                checked_on=str(sup.get("checked_on", "")),
            ),
            instrument_id=raw.get("instrument_id"),
            target_population=str(raw["target_population"]),
            prerequisites=Predicate(raw["prerequisites"]) if raw.get("prerequisites") else None,
            exclusions=Predicate(raw["exclusions"]) if raw.get("exclusions") else None,
            required_facts=tuple(str(f) for f in raw.get("required_facts", [])),
            action=Action(str(act["code"]), str(act["text"]), tuple(act.get("conditional_on", []))) if act else None,
            recommendation_strength=raw.get("recommendation_strength"),
            evidence_level=raw.get("evidence_level"),
            consensus_rate=raw.get("consensus_rate"),
            basis=str(raw.get("basis", "")),
            encoding_rationale=str(raw.get("encoding_rationale", "")),
            interpretation_points=tuple(str(p) for p in raw.get("interpretation_points", [])),
            record_sha256=str(raw.get("record_sha256", "")),
        )




@dataclass
class RuleCatalogue:
    """Hash-verified, read-only rule set.  Refuses to load tampered records."""

    catalogue_id: str
    catalogue_version: str
    records: Tuple[RuleRecord, ...]
    catalogue_sha256: str = ""

    def __post_init__(self) -> None:
        bad = [r.rule_id for r in self.records if not r.verify()]
        if bad:
            raise CatalogueIntegrityError(f"records failed integrity: {bad}")
        ids = [r.rule_id for r in self.records]
        if len(ids) != len(set(ids)):
            raise CatalogueIntegrityError("duplicate rule ids")
        if not self.catalogue_sha256:
            self.catalogue_sha256 = self.compute_sha256()
        elif self.catalogue_sha256 != self.compute_sha256():
            raise CatalogueIntegrityError("catalogue hash mismatch")

    def compute_sha256(self) -> str:
        digest = hashlib.sha256()
        for r in self.records:
            digest.update(f"{r.rule_id}\t{r.record_sha256}\n".encode("utf-8"))
        return digest.hexdigest()

    @property
    def executable(self) -> Tuple[RuleRecord, ...]:
        return tuple(r for r in self.records if r.status is RuleStatus.EXECUTABLE)

    def get(self, rule_id: str) -> RuleRecord:
        for r in self.records:
            if r.rule_id == rule_id:
                return r
        raise KeyError(rule_id)

    def by_scope(self, scope: str) -> Tuple[RuleRecord, ...]:
        return tuple(r for r in self.records if r.scope == scope)

    def scopes(self) -> Tuple[str, ...]:
        return tuple(sorted({r.scope for r in self.records}))

    # -- persistence ---------------------------------------------------

    @classmethod
    def load_jsonl(cls, path: Path, *, expected_sha256: Optional[str] = None) -> "RuleCatalogue":
        path = Path(path)
        raw_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        records = tuple(RuleRecord.from_dict(json.loads(ln)) for ln in raw_lines)
        if not records:
            raise CatalogueIntegrityError("empty catalogue")
        cat_ids = {r.catalogue_id for r in records}
        cat_versions = {r.catalogue_version for r in records}
        if len(cat_ids) != 1 or len(cat_versions) != 1:
            raise CatalogueIntegrityError("mixed catalogue identity")
        catalogue = cls(cat_ids.pop(), cat_versions.pop(), records)
        if expected_sha256 is not None and catalogue.catalogue_sha256 != expected_sha256:
            raise CatalogueIntegrityError("catalogue sha256 does not match manifest")
        return catalogue

    def write_jsonl(self, path: Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False) + "\n" for r in self.records)
        path.write_text(body, encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "Action",
    "ApplicabilityState",
    "CatalogueIntegrityError",
    "Predicate",
    "PredicateOutcome",
    "RuleCatalogue",
    "RuleRecord",
    "RuleStatus",
    "SourceBinding",
    "Supersession",
    "canonical_json",
    "sha256_text",
]
