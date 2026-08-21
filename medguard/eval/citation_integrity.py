"""Citation integrity: the provenance endpoint that survives condition scoping.

Why this is not citation *precision*. Retrieval here is scoped by the image-bound
committed diagnosis, so a citation can only ever name a record of the committed
condition. Precision measured against condition is therefore 1.0 by construction
and measures nothing (see the F2 finding in
research/experiments/medguard_h2h_v1/guideline_catalog_feasibility_v1.json). It
must never be compared to MedGraphRAG's human-judged citation precision of 92.

What integrity measures instead is whether each citation an arm emits is faithful
to the frozen catalog. Three failure modes are checkable and non-trivial, because
a generator can:

1.  hallucinate an id that no record carries (``hallucinated_id``);
2.  cite a real id that was never in the candidate set offered for this lesion --
    e.g. a basal-cell record cited for a melanoma commitment (``unoffered_id``);
3.  attach a real, offered id to a claim of a type the record is not eligible for
    -- e.g. a management record cited to back a diagnosis claim
    (``claim_type_mismatch``).

A fourth guard, ``provenance_mismatch``, fires when an arm carries its own copy of
the evidence (source id / version / citation id / content hash) and any asserted
field disagrees with the frozen record -- a tampered payload under a real id. It
is checked only for fields the arm actually asserts; the base guarantee that a
resolvable id maps to a hash-verified record is provided by the catalog loader.

The denominator is citations, never cases. An arm that cites nothing has *no*
integrity score, not a perfect one -- otherwise "cite less" would look like
"cite better", the same trap that let an evidence-free arm earn a free PASS.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable


HALLUCINATED_ID = "hallucinated_id"
UNOFFERED_ID = "unoffered_id"
PROVENANCE_MISMATCH = "provenance_mismatch"
CLAIM_TYPE_MISMATCH = "claim_type_mismatch"

# Most-serious-first. A citation that fails several checks reports one primary
# reason by this order, and lists all of them in ``failed_checks``.
_FAILURE_PRECEDENCE = (
    HALLUCINATED_ID,      # nothing else is knowable if it does not resolve
    UNOFFERED_ID,         # a real id that had no business being here at all
    PROVENANCE_MISMATCH,  # a real, offered id but the payload was altered
    CLAIM_TYPE_MISMATCH,  # a real, offered id misused for the wrong claim
)

# Asserted-provenance field -> attribute name on the resolved record.
_PROVENANCE_FIELDS = ("source_id", "source_version", "citation_id", "content_sha256")


@runtime_checkable
class CitationResolver(Protocol):
    """The two primitives ``FrozenGuidelineRetriever`` already exposes."""

    def resolve(self, document_id: str) -> Optional[Any]:
        """Return the frozen record for an id, or None if the catalog lacks it."""

    def offered_ids(self, committed_diagnosis: str, claim_type: Optional[str] = None) -> set:
        """The candidate id set a citation for this commitment must come from."""


@dataclass(frozen=True)
class AssertedCitation:
    """One citation as an arm asserts it: an id bound to a claim of some type.

    ``claim_type`` is required -- in this protocol a citation always backs a typed
    claim, and appropriateness cannot be judged without it. The provenance fields
    are optional: supply them only when the arm carries its own copy of the
    evidence, in which case they are checked for exact equality against the frozen
    record.
    """

    cited_id: str
    claim_type: str
    claim_id: str = ""
    source_id: Optional[str] = None
    source_version: Optional[str] = None
    citation_id: Optional[str] = None
    content_sha256: Optional[str] = None

    def asserted_provenance(self) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for name in _PROVENANCE_FIELDS:
            value = getattr(self, name)
            if value:
                values[name] = str(value)
        return values


@dataclass(frozen=True)
class CitationCheck:
    """The verdict on a single citation, with every failing guard listed."""

    cited_id: str
    claim_type: str
    claim_id: str
    resolvable: bool
    offered: bool
    provenance_intact: bool
    claim_type_eligible: bool
    integrity_ok: bool
    failed_checks: Sequence[str]
    failure_reason: str  # "" when intact, else the primary reason by precedence


@dataclass(frozen=True)
class CaseCitationScore:
    """Per-lesion aggregate. ``all_intact`` is None when nothing was cited."""

    committed_diagnosis: str
    n_citations: int
    n_intact: int
    all_intact: Optional[bool]
    checks: Sequence[CitationCheck] = field(default_factory=tuple)


def _resolved_attr(resolved: Any, name: str) -> Optional[str]:
    if isinstance(resolved, Mapping):
        value = resolved.get(name)
    else:
        value = getattr(resolved, name, None)
    return None if value is None else str(value)


def _eligible_claim_types(resolved: Any) -> frozenset[str]:
    if isinstance(resolved, Mapping):
        values = resolved.get("eligible_claim_types", ())
    else:
        values = getattr(resolved, "eligible_claim_types", ())
    return frozenset(str(item) for item in values)


def check_citation(
    citation: AssertedCitation,
    committed_diagnosis: str,
    resolver: CitationResolver,
) -> CitationCheck:
    """Verify one citation against the frozen catalog for this commitment."""
    if not str(citation.cited_id).strip():
        raise ValueError("citation cited_id must be non-empty")
    if not str(citation.claim_type).strip():
        raise ValueError(f"citation {citation.cited_id!r} must state the claim_type it backs")

    cited_id = str(citation.cited_id)
    resolved = resolver.resolve(cited_id)
    failed: List[str] = []

    if resolved is None:
        # Unresolvable: the remaining guards are unknowable, so they are False.
        return CitationCheck(
            cited_id=cited_id, claim_type=citation.claim_type, claim_id=citation.claim_id,
            resolvable=False, offered=False, provenance_intact=False,
            claim_type_eligible=False, integrity_ok=False,
            failed_checks=(HALLUCINATED_ID,), failure_reason=HALLUCINATED_ID,
        )

    offered = cited_id in {str(item) for item in resolver.offered_ids(committed_diagnosis)}
    if not offered:
        failed.append(UNOFFERED_ID)

    provenance_intact = True
    for name, asserted in citation.asserted_provenance().items():
        if _resolved_attr(resolved, name) != asserted:
            provenance_intact = False
            break
    if not provenance_intact:
        failed.append(PROVENANCE_MISMATCH)

    claim_type_eligible = citation.claim_type in _eligible_claim_types(resolved)
    if not claim_type_eligible:
        failed.append(CLAIM_TYPE_MISMATCH)

    integrity_ok = not failed
    reason = next((r for r in _FAILURE_PRECEDENCE if r in failed), "")
    return CitationCheck(
        cited_id=cited_id, claim_type=citation.claim_type, claim_id=citation.claim_id,
        resolvable=True, offered=offered, provenance_intact=provenance_intact,
        claim_type_eligible=claim_type_eligible, integrity_ok=integrity_ok,
        failed_checks=tuple(failed), failure_reason=reason,
    )


def score_case(
    committed_diagnosis: str,
    citations: Sequence[AssertedCitation],
    resolver: CitationResolver,
) -> CaseCitationScore:
    """Score every citation a lesion emitted; no citations -> all_intact=None."""
    checks = [check_citation(c, committed_diagnosis, resolver) for c in citations]
    n = len(checks)
    intact = sum(1 for check in checks if check.integrity_ok)
    return CaseCitationScore(
        committed_diagnosis=committed_diagnosis,
        n_citations=n,
        n_intact=intact,
        all_intact=(intact == n) if n else None,
        checks=tuple(checks),
    )


def summarize(case_scores: Sequence[CaseCitationScore]) -> Dict[str, Any]:
    """Arm-level citation-integrity endpoint over a set of scored lesions.

    ``integrity_rate`` is the citation-level rate (the primary provenance number)
    and is None when the arm cited nothing across every lesion. ``cases_all_intact``
    counts lesions whose whole citation set is clean; lesions that cited nothing are
    reported separately and excluded from both denominators.
    """
    total = sum(score.n_citations for score in case_scores)
    intact = sum(score.n_intact for score in case_scores)
    with_citations = [score for score in case_scores if score.n_citations]
    all_intact = sum(1 for score in with_citations if score.all_intact)
    breakdown: Counter = Counter()
    for score in case_scores:
        for check in score.checks:
            if not check.integrity_ok:
                breakdown[check.failure_reason] += 1
    return {
        "total_citations": total,
        "intact_citations": intact,
        "integrity_rate": round(intact / total, 4) if total else None,
        "cases_scored": len(case_scores),
        "cases_with_citations": len(with_citations),
        "cases_without_citations": len(case_scores) - len(with_citations),
        "cases_all_intact": all_intact,
        "case_all_intact_rate": (
            round(all_intact / len(with_citations), 4) if with_citations else None
        ),
        "failure_breakdown": dict(breakdown),
    }
