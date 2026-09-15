"""WP-AG: explanation claim checker."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from dermpact.core.gate_v2 import allowed_claims_v2

_NUMBER_RE = re.compile(r"-?\d+\.\d+%|-?\d+%|-?\d+\.\d+(?!\d)")

_FORBIDDEN_LITERAL: Tuple[str, ...] = (
    "verified diagnosis", "caused melanoma", "this feature caused",
    "removing this feature would reduce risk", "proves the diagnosis",
    "familiar image", "checklist applies",
)

_ADVICE_RE = re.compile(
    r"\b(recommend|suggest|advise|start|begin|should\s+(?:start|begin|receive|undergo))\w*\b"
    r"[^.]{0,40}\b(treatment|excision margin|margin|staging|prognosis|survival)\b",
    re.IGNORECASE,
)

_ACTION_CLAIM_RE = re.compile(
    r"\b(i have (committed|booked|scheduled|ordered)|the review is done|i (asked|booked|scheduled)|"
    r"has been committed|follow-up (has been )?booked)\b",
    re.IGNORECASE,
)

_CASE_COMPARISON_OVERCLAIM_RE = re.compile(
    r"\b(this|that|the)\s+(retrieved\s+|historical\s+|comparison\s+|similar\s+)*case\b"
    r"[^.]{0,60}\b(proves?|confirms?|supports?)\b"
    r"|\b(this|that|the)\s+comparison\b[^.]{0,60}\b(proves?|confirms?|supports?)\b",
    re.IGNORECASE,
)

_CASE_COMPARISON_VOTE_RE = re.compile(
    r"\bi vote\b"
    r"|\bmajority verdict\b(?=[^.?!]{0,100}\b(?:is|was|would be|favors?|indicates?|therefore)\b)"
    r"|\bconsensus\b(?=[^.?!]{0,60}\b(?:is|was|says?|indicates?|suggests?|favors?)\b)"
    r"|\baverag(?:e|ed) probability\b(?=[^.?!]{0,60}\b(?:is|was|of)\b)"
    r"|\bcurrent (?:case )?diagnosis is\b",
    re.IGNORECASE,
)

_CASE_COMPARISON_CONTEXT_RE = re.compile(
    r"\b(retrieved|similar|historical|comparison)\s+(?:case|cases|label|labels|outcome|outcomes)\b",
    re.IGNORECASE,
)

_SUPPORTING_ACTION_TOOLS = frozenset({
    "propose_fact_amendment", "propose_criterion_amendment", "propose_review", "propose_reacquisition",
})

_RULE_ID_RE = re.compile(r"eado-mel-p1-2024:R\d+")


@dataclass(frozen=True)
class CheckerRejection:
    code: str
    span: str
    repair_hint: str


@dataclass(frozen=True)
class CheckerVerdict:
    ok: bool
    rejections: Tuple[CheckerRejection, ...]
    matched_claims: Tuple[str, ...]


def _tool_result_ok(tr: Any) -> bool:
    return bool(tr.ok if hasattr(tr, "ok") else tr.get("ok"))


def _tool_result_tool(tr: Any) -> str:
    return str(tr.tool if hasattr(tr, "tool") else tr.get("tool", ""))


def _tool_result_propositions(tr: Any) -> Sequence[Mapping[str, Any]]:
    return tr.propositions if hasattr(tr, "propositions") else tr.get("propositions", ())


def _proposition_index(tool_results: Sequence[Any]) -> Dict[str, Mapping[str, Any]]:
    index: Dict[str, Mapping[str, Any]] = {}
    for tr in tool_results:
        for p in _tool_result_propositions(tr):
            index[str(p["id"])] = p
    return index


def _percent_strings(value: float) -> Tuple[str, ...]:
    pct = value * 100.0
    return (f"{pct:.0f}", f"{pct:.1f}")


def _matches_reported_decimal(
    source_value: float, numeric_text: str, parsed_value: float, *, minimum_decimals: int = 3,
) -> bool:
    decimals = len(numeric_text.partition(".")[2]) if "." in numeric_text else 0
    precision = decimals if decimals >= minimum_decimals else 6
    return round(float(source_value), precision) == round(parsed_value, precision)


def _known_rule_ids(record: Any) -> frozenset:
    rows = (record.applicability or {}).get("results", []) if getattr(record, "applicability", None) else []
    return frozenset(row["rule_id"] for row in rows)


def check_explanation(
    text: str, claims: Sequence[str], record: Any, tool_results: Sequence[Any], workflow_config: Any,
) -> CheckerVerdict:
    rejections: List[CheckerRejection] = []
    matched: List[str] = []

    lower = text.lower()
    for phrase in _FORBIDDEN_LITERAL:
        if phrase in lower:
            rejections.append(CheckerRejection("forbidden_phrase", phrase, f"remove or rephrase: {phrase!r} is never permitted"))

    advice_match = _ADVICE_RE.search(text)
    if advice_match:
        rejections.append(CheckerRejection(
            "out_of_scope_advice", advice_match.group(0),
            "this system gives no treatment/margin/staging/prognosis/survival advice; "
            "state only the catalogue's scope note and the in-scope diagnostic next step",
        ))

    action_match = _ACTION_CLAIM_RE.search(text)
    if action_match:
        supported = any(_tool_result_ok(tr) and _tool_result_tool(tr) in _SUPPORTING_ACTION_TOOLS for tr in tool_results)
        if not supported:
            rejections.append(CheckerRejection(
                "unsupported_action_claim", action_match.group(0),
                "no tool receipt supports this action; report only what a propose_* tool actually returned",
            ))

    overclaim_match = _CASE_COMPARISON_OVERCLAIM_RE.search(text)
    if overclaim_match:
        rejections.append(CheckerRejection(
            "case_comparison_overclaim", overclaim_match.group(0),
            "a retrieved historical/comparison case is a non-diagnostic reminder; it never proves, "
            "confirms or supports this case's diagnosis -- report only the matched/different fields "
            "and the historical outcome's evidence grade",
        ))

    comparison_vote_match = _CASE_COMPARISON_VOTE_RE.search(text)
    has_comparison_result = any(
        _tool_result_ok(tr) and _tool_result_tool(tr) == "retrieve_case_comparisons"
        for tr in tool_results
    )
    has_comparison_context = has_comparison_result or bool(_CASE_COMPARISON_CONTEXT_RE.search(text))
    if comparison_vote_match and has_comparison_context:
        rejections.append(CheckerRejection(
            "comparison_vote_overclaim", comparison_vote_match.group(0),
            "retrieved cases are qualitative comparisons only; do not vote, form a consensus, "
            "average probabilities, or assign the current diagnosis from their labels",
        ))

    allowed = allowed_claims_v2(record)
    proposition_index = _proposition_index(tool_results)
    valid_claim_ids: List[str] = []
    for claim in claims:
        if not isinstance(claim, str):
            rejections.append(CheckerRejection(
                "invalid_claim_id", "<non-string claim>",
                "claims[] may contain only proposition-id strings such as 'p1'",
            ))
            continue
        valid_claim_ids.append(claim)
    claim_ids = list(dict.fromkeys(valid_claim_ids))
    known_rules = _known_rule_ids(record)

    for rule_match in _RULE_ID_RE.finditer(text):
        rid = rule_match.group(0)
        if rid not in known_rules:
            rejections.append(CheckerRejection(
                "unsupported_quantity", rid, f"{rid!r} is not one of this record's own evaluated rule ids",
            ))

    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        is_percent = token.endswith("%")
        numeric_text = token[:-1] if is_percent else token
        try:
            value = float(numeric_text)
        except ValueError:
            continue

        found = False
        if is_percent:
            for path, v in allowed.claims:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if numeric_text in _percent_strings(float(v)):
                        found = True
                        break
            if not found:
                for (path, v) in allowed.derived_claims:
                    if path.endswith("_percent") and isinstance(v, (int, float)) and abs(float(v) - value) < 1e-6:
                        found = True
                        break
        else:
            for path, v in allowed.claims:
                if isinstance(v, (int, float)) and not isinstance(v, bool) and _matches_reported_decimal(float(v), numeric_text, value):
                    found = True
                    break

        if not found:
            for cid in claim_ids:
                prop = proposition_index.get(cid)
                if prop is not None:
                    prop_value = prop.get("value")
                    if isinstance(prop_value, (int, float)) and not isinstance(prop_value, bool) and _matches_reported_decimal(
                        float(prop_value), numeric_text, value, minimum_decimals=2
                    ):
                        found = True
                        matched.append(cid)
                        break
                    if str(prop_value) == numeric_text:
                        found = True
                        matched.append(cid)
                        break

        if not found:
            rejections.append(CheckerRejection(
                "unsupported_quantity", token,
                f"replace {token} with a value from the record (e.g. melanoma_head.probability), "
                "or cite a supporting proposition id in claims[]",
            ))

    for cid in claim_ids:
        if cid in proposition_index:
            if cid not in matched:
                matched.append(cid)
        else:
            rejections.append(CheckerRejection("unknown_proposition", cid, f"{cid!r} does not match any tool result proposition from this turn"))

    return CheckerVerdict(ok=not rejections, rejections=tuple(rejections), matched_claims=tuple(dict.fromkeys(matched)))


__all__ = ["CheckerRejection", "CheckerVerdict", "check_explanation"]
