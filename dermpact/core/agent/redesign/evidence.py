"""Read-only host adapter and version-bound evidence; never uses canned answers."""
from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Protocol

from dermpact.core.agent.tools import ToolContext, ToolRegistry


TOOLS = {
    "read_reliability": "Read selective referral, recorded trigger reasons and ACTIVE advisory warnings. Default is a compact summary; optional detail='full' includes raw measurements.",
    "read_record_section": "Read predicted diagnosis, separate melanoma probability and the two highest-ranked findings. Required section='overview'; optional detail='full' includes all raw fields.",
    "retrieve_case_comparisons": "Read availability of standard matches and separately eligible contrasting examples, with provenance and limits. Optional detail='full' includes individual examples. Requires session permission.",
}


class EvidenceError(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def probability_percent(value):
    number = Decimal(str(value))
    if not number.is_finite() or not 0 <= number <= 1:
        raise EvidenceError("invalid_melanoma_probability")
    return format((100 * number).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")


def comparison_counts(comparison):
    matches, counter = comparison["matches"], comparison.get("counterexample")
    def identity(example):
        key = example.get("case_id") or example.get("episode_id")
        if not key:
            raise EvidenceError("missing_historical_example_identity")
        return key
    ids = {identity(m) for m in matches}
    counter_id = identity(counter) if counter is not None else None
    return {"standard_matches": len(matches), "contrasting_available": counter is not None,
            "contrasting_overlaps_standard": counter_id in ids if counter is not None else None,
            "distinct_examples": len(ids | ({counter_id} if counter is not None else set()))}


def at_pointer(value: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise EvidenceError("invalid_evidence_pointer")
    current = value
    try:
        for part in pointer[1:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, list):
                if not key.isdecimal():
                    raise ValueError()
                current = current[int(key)]
            else:
                current = current[key]
    except (KeyError, TypeError, IndexError, ValueError):
        raise EvidenceError("unknown_evidence_pointer") from None
    return current


def _without_answers(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_answers(v) for k, v in value.items()
                if k not in {"grounded_summary_text", "propositions"}}
    if isinstance(value, (list, tuple)):
        return [_without_answers(v) for v in value]
    return value


class EvidenceSource(Protocol):
    def identity(self) -> dict[str, Any]: ...
    def tools(self) -> dict[str, str]: ...
    def read(self, tool: str, args: dict[str, Any]) -> dict[str, Any]: ...


class HostEvidenceSource:
    """Reuses the real record and tool registry without granting mutation tools."""
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.registry = ToolRegistry(ctx)

    def identity(self) -> dict[str, Any]:
        record = self.ctx.store.load_case(self.ctx.case_id).current_record
        return {"case_id": self.ctx.case_id, "record_version": record.identity.record_version,
                "record_sha256": record.record_sha256, "source_label": record.identity.source_label}

    def tools(self) -> dict[str, str]:
        return {k: v for k, v in TOOLS.items()
                if k != "retrieve_case_comparisons" or self.ctx.memory_request_explicit}

    def read(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool not in self.tools():
            raise EvidenceError("tool_not_permitted")
        identity = self.identity()
        for key in ("case_id", "record_version"):
            if key in args and args[key] != identity[key]:
                raise EvidenceError("wrong_case_or_version_requested")
        permitted = {"case_id", "record_version", "detail", "section"} if tool == "read_record_section" else {"case_id", "record_version", "detail"}
        if set(args) - permitted:
            raise EvidenceError("unexpected_tool_argument")
        if args.get("detail", "summary") not in {"summary", "full"}:
            raise EvidenceError("invalid_evidence_detail")
        if tool == "read_record_section" and args.get("section", "overview") != "overview":
            raise EvidenceError("unsupported_record_section")
        bound = {"case_id": identity["case_id"], "record_version": identity["record_version"]}
        if tool == "read_record_section":
            bound["section"] = "overview"
        result = self.registry.call(tool, bound)
        if self.identity() != identity:
            raise EvidenceError("record_changed_during_read")
        if result.bindings.get("case_id") != identity["case_id"] or result.bindings.get("record_version") != identity["record_version"]:
            raise EvidenceError("receipt_binding_mismatch")
        return {**identity, "tool": tool, "ok": result.ok, "detail": args.get("detail", "summary"),
                "data": _without_answers(copy.deepcopy(result.data)), "error": result.error}


def interpreted_view(tool: str, data: dict) -> tuple[dict, dict] | None:
    """Source-only projections, never given a question, reference answer or draft."""
    if not isinstance(data, dict):
        return None
    if tool == "read_record_section":
        if "perception" in data:
            p = data["perception"]
            diagnosis, melanoma = p.get("diagnosis"), p.get("melanoma_probability")
            criteria = p.get("criterion_positive_probabilities")
            roots = ["/data/perception/diagnosis", "/data/perception/melanoma_probability",
                     "/data/perception/criterion_positive_probabilities"]
        elif "heads" in data:
            p = data["heads"]
            diagnosis = p.get("diagnosis_head", {}).get("provisional_class")
            melanoma = p.get("melanoma_head", {}).get("probability")
            criteria = p.get("criterion_probabilities")
            roots = ["/data/heads/diagnosis_head/provisional_class", "/data/heads/melanoma_head/probability",
                     "/data/heads/criterion_probabilities"]
        else:
            return None
        if not isinstance(criteria, dict) or len(criteria) < 2:
            return None
        if any(type(v) not in (int, float) or not 0 <= v <= 1 for v in criteria.values()):
            raise EvidenceError("invalid_criterion_probability")
        ranked = sorted(criteria, key=lambda k: (-criteria[k], k))[:2]
        return ({"predicted_diagnosis": diagnosis.replace("_", " ") if isinstance(diagnosis, str) else None,
                 "separate_melanoma_probability": melanoma,
                 "separate_melanoma_percentage": probability_percent(melanoma) if melanoma is not None else None,
                 "two_strongest_predicted_findings": [k.replace("_", " ") for k in ranked],
                 "evidence_kind": "model predictions, not confirmed clinical observations"},
                {"source_fields": roots, "operations": "label formatting; descending probability rank, lexical tie-break, top two; fraction unchanged; percentage = fraction * 100, two decimals, ROUND_HALF_UP"})
    if tool == "read_reliability":
        disposition, ood = data.get("disposition"), data.get("ood_state")
        warnings = data.get("warnings")
        return ({"selectively_referred_for_review": {"review": True, "display": False}.get(disposition),
                 "recorded_referral_trigger_reasons": [w["reason"] for w in warnings] if isinstance(warnings, list) else None,
                 "active_advisory_warnings": ["unfamiliarity"] if ood == "flagged" else [] if ood == "not_flagged" else None,
                 "advisory_role_when_active": "informational only",
                 "advisory_warnings_can_trigger_referral": False},
                {"source_fields": ["/data/disposition", "/data/warnings", "/data/ood_state"],
                 "operations": "recorded disposition decode; copy reasons; advisory activation from ood_state only; fixed advisory-only contract"})
    if tool == "retrieve_case_comparisons":
        if data.get("status") not in {"ok", "no_match"}:
            return ({"availability": "unknown", "retrieval_status": data.get("status"), "reason": data.get("reason")},
                    {"source_fields": ["/data/status", "/data/reason"], "operations": "unavailable retrieval is not zero matches"})
        matches, contrasting = data.get("matches"), data.get("counterexample")
        if not isinstance(matches, list):
            return None
        if data.get("counterexample_status") == "available" and not isinstance(contrasting, dict):
            raise EvidenceError("inconsistent_counterexample_receipt")
        examples = matches + ([contrasting] if isinstance(contrasting, dict) else [])
        counts = comparison_counts(data)
        provenance = {(e.get("outcome", {}).get("evidence_grade") or "unknown",
                       e.get("outcome", {}).get("source") or "unknown") for e in examples}
        return ({"standard_similarity_match_count": len(matches),
                 "separately_eligible_contrasting_example_available": isinstance(contrasting, dict),
                 "any_eligible_historical_example_available": bool(examples),
                 "contrasting_example_relation_to_standard_matches": (
                     "not_applicable" if not matches or not isinstance(contrasting, dict) else
                     "overlap" if counts["contrasting_overlaps_standard"] else "distinct"),
                 "distinct_historical_example_count": counts["distinct_examples"],
                 "label_provenance": [{"grade": g, "source": s} for g, s in sorted(provenance)],
                 "historical_label_scope": "Labels describe historical cases, not independent confirmation of this lesion's diagnosis." if examples else None,
                 "can_change_stored_referral_decision": False},
                {"source_fields": ["/data/status", "/data/matches", "/data/counterexample", "/data/counterexample_status"],
                 "operations": "count standard matches; preserve separately thresholded counterexample eligibility; compare case identity (episode identity fallback) for overlap and distinct union; collect original label provenance; read-only contract"})
    return None


def bind_receipt(raw: dict[str, Any], identity: dict[str, Any], index: int) -> dict[str, Any]:
    for key in ("case_id", "record_version", "record_sha256"):
        if raw.get(key) != identity[key]:
            raise EvidenceError("receipt_binding_mismatch")
    receipt = {**copy.deepcopy(raw), "id": f"e{index}"}
    receipt["policy"] = {
        "record_mutation_allowed": False, "diagnosis_mutation_allowed": False,
        "referral_mutation_allowed": False,
        "interpretation": "Read-only evidence; may inform clinician review but cannot change the stored diagnosis, probabilities, or referral decision.",
    }
    interpreted = interpreted_view(raw["tool"], raw.get("data")) if raw.get("ok") else None
    if interpreted is not None:
        receipt["view"], receipt["view_provenance"] = interpreted
        receipt["view_version"] = "dermpact-source-view-v4-advisory-role"
    receipt["sha256"] = digest(receipt)
    return receipt


def fact_catalog(receipts: list[dict]) -> dict[str, dict]:
    """Stable local identifiers for exact leaf fields, not prewritten claims."""
    result = {}
    def leaves(value, pointer):
        if isinstance(value, (dict, list)) and not value:
            yield pointer, value
        if isinstance(value, dict):
            for key in sorted(value):
                yield from leaves(value[key], pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from leaves(item, pointer + "/" + str(index))
        else:
            yield pointer, value
    for receipt in receipts:
        keys = ["error", "policy", "ok"]
        if "view" in receipt:
            keys.append("view")
        if "view" not in receipt or receipt.get("detail") == "full":
            keys.append("data")
        fields = list(leaves({k: receipt.get(k) for k in keys}, ""))
        for index, (pointer, value) in enumerate(fields, 1):
            result[f"{receipt['id']}.f{index}"] = {
                "source_id": receipt["id"], "pointer": pointer, "value": value,
            }
    return result


def wire_receipt(receipt: dict) -> dict:
    return {k: receipt[k] for k in ("id", "case_id", "record_version", "record_sha256", "tool", "ok")} | {
        "schema_semantics": "The view contains deterministic source facts, not a written answer. Empty arrays mean none; null means unknown. Probability fields are fractions; the percentage field is already converted and rounded to two decimals. Ranked findings come from descending model probabilities. Active advisories are separate from their policy. Standard matches and contrasting examples have different eligibility rules and can overlap. Their relation is not_applicable when either category is absent, otherwise overlap or distinct. These reads cannot change the recorded decision. Full raw detail can be requested when needed.",
        "facts": fact_catalog([receipt])}
