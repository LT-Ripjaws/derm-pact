"""WP-AG: scope-checked typed tools."""
from __future__ import annotations

import dataclasses
import json
import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from dermpact.core import versions as V
from dermpact.core.clarification import decision_relevant_questions
from dermpact.core.episode import derive_state, mark_awaiting_information, request_review, what_remains
from dermpact.core.ledger_v2 import AmendmentLedgerV2, AmendmentProposal
from dermpact.core.memory import (
    ComparisonPolicy, MemoryBank, DATA_ROLE_DEVELOPMENT, DATA_ROLE_DEPLOYMENT_QUERY,
    DATA_ROLE_EVALUATION_QUERY, compare as memory_compare, refresh as memory_refresh, reminder_allowed,
)
from dermpact.core.record_v2 import CLINICIAN_ASSERTION_FACTS, StructuredCaseRecordV2
from dermpact.core.store import CaseStore
from dermpact.core.xai import ReceiptMismatch, receipt_provenance, verify_receipt
from dermpact.pipeline.guideline_retriever import FrozenGuidelineRetriever

XAI_KINDS = frozenset({"region", "clinical"})
LIBRARY_CLAIM_TYPES = frozenset({"assessment", "diagnosis", "visual_finding", "differential"})
_GLOBAL_TO_LIBRARY_DIAGNOSIS = {
    "melanoma": "melanoma",
    "basal_cell_carcinoma": "bcc",
    "actinic_keratosis_or_intraepithelial_carcinoma": "akiec",
}

def _overview_section(d: Dict[str, Any]) -> Dict[str, Any]:
    reliability = d["reliability"]
    return {
        "heads": {"diagnosis_head": d["perception"]["diagnosis_head"],
                  "melanoma_head": d["perception"]["melanoma_head"],
                  "criterion_probabilities": d["perception"]["criterion_probabilities"]},
        "reliability": {key: reliability.get(key) for key in (
            "disposition", "iqa_state", "ood_state", "view_stability_state", "selective_abstention", "warnings"
        )},
        "image_status": d["image_status"],
        "checklist": d["checklist"], "agreement": d["agreement"],
        "next_step": d["next_step"],
    }


_RECORD_SECTIONS: Mapping[str, Callable[[Dict[str, Any]], Any]] = {
    "overview": _overview_section,
    "identity": lambda d: d["identity"],
    "inputs": lambda d: d["inputs"],
    "perception": lambda d: d["perception"],
    "heads": lambda d: {
        "diagnosis_head": d["perception"]["diagnosis_head"],
        "melanoma_head": d["perception"]["melanoma_head"],
        "criterion_probabilities": d["perception"]["criterion_probabilities"],
    },
    "reliability": lambda d: d["reliability"],
    "image_status": lambda d: d["image_status"],
    "checklist": lambda d: d["checklist"],
    "agreement": lambda d: d["agreement"],
    "clinician_assertions": lambda d: d["clinician_assertions"],
    "facts": lambda d: d["facts"],
    "applicability": lambda d: d["applicability"],
    "next_step": lambda d: d["next_step"],
    "versions": lambda d: d["versions"],
    "amendments": lambda d: d["amendments"],
}




@dataclass(frozen=True)
class ToolContext:
    case_id: str
    store: CaseStore
    ledger: AmendmentLedgerV2
    engine: Any
    catalogue: Any
    guideline_retriever: Optional[FrozenGuidelineRetriever]
    bundle_calibrators: Optional[Mapping[str, Any]]
    memory_bank: Optional[MemoryBank]
    memory_policy: Optional[ComparisonPolicy]
    memory_request_explicit: bool
    inference_client: Any
    xai_job_api: Optional[Callable[..., Mapping[str, Any]]]
    session_id: str
    actor: str
    mode: str  # "offline_deterministic" | "live_provider"


@dataclass(frozen=True)
class ToolError:
    code: str
    detail: str
    repairable: bool = False


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    args: Dict[str, Any]
    bindings: Dict[str, Any]
    data: Any = None
    propositions: Tuple[Dict[str, Any], ...] = ()
    error: Optional[Dict[str, Any]] = None
    duplicate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "tool": self.tool, "args": self.args, "bindings": self.bindings,
            "data": self.data, "propositions": list(self.propositions), "error": self.error,
            "duplicate": self.duplicate,
        }




@dataclass(frozen=True)
class ReadRecordSectionArgs:
    case_id: str
    record_version: int
    section: str


@dataclass(frozen=True)
class ReadReliabilityArgs:
    case_id: str
    record_version: int


@dataclass(frozen=True)
class FindDecisiveMissingFactsArgs:
    case_id: str
    record_version: int


@dataclass(frozen=True)
class ProposeFactAmendmentArgs:
    case_id: str
    record_version: int
    fact: str
    value: Any
    statement: str = ""


@dataclass(frozen=True)
class ProposeCriterionAmendmentArgs:
    case_id: str
    record_version: int
    criterion: str
    value: str
    statement: str = ""


@dataclass(frozen=True)
class PreviewConsequencesArgs:
    case_id: str
    proposal_id: str


@dataclass(frozen=True)
class RefreshApplicabilityArgs:
    case_id: str
    last_known_record_version: int


@dataclass(frozen=True)
class RetrieveGuidelineEvidenceArgs:
    case_id: str
    record_version: int
    rule_id: str


@dataclass(frozen=True)
class SearchGuidelineLibraryArgs:
    case_id: str
    record_version: int
    claim_type: str = "assessment"
    limit: int = 5


@dataclass(frozen=True)
class RetrieveCaseComparisonsArgs:
    case_id: str
    record_version: int


@dataclass(frozen=True)
class RequestMeasuredXaiArgs:
    case_id: str
    record_version: int
    kind: str


@dataclass(frozen=True)
class ReadMeasuredXaiArgs:
    case_id: str
    record_version: int
    kind: str


@dataclass(frozen=True)
class ProposeReviewArgs:
    case_id: str
    record_version: int
    reason: str


@dataclass(frozen=True)
class ProposeReacquisitionArgs:
    case_id: str
    record_version: int
    reason: str


@dataclass(frozen=True)
class ReadReviewHistoryArgs:
    case_id: str
    record_version: int


@dataclass(frozen=True)
class SubmitExplanationArgs:
    case_id: str
    record_version: int
    text: str
    claims: Tuple[str, ...] = ()


_TOOL_SPECS: Dict[str, type] = {
    "read_record_section": ReadRecordSectionArgs,
    "read_reliability": ReadReliabilityArgs,
    "find_decisive_missing_facts": FindDecisiveMissingFactsArgs,
    "propose_fact_amendment": ProposeFactAmendmentArgs,
    "propose_criterion_amendment": ProposeCriterionAmendmentArgs,
    "preview_consequences": PreviewConsequencesArgs,
    "refresh_applicability": RefreshApplicabilityArgs,
    "retrieve_guideline_evidence": RetrieveGuidelineEvidenceArgs,
    "search_guideline_library": SearchGuidelineLibraryArgs,
    "retrieve_case_comparisons": RetrieveCaseComparisonsArgs,
    "request_measured_xai": RequestMeasuredXaiArgs,
    "read_measured_xai": ReadMeasuredXaiArgs,
    "propose_review": ProposeReviewArgs,
    "propose_reacquisition": ProposeReacquisitionArgs,
    "read_review_history": ReadReviewHistoryArgs,
    "submit_explanation": SubmitExplanationArgs,
}




def _bindings(ctx: ToolContext, record: Optional[StructuredCaseRecordV2]) -> Dict[str, Any]:
    if record is None:
        return {"case_id": ctx.case_id}
    return {
        "case_id": ctx.case_id, "record_sha256": record.record_sha256,
        "record_version": record.identity.record_version, "model_id": record.perception.model_id,
        "catalogue_sha256": ctx.catalogue.catalogue_sha256,
        "memory_bank_sha256": (ctx.memory_bank.bank_sha256 if ctx.memory_bank is not None else None),
        "xai_protocol": V.XAI_PROTOCOL_VERSION, "workflow_config": V.WORKFLOW_CONFIG_VERSION,
    }


def _walk_leaves(prefix: str, value: Any, out: List[Tuple[str, Any]], *, max_leaves: int = 400) -> None:
    if len(out) >= max_leaves:
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            _walk_leaves(f"{prefix}.{k}" if prefix else str(k), v, out, max_leaves=max_leaves)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _walk_leaves(f"{prefix}[{i}]", v, out, max_leaves=max_leaves)
    else:
        out.append((prefix, value))


def _canon(args: Mapping[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, default=str)


def _default_clock() -> str:
    return datetime.now(timezone.utc).isoformat()


def _receipt_kind(receipt: Mapping[str, Any]) -> Optional[str]:
    if "kind" in receipt:
        return str(receipt["kind"])
    if "results" in receipt and "heatmap" in receipt:
        return "region"
    if "targets" in receipt:
        return "clinical"
    return None


def _memory_grounded_summary(payload: Mapping[str, Any]) -> str:
    matches = list(payload.get("matches") or ())
    counts = (payload.get("descriptive_reference_label_counts") or {}).get("counts") or {}
    if not matches:
        policy = (payload.get("receipt") or {}).get("policy") or {}
        return (
            "No eligible historical comparison case was found; none met the frozen minimum similarity "
            f"threshold {policy.get('min_similarity')}. No historical label is used to alter "
            "the current prediction."
        )
    ordered_counts = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    rendered_counts = [
        f"{count} {'has' if int(count) == 1 else 'have'} dataset label {label}"
        for label, count in ordered_counts
    ]
    count_text = rendered_counts[0] if len(rendered_counts) == 1 else ", ".join(rendered_counts[:-1]) + " and " + rendered_counts[-1]
    match_text = "; ".join(
        f"{match.get('case_id')} similarity {match.get('similarity')}"
        for match in matches
    )
    counterexample = payload.get("counterexample")
    counter_text = (
        f" A discordant reference case is {counterexample.get('case_id')} with dataset label "
        f"{(counterexample.get('outcome') or {}).get('diagnosis')} and similarity {counterexample.get('similarity')}."
        if counterexample else " No discordant counterexample cleared the frozen threshold."
    )
    return (
        f"Historical comparison: Among the {len(matches)} retrieved reference cases, {count_text}. "
        f"Retrieved similarities: {match_text}.{counter_text} "
        "This is a descriptive historical comparison and is not used to alter the current prediction."
    )


def _xai_grounded_summary(kind: str, receipt: Mapping[str, Any], record: StructuredCaseRecordV2) -> str:
    if kind == "region":
        raw_results = receipt.get("results") or ()
        if isinstance(raw_results, Mapping):
            results = [
                {"region_index": key, **(value if isinstance(value, Mapping) else {})}
                for key, value in raw_results.items()
            ]
        else:
            results = list(raw_results)
        if not results:
            return "No measured region-sensitivity result is available in the verified receipt."
        rendered = "; ".join(
            f"region {row.get('region_index')} local-mean melanoma-head absolute delta "
            f"{row.get('rank_key_local_mean_melanoma_head_abs_delta')}"
            for row in results
        )
        return (
            f"Measured region sensitivity: {rendered}. These are perturbation sensitivity "
            "measurements, not clinical causality; masked inputs may be out of distribution."
        )
    availability = record.inputs.clinical_availability
    if receipt.get("status") == "not_applicable":
        return (
            f"Clinical-photo sensitivity is not applicable because clinical_availability={availability}. "
            f"Members measured: {receipt.get('members_measured', 0)}. No clinical-photo sensitivity "
            "or contribution was computed."
        )
    targets = receipt.get("targets") or {}
    deltas = [float(row.get("aggregated_delta", 0.0)) for row in targets.values()]
    if deltas and all(value == 0.0 for value in deltas):
        return (
            f"Measured clinical-photo sensitivity is available and completed across {receipt.get('members_measured')} members. "
            "All reported target intervention deltas were exactly 0.0. This is a measured model "
            "sensitivity result, not clinical causality."
        )
    selected = []
    for target in ("diagnosis_head_melanoma", "melanoma_head"):
        if target in targets:
            selected.append(f"{target} aggregated delta {targets[target].get('aggregated_delta')}")
    return (
        "Measured clinical-photo sensitivity: " + "; ".join(selected) + ". "
        "This is a measured model sensitivity result, not clinical causality."
    )


def _xai_grounded_fields(kind: str, receipt: Mapping[str, Any], record: StructuredCaseRecordV2) -> Dict[str, Any]:
    if kind == "region":
        raw_results = receipt.get("results") or ()
        if isinstance(raw_results, Mapping):
            results = [{"region_index": key, **(value if isinstance(value, Mapping) else {})}
                       for key, value in raw_results.items()]
        else:
            results = list(raw_results)
        return {
            "kind": "region", "method": "local_mean", "target": "melanoma_head",
            "measure": "absolute perturbation delta", "regions": results,
            "interpretation": "model sensitivity measurement; not clinical causality",
            "ood_qualification": receipt.get("ood_qualification"),
        }
    targets = receipt.get("targets") or {}
    deltas = [float(row.get("aggregated_delta", 0.0)) for row in targets.values()]
    return {
        "kind": "clinical", "status": receipt.get("status"),
        "clinical_availability": record.inputs.clinical_availability,
        "members_measured": receipt.get("members_measured", 0),
        "all_reported_target_deltas_exact_zero": bool(deltas and all(value == 0.0 for value in deltas)),
        "target_aggregated_deltas": {name: row.get("aggregated_delta") for name, row in targets.items()},
        "reason": receipt.get("reason"),
        "interpretation": ("no clinical-photo sensitivity or contribution was computed"
                           if receipt.get("status") == "not_applicable" else
                           "measured model sensitivity result; not clinical causality"),
    }




class ToolRegistry:
    """Host-owned tool dispatcher for one controller run."""

    def __init__(self, ctx: ToolContext, *, clock: Callable[[], str] = _default_clock):
        self.ctx = ctx
        self._clock = clock
        self._cache: Dict[Tuple[str, str, str], ToolResult] = {}
        self._proposal_owners: Dict[str, Tuple[str, str]] = {}
        self._proposal_previews: Dict[str, Dict[str, Any]] = {}
        self._proposition_counter = itertools.count(1)
        self._episode_id = f"episode-{ctx.case_id}"
        self.history: List[ToolResult] = []

    # -- dispatch ---------------------------------------------------------

    def call(self, tool: str, args: Mapping[str, Any]) -> ToolResult:
        spec = _TOOL_SPECS.get(tool)
        if spec is None:
            result = ToolResult(ok=False, tool=str(tool), args=dict(args), bindings={"case_id": self.ctx.case_id},
                                error={"code": "unknown_tool", "detail": f"{tool!r} is not a registered tool", "repairable": True})
            self.history.append(result)
            return result
        try:
            typed = spec(**{k: (tuple(v) if spec is SubmitExplanationArgs and k == "claims" else v) for k, v in args.items()})
        except TypeError as exc:
            result = ToolResult(ok=False, tool=str(tool), args=dict(args), bindings={"case_id": self.ctx.case_id},
                                error={"code": "invalid_args", "detail": str(exc), "repairable": True})
            self.history.append(result)
            return result
        method = getattr(self, tool)
        return method(typed)

    # -- shared preamble ----------------------------------------------------

    def _next_pid(self) -> str:
        return f"p{next(self._proposition_counter)}"

    def _propositions_from(self, payload: Any, *, ref_prefix: str) -> Tuple[Dict[str, Any], ...]:
        leaves: List[Tuple[str, Any]] = []
        _walk_leaves("", payload, leaves)
        return tuple(
            {"id": self._next_pid(), "path_or_ref": f"{ref_prefix}.{path}" if path else ref_prefix,
             "value": value, "text": f"{ref_prefix}.{path}={value!r}" if path else f"{ref_prefix}={value!r}"}
            for path, value in leaves
        )

    def _cache_key(self, tool: str, args: Mapping[str, Any], record: Optional[StructuredCaseRecordV2]) -> Tuple[str, str, str]:
        return (tool, _canon(args), record.record_sha256 if record is not None else "")

    def _finish(
        self, tool: str, args: Mapping[str, Any], record: Optional[StructuredCaseRecordV2], *, ok: bool,
        data: Any = None, propositions: Sequence[Dict[str, Any]] = (), error: Optional[ToolError] = None,
        cache: bool = True,
    ) -> ToolResult:
        result = ToolResult(
            ok=ok, tool=tool, args=dict(args), bindings=_bindings(self.ctx, record), data=data,
            propositions=tuple(propositions), error=(dataclasses.asdict(error) if error is not None else None),
        )
        if cache:
            self._cache[self._cache_key(tool, args, record)] = result
        self.history.append(result)
        return result

    def _dispatch(self, tool: str, args: Mapping[str, Any], *, record_version_key: str = "record_version"):
        """Case + record-version scope check, then duplicate-cache lookup."""
        case_id = args.get("case_id")
        if case_id != self.ctx.case_id:
            result = self._finish(tool, args, None, ok=False, cache=False,
                                  error=ToolError("wrong_case", f"tool call is bound to case {case_id!r}, not {self.ctx.case_id!r}"))
            return result, None
        try:
            case_state = self.ctx.store.load_case(self.ctx.case_id)
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed tool error
            result = self._finish(tool, args, None, ok=False, cache=False,
                                  error=ToolError("case_not_found", str(exc)))
            return result, None
        record = case_state.current_record
        expected = args.get(record_version_key)
        if expected is not None and expected != record.identity.record_version:
            result = self._finish(tool, args, record, ok=False, cache=False,
                                  error=ToolError("stale_record_version",
                                                   f"record_version {expected} does not match current head {record.identity.record_version}"))
            return result, record
        key = self._cache_key(tool, args, record)
        cached = self._cache.get(key)
        if cached is not None:
            dup = dataclasses.replace(cached, duplicate=True)
            self.history.append(dup)
            return dup, record
        return None, record

    # -- tools ----------------------------------------------------------

    def read_record_section(self, args: ReadRecordSectionArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("read_record_section", a)
        if early is not None:
            return early
        if args.section not in _RECORD_SECTIONS:
            return self._finish("read_record_section", a, record, ok=False,
                                error=ToolError("unknown_section", f"{args.section!r} is not a known record section"))
        payload = _RECORD_SECTIONS[args.section](record.to_dict())
        propositions = self._propositions_from(payload, ref_prefix=f"record.{args.section}")
        return self._finish("read_record_section", a, record, ok=True, data=payload, propositions=propositions)

    def read_reliability(self, args: ReadReliabilityArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("read_reliability", a)
        if early is not None:
            return early
        payload = record.reliability.to_dict()
        propositions = self._propositions_from(payload, ref_prefix="record.reliability")
        return self._finish("read_reliability", a, record, ok=True, data=payload, propositions=propositions)

    def find_decisive_missing_facts(self, args: FindDecisiveMissingFactsArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("find_decisive_missing_facts", a)
        if early is not None:
            return early
        case_state = self.ctx.store.load_case(self.ctx.case_id)
        analysis = decision_relevant_questions(
            record, engine=self.ctx.engine, catalogue=self.ctx.catalogue,
            calibrators=self.ctx.bundle_calibrators, bundle=None, episode_events=case_state.episode_events,
        )
        payload = analysis.to_dict()
        propositions = tuple(
            {"id": self._next_pid(), "path_or_ref": f"clarification.questions[{i}].fact",
             "value": q["fact"], "text": q["question_text"]}
            for i, q in enumerate(payload["questions"])
        )
        return self._finish("find_decisive_missing_facts", a, record, ok=True, data=payload, propositions=propositions)

    def propose_fact_amendment(self, args: ProposeFactAmendmentArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("propose_fact_amendment", a)
        if early is not None:
            return early
        if args.fact not in CLINICIAN_ASSERTION_FACTS:
            return self._finish("propose_fact_amendment", a, record, ok=False,
                                error=ToolError("unknown_fact", f"{args.fact!r} is not one of the nine amendable clinician facts"))
        receipt = self.ctx.ledger.propose(
            record, AmendmentProposal(kind="clinician_fact", field=args.fact, value=args.value),
            actor=self.ctx.actor, session_id=self.ctx.session_id, proposed_at=self._clock(), statement=args.statement,
        )
        if receipt.status != "proposed":
            return self._finish("propose_fact_amendment", a, record, ok=False,
                                error=ToolError(receipt.reason or "rejected", receipt.detail))
        self._proposal_owners[receipt.proposal_id] = (self.ctx.case_id, self.ctx.session_id)
        self._proposal_previews[receipt.proposal_id] = dataclasses.asdict(receipt)
        propositions = (
            {"id": self._next_pid(), "path_or_ref": f"ledger.proposal[{receipt.proposal_id}].after",
             "value": receipt.after, "text": f"proposed {args.fact} = {receipt.after!r}"},
        )
        return self._finish("propose_fact_amendment", a, record, ok=True, data=dataclasses.asdict(receipt), propositions=propositions)

    def propose_criterion_amendment(self, args: ProposeCriterionAmendmentArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("propose_criterion_amendment", a)
        if early is not None:
            return early
        receipt = self.ctx.ledger.propose(
            record, AmendmentProposal(kind="criterion_value", field=args.criterion, value=args.value),
            actor=self.ctx.actor, session_id=self.ctx.session_id, proposed_at=self._clock(), statement=args.statement,
        )
        if receipt.status != "proposed":
            return self._finish("propose_criterion_amendment", a, record, ok=False,
                                error=ToolError(receipt.reason or "rejected", receipt.detail))
        self._proposal_owners[receipt.proposal_id] = (self.ctx.case_id, self.ctx.session_id)
        self._proposal_previews[receipt.proposal_id] = dataclasses.asdict(receipt)
        propositions = (
            {"id": self._next_pid(), "path_or_ref": f"ledger.proposal[{receipt.proposal_id}].after",
             "value": receipt.after, "text": f"proposed {args.criterion} = {receipt.after!r}"},
        )
        return self._finish("propose_criterion_amendment", a, record, ok=True, data=dataclasses.asdict(receipt), propositions=propositions)

    def preview_consequences(self, args: PreviewConsequencesArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        if args.case_id != self.ctx.case_id:
            return self._finish("preview_consequences", a, None, ok=False, cache=False,
                                error=ToolError("wrong_case", f"tool call is bound to case {args.case_id!r}, not {self.ctx.case_id!r}"))
        owner = self._proposal_owners.get(args.proposal_id)
        if owner is None or owner != (self.ctx.case_id, self.ctx.session_id):
            return self._finish("preview_consequences", a, None, ok=False, cache=False,
                                error=ToolError("unknown_proposal",
                                                 f"proposal {args.proposal_id!r} is not a pending proposal for this case/session"))
        cached_receipt = self._proposal_previews[args.proposal_id]
        record = self.ctx.store.load_case(self.ctx.case_id).current_record
        propositions = (
            {"id": self._next_pid(), "path_or_ref": f"ledger.proposal[{args.proposal_id}].consequence_preview_sha256",
             "value": cached_receipt["consequence_preview_sha256"], "text": "previewed consequence (before/after)"},
        )
        return self._finish("preview_consequences", a, record, ok=True, data=cached_receipt.get("consequence_preview"),
                            propositions=propositions, cache=False)

    def refresh_applicability(self, args: RefreshApplicabilityArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        if args.case_id != self.ctx.case_id:
            return self._finish("refresh_applicability", a, None, ok=False, cache=False,
                                error=ToolError("wrong_case", f"tool call is bound to case {args.case_id!r}, not {self.ctx.case_id!r}"))
        record = self.ctx.store.load_case(self.ctx.case_id).current_record
        if record.identity.record_version <= args.last_known_record_version:
            return self._finish("refresh_applicability", a, record, ok=False, cache=False,
                                error=ToolError("no_new_commit", "no commit has happened since the last read"))
        payload = {"applicability": record.applicability, "next_step": record.next_step.to_dict()}
        propositions = self._propositions_from(payload, ref_prefix="record")
        return self._finish("refresh_applicability", a, record, ok=True, data=payload, propositions=propositions, cache=False)

    def retrieve_guideline_evidence(self, args: RetrieveGuidelineEvidenceArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("retrieve_guideline_evidence", a)
        if early is not None:
            return early
        rule = next((r for r in self.ctx.catalogue.records if r.rule_id == args.rule_id), None)
        if rule is None:
            return self._finish("retrieve_guideline_evidence", a, record, ok=False,
                                error=ToolError("unknown_rule", f"{args.rule_id!r} is not in the catalogue"))
        payload = {
            "rule_id": rule.rule_id, "status": rule.status.value, "scope": rule.scope,
            "jurisdiction": rule.jurisdiction, "guideline_version": rule.guideline_version,
            "authority": rule.authority, "supersession": rule.supersession.as_dict(),
            "source": rule.source.as_dict(),
        }
        propositions = (
            {"id": self._next_pid(), "path_or_ref": f"catalogue.{rule.rule_id}.source.span_text",
             "value": rule.source.span_text, "text": rule.source.span_text},
        )
        return self._finish("retrieve_guideline_evidence", a, record, ok=True, data=payload, propositions=propositions)

    def search_guideline_library(self, args: SearchGuidelineLibraryArgs) -> ToolResult:
        """Retrieve source-exact diagnostic evidence from the 66-row library."""
        a = dataclasses.asdict(args)
        early, record = self._dispatch("search_guideline_library", a)
        if early is not None:
            return early
        if self.ctx.guideline_retriever is None:
            return self._finish("search_guideline_library", a, record, ok=False,
                                error=ToolError("guideline_library_unavailable", "the 66-record library is not configured"))
        if args.claim_type not in LIBRARY_CLAIM_TYPES:
            return self._finish(
                "search_guideline_library", a, record, ok=False,
                error=ToolError("claim_type_out_of_scope",
                                "only assessment, diagnosis, visual_finding and differential evidence is exposed; "
                                "management, treatment, staging and follow-up remain outside scope"),
            )
        if not 1 <= int(args.limit) <= 10:
            return self._finish("search_guideline_library", a, record, ok=False,
                                error=ToolError("invalid_limit", "limit must be between 1 and 10"))
        provisional = str(record.perception.diagnosis_head.get("provisional_class") or "")
        commitment = _GLOBAL_TO_LIBRARY_DIAGNOSIS.get(provisional)
        hits = (self.ctx.guideline_retriever.search(commitment, args.claim_type, int(args.limit))
                if commitment is not None else [])
        payload = {
            "library_size": len(self.ctx.guideline_retriever.records),
            "library_conditions": self.ctx.guideline_retriever.conditions(),
            "patient_commitment": provisional,
            "claim_type": args.claim_type,
            "decision_authority": "retrieval_only_not_executable",
            "scope_warning": ("These passages may support method-level rationale only. They do not establish this "
                              "patient's diagnosis and cannot alter the reviewed-v2 next step."),
            "hits": [hit.to_evidence_record() for hit in hits],
        }
        propositions = self._propositions_from(payload, ref_prefix="guideline_library")
        return self._finish("search_guideline_library", a, record, ok=True, data=payload, propositions=propositions)

    def retrieve_case_comparisons(self, args: RetrieveCaseComparisonsArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("retrieve_case_comparisons", a)
        if early is not None:
            return early
        if self.ctx.memory_bank is None:
            return self._finish("retrieve_case_comparisons", a, record, ok=False,
                                error=ToolError("memory_unavailable", "no comparison memory bank configured"))
        if self.ctx.memory_policy is None:
            return self._finish("retrieve_case_comparisons", a, record, ok=False,
                                error=ToolError("memory_policy_unavailable", "no bank-bound comparison policy configured"))
        if not self.ctx.memory_request_explicit:
            case_state = self.ctx.store.load_case(self.ctx.case_id)
            if not reminder_allowed(tuple(e.to_dict() for e in case_state.episode_events)):
                return self._finish("retrieve_case_comparisons", a, record, ok=False,
                                    error=ToolError("memory_reminder_budget_exhausted", "one unsolicited comparison reminder was already displayed"))
        try:
            prediction = self.ctx.inference_client.predict(case_id=self.ctx.case_id)
            embedding = prediction["prediction"]["sentinel_embedding"]
        except Exception as exc:  # noqa: BLE001
            return self._finish("retrieve_case_comparisons", a, record, ok=False,
                                error=ToolError("embedding_unavailable", str(exc)))
        values = (record.facts or {}).get("values", {})
        source_to_role = {
            "development_allowlisted_case": DATA_ROLE_DEVELOPMENT,
            "development_oof_benchmark": DATA_ROLE_DEVELOPMENT,
            "synthetic_workflow_fixture": DATA_ROLE_DEVELOPMENT,
            "evaluation_query_only": DATA_ROLE_EVALUATION_QUERY,
            "deployment_query_only": DATA_ROLE_DEPLOYMENT_QUERY,
        }
        query = {
            "case_id": record.identity.case_id, "embedding": embedding, "embedding_model": V.MEMORY_EMBEDDING_MODEL,
            "data_role": source_to_role[record.identity.source_label], "image_sha256": record.inputs.dermoscopic.sanitized_sha256,
            "site": values.get("lesion.site"), "sex": values.get("patient.sex"), "elevation": values.get("lesion.elevation"),
            "criterion_values": dict(record.perception.criterion_values),
            "provisional_class": record.perception.diagnosis_head.get("provisional_class"),
            "identity": {"patient_id": None, "lesion_id": record.identity.case_id},
        }
        result = memory_compare(query, self.ctx.memory_bank, self.ctx.memory_policy)
        result = memory_refresh(result, self.ctx.memory_bank)
        payload = result.to_dict()
        payload["memory_request_explicit"] = self.ctx.memory_request_explicit
        payload["unsolicited_reminder_proposed"] = bool(not self.ctx.memory_request_explicit and result.status == "ok")
        label_counts: Dict[str, int] = {}
        for match in payload.get("matches", []):
            label = str((match.get("outcome") or {}).get("diagnosis") or "").strip()
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
        payload["descriptive_reference_label_counts"] = {
            "n_retrieved": len(payload.get("matches", [])),
            "counts": dict(sorted(label_counts.items())),
            "scope": "returned_reference_cases_only",
            "diagnostic_effect": "not_used_to_alter_current_prediction",
        }
        payload["grounded_summary_text"] = _memory_grounded_summary(payload)
        propositions = self._propositions_from(payload, ref_prefix="memory")
        return self._finish("retrieve_case_comparisons", a, record, ok=True, data=payload, propositions=propositions)

    def request_measured_xai(self, args: RequestMeasuredXaiArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("request_measured_xai", a)
        if early is not None:
            return early
        if args.kind not in XAI_KINDS:
            return self._finish("request_measured_xai", a, record, ok=False,
                                error=ToolError("invalid_xai_kind", f"{args.kind!r} not in {sorted(XAI_KINDS)}"))
        if self.ctx.xai_job_api is None:
            return self._finish("request_measured_xai", a, record, ok=False,
                                error=ToolError("xai_unavailable", "no xai job API configured"))
        job = self.ctx.xai_job_api(case_id=self.ctx.case_id, record_sha256=record.record_sha256, kind=args.kind)
        job = dict(job)
        propositions = self._propositions_from(job, ref_prefix="xai.request")
        return self._finish("request_measured_xai", a, record, ok=True, data=job, propositions=propositions)

    def read_measured_xai(self, args: ReadMeasuredXaiArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("read_measured_xai", a)
        if early is not None:
            return early
        if args.kind not in XAI_KINDS:
            return self._finish("read_measured_xai", a, record, ok=False,
                                error=ToolError("invalid_xai_kind", f"{args.kind!r} not in {sorted(XAI_KINDS)}"))
        receipts = self.ctx.store.load_xai_receipts(self.ctx.case_id)
        candidates = [r for r in receipts if _receipt_kind(r) == args.kind]
        if not candidates:
            return self._finish("read_measured_xai", a, record, ok=False,
                                error=ToolError("no_receipt", f"no stored {args.kind} XAI receipt for this case"))
        receipt = candidates[-1]
        record_dict = record.to_dict()
        try:
            verify_receipt(receipt, record_dict)
        except ReceiptMismatch as exc:
            return self._finish("read_measured_xai", a, record, ok=False,
                                error=ToolError("receipt_mismatch", str(exc)))
        evidence = receipt.get("results") or receipt.get("targets") or {}
        summary_text = _xai_grounded_summary(args.kind, receipt, record)
        propositions = self._propositions_from(evidence, ref_prefix=f"xai.{args.kind}")
        propositions += self._propositions_from(
            _xai_grounded_fields(args.kind, receipt, record), ref_prefix=f"xai.{args.kind}.summary_fields"
        )
        data = {**receipt, "receipt_provenance": receipt_provenance(receipt, record_dict),
                "grounded_summary_text": summary_text}
        return self._finish("read_measured_xai", a, record, ok=True, data=data, propositions=propositions)

    def propose_review(self, args: ProposeReviewArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("propose_review", a)
        if early is not None:
            return early
        case_state = self.ctx.store.load_case(self.ctx.case_id)
        state = derive_state(self._episode_id, self.ctx.case_id, case_state.episode_events)
        idempotency_key = f"agent-propose-review:{record.record_sha256}:{args.reason}"
        result = request_review(
            state, reason=args.reason, actor=self.ctx.actor, session_id=self.ctx.session_id,
            idempotency_key=idempotency_key, at=self._clock(), current_record_version=record.identity.record_version,
        )
        payload = {
            "kind": "review", "status": result.status, "reason": result.reason, "detail": result.detail,
            "proposed_event": (result.event.to_dict() if result.event is not None else None),
        }
        error = None if result.status == "ok" else ToolError(result.reason or "rejected", result.detail)
        propositions = self._propositions_from(payload, ref_prefix="episode.review")
        return self._finish("propose_review", a, record, ok=(result.status == "ok"), data=payload,
                            propositions=propositions, error=error)

    def propose_reacquisition(self, args: ProposeReacquisitionArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("propose_reacquisition", a)
        if early is not None:
            return early
        case_state = self.ctx.store.load_case(self.ctx.case_id)
        state = derive_state(self._episode_id, self.ctx.case_id, case_state.episode_events)
        reason = f"reacquisition requested: {args.reason}"
        idempotency_key = f"agent-propose-reacquisition:{record.record_sha256}:{args.reason}"
        result = request_review(
            state, reason=reason, actor=self.ctx.actor, session_id=self.ctx.session_id,
            idempotency_key=idempotency_key, at=self._clock(), current_record_version=record.identity.record_version,
        )
        payload = {
            "kind": "reacquisition", "status": result.status, "reason": result.reason, "detail": result.detail,
            "proposed_event": (result.event.to_dict() if result.event is not None else None),
        }
        error = None if result.status == "ok" else ToolError(result.reason or "rejected", result.detail)
        propositions = self._propositions_from(payload, ref_prefix="episode.reacquisition")
        return self._finish("propose_reacquisition", a, record, ok=(result.status == "ok"), data=payload,
                            propositions=propositions, error=error)

    def read_review_history(self, args: ReadReviewHistoryArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("read_review_history", a)
        if early is not None:
            return early
        case_state = self.ctx.store.load_case(self.ctx.case_id)
        state = derive_state(self._episode_id, self.ctx.case_id, case_state.episode_events)
        pending_proposal = None
        for pid, (case_id, session_id) in self._proposal_owners.items():
            if case_id == self.ctx.case_id and session_id == self.ctx.session_id:
                pending_proposal = {"proposal_id": pid}
        remains = what_remains(state, record, pending_proposal=pending_proposal)
        payload = {"episode": state.to_dict(), "what_remains": {"action": remains.action, "sentence": remains.sentence}}
        propositions = (
            {"id": self._next_pid(), "path_or_ref": "episode.what_remains.action", "value": remains.action, "text": remains.sentence},
        )
        return self._finish("read_review_history", a, record, ok=True, data=payload, propositions=propositions)

    def submit_explanation(self, args: SubmitExplanationArgs) -> ToolResult:
        a = dataclasses.asdict(args)
        early, record = self._dispatch("submit_explanation", a)
        if early is not None:
            return early
        from dermpact.core.agent.checker import check_explanation
        from dermpact.core.agent.config import WORKFLOW_CONFIG
        verdict = check_explanation(args.text, list(args.claims), record, self.history, WORKFLOW_CONFIG)
        payload = {
            "ok": verdict.ok, "rejections": [dataclasses.asdict(r) for r in verdict.rejections],
            "matched_claims": list(verdict.matched_claims),
        }
        error = None if verdict.ok else ToolError("explanation_rejected", "; ".join(r.code for r in verdict.rejections), repairable=True)
        return self._finish("submit_explanation", a, record, ok=verdict.ok, data=payload, error=error)


__all__ = [
    "ToolContext", "ToolError", "ToolResult", "ToolRegistry", "XAI_KINDS",
    "ReadRecordSectionArgs", "ReadReliabilityArgs", "FindDecisiveMissingFactsArgs",
    "ProposeFactAmendmentArgs", "ProposeCriterionAmendmentArgs", "PreviewConsequencesArgs",
    "RefreshApplicabilityArgs", "RetrieveGuidelineEvidenceArgs", "SearchGuidelineLibraryArgs", "RetrieveCaseComparisonsArgs",
    "RequestMeasuredXaiArgs", "ReadMeasuredXaiArgs", "ProposeReviewArgs", "ProposeReacquisitionArgs",
    "ReadReviewHistoryArgs", "SubmitExplanationArgs",
]
