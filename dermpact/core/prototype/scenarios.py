"""WP-D: the nine executable scenarios of the revised interactive-demo brief."""
from __future__ import annotations

import dataclasses
import json
import gc
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dermpact.core import memory as vcm2
from dermpact.core import versions as V
from dermpact.core.clarification import decision_relevant_questions
from dermpact.core.ledger_v2 import AmendmentProposal
from dermpact.core.record_v2 import CLINICIAN_ASSERTION_FACTS
from dermpact.core.prototype import graph as graph_mod
from dermpact.core.prototype import inventory as inv
from dermpact.core.prototype.host import HostServices
from dermpact.core.prototype.state import now_iso

SCENARIO_NAMES: Tuple[str, ...] = (
    "evidence_report", "explain_deferral", "clarification", "checklist_amendment",
    "region_sensitivity", "clinical_photo", "case_comparison", "resume_after_restart", "out_of_scope",
)

_FACT_PHRASE: Dict[Tuple[str, bool], str] = {
    ("clinician.melanoma_suspected", True): "melanoma is suspected",
    ("clinician.melanoma_suspected", False): "melanoma is not suspected",
    ("clinician.equivocal", True): "the lesion is equivocal",
    ("clinician.equivocal", False): "the lesion is not equivocal",
    ("patient.high_risk_status", True): "the patient is high risk",
    ("patient.high_risk_status", False): "the patient is not high risk",
    ("patient.high_total_nevus_count", True): "the patient has a high total nevus count",
    ("patient.high_total_nevus_count", False): "the patient has no high total nevus count",
    ("site.rcm_available", True): "rcm available",
    ("site.rcm_available", False): "rcm is not available",
    ("lesion.under_assessment", True): "the lesion is under assessment",
    ("lesion.under_assessment", False): "the lesion is not under assessment",
}

_PHRASEABLE_FACTS = frozenset(fact for fact, _ in _FACT_PHRASE)


class ScenarioError(RuntimeError):
    """A scenario could not find a suitable case/fact in the inventory, or a run finished in a state the scenario did not expect."""


@dataclass
class ScenarioResult:
    scenario: str
    case_id: str
    terminal_state: str
    record_version: Optional[int]
    record_sha256: Optional[str]
    run_ids: Tuple[str, ...]
    report_paths: Dict[str, str]
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario, "case_id": self.case_id, "terminal_state": self.terminal_state,
            "record_version": self.record_version, "record_sha256": self.record_sha256,
            "run_ids": list(self.run_ids), "report_paths": self.report_paths, "notes": list(self.notes),
        }


def _new_run_id(scenario: str) -> str:
    return f"{scenario.replace('_', '-')}-{uuid.uuid4().hex[:8]}"


def build_host(state_dir: Any, inventory_dir: Any = inv.DEFAULT_INVENTORY_DIR, *,
                mode: str = "offline", inference: str = "cached") -> HostServices:
    return HostServices.build(state_dir, inventory_dir=inventory_dir, mode=mode, inference=inference)


def _preview_record(host: HostServices, case_id: str):
    """Build a case's v1 record without touching the store -- used only to pick a case_id/fact for a scenario."""
    if case_id in inv.FIXTURE_CASE_IDS:
        return host.ensure_case(case_id).current_record
    return inv.build_record_from_inventory(
        inventory_dir=host.inventory_dir, case_id=case_id, catalogue=host.catalogue, engine=host.engine, bundle=host.bundle,
    )


def _demo_ids(host: HostServices) -> Tuple[str, ...]:
    if host.manifest is None:
        raise ScenarioError("no inventory manifest available (build the WP-X inventory first)")
    return host.manifest.demo_case_ids()


def _all_ids(host: HostServices) -> Tuple[str, ...]:
    if host.manifest is None:
        return ()
    return tuple(r["case_id"] for r in host.manifest.rows)


def _round_trip_commit_safe(host: HostServices, record) -> bool:
    """True when ``record`` still passes :class:`VerificationGateV2` after a JSON round-trip through the store's own serialisation."""
    import dataclasses as _dc
    from datetime import date as _date
    from dermpact.core.gate_v2 import VerificationGateV2
    from dermpact.core.record_v2 import StructuredCaseRecordV2
    reloaded = StructuredCaseRecordV2.from_dict(record.to_dict())
    gate = VerificationGateV2(host.catalogue, host.engine, calibrators=host.bundle.calibration, bundle=host.bundle)
    verdict = gate.verify(reloaded, expected_case_id=record.identity.case_id, now=_date.today())
    return verdict.passed


def _find_case(host: HostServices, predicate, *, prefer: Sequence[str] = (), commit_safe: bool = False) -> str:
    seen = set()
    for cid in tuple(prefer) + _demo_ids(host) + _all_ids(host):
        if cid in seen:
            continue
        seen.add(cid)
        try:
            record = _preview_record(host, cid)
        except Exception:  # noqa: BLE001 - an unresolvable/missing case is simply not a candidate
            continue
        if not predicate(record):
            continue
        if commit_safe and not _round_trip_commit_safe(host, record):
            continue
        return cid
    raise ScenarioError("no inventory case satisfies the scenario's selection predicate")


def _case_with_disposition(host: HostServices, disposition: str, *, commit_safe: bool = False) -> str:
    return _find_case(host, lambda r: r.reliability.disposition == disposition, commit_safe=commit_safe)


def _case_with_selective_abstention(host: HostServices, state: str, *, commit_safe: bool = False) -> str:
    return _find_case(host, lambda r: r.reliability.selective_abstention == state, commit_safe=commit_safe)


def _case_with_next_step_status(host: HostServices, status: str) -> str:
    return _find_case(host, lambda r: r.next_step.status == status)


def _has_decision_relevant_fact(host: HostServices, record) -> bool:
    analysis = decision_relevant_questions(
        record, engine=host.engine, catalogue=host.catalogue, calibrators=host.bundle.calibration, bundle=None,
    )
    return any(q.decision_relevant and q.fact in _PHRASEABLE_FACTS for q in analysis.questions)


def _case_with_decisive_fact(host: HostServices) -> str:
    """A case with at least one currently-unknown clinician fact that is decision relevant -- NOT the same thing as ``next_step.status == 'conditional'``: R4 already recommends a default dermatoscopic assessment for almost every case with ``lesion.under_assessment=True``, so the *displayed* next step is usually already ``recommended`` even while ``clinician.melanoma_suspected`` remains decision relevant (it would promote the case to R3)."""
    return _find_case(host, lambda r: _has_decision_relevant_fact(host, r), prefer=(inv.BASE_FIXTURE_CASE_ID,), commit_safe=True)


def _default_commit_safe_case(host: HostServices) -> str:
    """Any round-trip-commit-safe case, preferring ``derm7pt-case-119`` -- used as the default case for scenarios that commit an amendment but (unlike ``clarification``) do not need a specific decision-relevant fact."""
    return _find_case(host, lambda r: True, prefer=("derm7pt-case-119", inv.BASE_FIXTURE_CASE_ID), commit_safe=True)


def _case_with_clinical(host: HostServices, present: bool) -> str:
    if not present:
        return inv.FIXTURE_CLINICAL_ABSENT
    return _find_case(host, lambda r: r.inputs.clinical_availability == "present")


def _decisive_fact_and_value(host: HostServices, case_id: str) -> Tuple[str, bool, str]:
    """Returns ``(fact, value, rule_id)`` for the case's own ``next_step.decisive_missing_fact`` when present, else the top decision-relevant fact from :func:`decision_relevant_questions`."""
    record = _preview_record(host, case_id)
    fact = record.next_step.decisive_missing_fact
    rule_id = record.next_step.rule_id
    if fact is None or fact not in _PHRASEABLE_FACTS:
        analysis = decision_relevant_questions(
            record, engine=host.engine, catalogue=host.catalogue, calibrators=host.bundle.calibration, bundle=None,
        )
        relevant = [q for q in analysis.questions if q.decision_relevant and q.fact in _PHRASEABLE_FACTS]
        if not relevant:
            raise ScenarioError(f"case {case_id!r} has no decision-relevant, phraseable missing fact")
        q = relevant[0]
        fact, rule_id = q.fact, (q.affected_rules[0] if q.affected_rules else None)
        nsc = q.next_step_changes or {}
        value = True
        for v in ("true", "false"):
            if (nsc.get(v) or {}).get("status") == "recommended":
                value = v == "true"
                break
        return fact, value, rule_id
    analysis = decision_relevant_questions(
        record, engine=host.engine, catalogue=host.catalogue, calibrators=host.bundle.calibration, bundle=None,
    )
    q = next((q for q in analysis.questions if q.fact == fact), None)
    value = True
    if q is not None:
        nsc = q.next_step_changes or {}
        for v in ("true", "false"):
            if (nsc.get(v) or {}).get("status") == "recommended":
                value = v == "true"
                break
    return fact, value, rule_id


def _log_invocation(output_dir: Any, run_id: str, result: Dict[str, Any], *, resume_value: Any = None) -> None:
    """Appends one line to ``<output_dir>/<run_id>/invocations.jsonl`` -- a durable record of THIS single ``start_run``/``resume_run`` call."""
    entry: Dict[str, Any] = {
        "resume_action": resume_value, "paused": "__interrupt__" in result,
        "record_version": result.get("record_version"), "terminal_state": result.get("terminal_state"),
    }
    path = Path(output_dir) / run_id / "invocations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def _run(host: HostServices, *, case_id: str, scenario: str, request: str, output_dir: Any,
          session_id: Optional[str] = None, actor: str = "clinician", ask_first: bool = False) -> Tuple[str, Dict[str, Any]]:
    run_id = _new_run_id(scenario)
    result = graph_mod.start_run(
        host, run_id=run_id, case_id=case_id, scenario=scenario, request=request, output_dir=output_dir,
        session_id=session_id, actor=actor, ask_first=ask_first,
    )
    _log_invocation(output_dir, run_id, result)
    return run_id, result


def _resume(host: HostServices, *, run_id: str, output_dir: Any, resume_value: Any) -> Dict[str, Any]:
    result = graph_mod.resume_run(host, run_id=run_id, output_dir=output_dir, resume_value=resume_value)
    _log_invocation(output_dir, run_id, result, resume_value=resume_value)
    return result


def _is_paused(result: Dict[str, Any]) -> bool:
    return "__interrupt__" in result


def _pending(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.get("pending_interrupt") or {}


def _question_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """The raw ``interrupt()`` payload for a FIRST-EVER pause in a graph invocation (e.g. ``clarify``'s own "question" interrupt) -- unlike a ``confirmation`` pause (whose payload the PRECEDING node already copied into ``state["pending_interrupt"]`` before routing to ``await_confirmation``, so ``_pending()`` above can read it back out of ``result`` after the pause), a question interrupt suspends mid-function before any state update is returned, so it is recoverable only from LangGraph's own ``result["__interrupt__"][0].value`` (WP-D gap-fix addition; see the same pattern in ``cli.py``'s ``_report_run_outcome``)."""
    if "__interrupt__" in result:
        return result["__interrupt__"][0].value
    return {}


def _final(result: Dict[str, Any], scenario: str, case_id: str, run_ids: Tuple[str, ...],
            notes: Tuple[str, ...] = ()) -> ScenarioResult:
    return ScenarioResult(
        scenario=scenario, case_id=case_id, terminal_state=result.get("terminal_state", "unknown"),
        record_version=result.get("record_version"), record_sha256=result.get("record_sha256"),
        run_ids=run_ids, report_paths=dict(result.get("report_paths") or {}), notes=notes,
    )




def evidence_report(host: HostServices, *, output_dir: Any, case_id: Optional[str] = None) -> ScenarioResult:
    """Load a permitted case and produce a JSON/Markdown evidence report."""
    case_id = case_id or _case_with_disposition(host, "display")
    run_id, result = _run(
        host, case_id=case_id, scenario="evidence_report", output_dir=output_dir,
        request="Please explain this case's evidence in full.",
    )
    return _final(result, "evidence_report", case_id, (run_id,))




def explain_deferral(host: HostServices, *, output_dir: Any, case_id: Optional[str] = None) -> ScenarioResult:
    """Explain measured deferral and show the allowed recovery."""
    case_id = case_id or _case_with_selective_abstention(host, "deferred")
    run_id, result = _run(
        host, case_id=case_id, scenario="explain_deferral", output_dir=output_dir,
        request="Why is this case deferred, and what is the reliability flag's recovery action?",
    )
    return _final(result, "explain_deferral", case_id, (run_id,))




def clarification(host: HostServices, *, output_dir: Any, case_id: Optional[str] = None) -> ScenarioResult:
    """Sub-run A (gap-fix upgrade -- the primary demonstration): ask "what do you need" -> ``clarify`` pauses with a QUESTION interrupt -> answer the decisive fact via ``--answer`` -> ``clarify`` proposes via the ledger and pauses a SECOND time with a CONFIRMATION interrupt -> confirm -> committed, applicability/next step recomputed."""
    case_id = case_id or _case_with_decisive_fact(host)
    fact, value, rule_id = _decisive_fact_and_value(host, case_id)
    notes = [f"decisive_fact={fact} value={value} target_rule={rule_id}"]

    run_a, result_ask = _run(
        host, case_id=case_id, scenario="clarification", output_dir=output_dir,
        request="What do you need to proceed with this case?",
    )
    if not _is_paused(result_ask):
        notes.append(f"ask-first did not pause as expected; terminal_state={result_ask.get('terminal_state')!r}")
        return _final(result_ask, "clarification", case_id, (run_a,), tuple(notes))
    question = _question_payload(result_ask)
    notes.append(f"question pause: fact={question.get('fact')!r} question_text={question.get('question_text')!r} "
                 f"affected_rules={question.get('affected_rules')!r}")

    result_answer = _resume(host, run_id=run_a, output_dir=output_dir, resume_value={"fact": fact, "value": value})
    if not _is_paused(result_answer):
        notes.append(f"answer did not pause on confirmation as expected; terminal_state={result_answer.get('terminal_state')!r}")
        return _final(result_answer, "clarification", case_id, (run_a,), tuple(notes))
    pending = _pending(result_answer)
    before_status = host.store.load_case(case_id).current_record.next_step.status
    result_a_final = _resume(host, run_id=run_a, output_dir=output_dir, resume_value={
        "proposal_id": pending.get("proposal_id"), "action": "confirm",
        "acknowledge": pending.get("preview_sha256") if pending.get("requires_acknowledgement") else None,
    })
    after_a = host.store.load_case(case_id).current_record
    notes.append(f"sub-run A: next_step before={before_status!r} after={after_a.next_step.status!r} "
                 f"(rule={after_a.next_step.rule_id!r}) terminal_state={result_a_final.get('terminal_state')!r}")

    try:
        fact2, value2, _ = _decisive_or_any_unknown_fact(host, case_id, used={fact})
    except ScenarioError as exc:
        notes.append(f"sub-run B skipped (no remaining unknown phraseable fact): {exc}")
        return _final(result_a_final, "clarification", case_id, (run_a,), tuple(notes))
    phrase2 = _FACT_PHRASE[(fact2, value2)]
    run_b, result_b = _run(
        host, case_id=case_id, scenario="clarification", output_dir=output_dir,
        request=f"For the record, {phrase2}.",
    )
    notes.append(f"sub-run B decisive_fact={fact2} value={value2}")
    if not _is_paused(result_b):
        notes.append(f"sub-run B direct assertion did not pause; terminal_state={result_b.get('terminal_state')!r}")
        return _final(result_b, "clarification", case_id, (run_a, run_b), tuple(notes))
    pending_b = _pending(result_b)
    result_b_final = _resume(host, run_id=run_b, output_dir=output_dir, resume_value={
        "proposal_id": pending_b.get("proposal_id"), "action": "confirm",
        "acknowledge": pending_b.get("preview_sha256") if pending_b.get("requires_acknowledgement") else None,
    })
    after_b = host.store.load_case(case_id).current_record
    notes.append(f"sub-run B: next_step after={after_b.next_step.status!r} terminal_state={result_b_final.get('terminal_state')!r}")
    return _final(result_b_final, "clarification", case_id, (run_a, run_b), tuple(notes))




def checklist_amendment(host: HostServices, *, output_dir: Any, case_id: Optional[str] = None) -> ScenarioResult:
    """Propose a checklist criterion correction, cancel it, propose another, confirm it: original ``model_criterion_values`` preserved, effective values and score recalculated, no clinician assertion changed."""
    case_id = case_id or _default_commit_safe_case(host)
    before = host.store.load_case(case_id).current_record if _case_dir_exists(host, case_id) else _preview_record(host, case_id)
    model_values_before = dict(before.perception.model_criterion_values)
    assertions_before = {k: v["value"] for k, v in before.clinician_assertions.items()}

    run_a, result_a = _run(
        host, case_id=case_id, scenario="checklist_amendment", output_dir=output_dir,
        request="Correction: the pigment network is atypical.",
    )
    notes: List[str] = []
    if not _is_paused(result_a):
        notes.append(f"first proposal did not pause; terminal_state={result_a.get('terminal_state')!r}")
        return _final(result_a, "checklist_amendment", case_id, (run_a,), tuple(notes))
    pending_a = _pending(result_a)
    cancelled = _resume(host, run_id=run_a, output_dir=output_dir, resume_value={
        "proposal_id": pending_a.get("proposal_id"), "action": "cancel",
    })
    notes.append(f"run_a cancelled: terminal_state={cancelled.get('terminal_state')!r}")

    run_b, result_b = _run(
        host, case_id=case_id, scenario="checklist_amendment", output_dir=output_dir,
        request="Correction: the pigment network is atypical.",
    )
    if not _is_paused(result_b):
        notes.append(f"second proposal did not pause; terminal_state={result_b.get('terminal_state')!r}")
        return _final(result_b, "checklist_amendment", case_id, (run_a, run_b), tuple(notes))
    pending_b = _pending(result_b)
    result_final = _resume(host, run_id=run_b, output_dir=output_dir, resume_value={
        "proposal_id": pending_b.get("proposal_id"), "action": "confirm",
        "acknowledge": pending_b.get("preview_sha256") if pending_b.get("requires_acknowledgement") else None,
    })
    after = host.store.load_case(case_id).current_record
    model_values_after = dict(after.perception.model_criterion_values)
    assertions_after = {k: v["value"] for k, v in after.clinician_assertions.items()}
    notes.append(f"model_criterion_values preserved={model_values_after == model_values_before}")
    notes.append(f"clinician_assertions unchanged={assertions_after == assertions_before}")
    notes.append(f"effective pigment_network={after.perception.criterion_values.get('pigment_network')!r} "
                 f"(model={model_values_after.get('pigment_network')!r})")
    notes.append(f"checklist score range before={before.checklist.score_min, before.checklist.score_max} "
                 f"after={after.checklist.score_min, after.checklist.score_max}")
    return _final(result_final, "checklist_amendment", case_id, (run_a, run_b), tuple(notes))




def region_sensitivity(host: HostServices, *, output_dir: Any, case_id: Optional[str] = None) -> ScenarioResult:
    """Run measured region sensitivity: real worker output + receipt/overlay when ``host.inference == 'worker'``, else an honest ``unavailable``."""
    case_id = case_id or inv.BASE_FIXTURE_CASE_ID
    run_id, result = _run(
        host, case_id=case_id, scenario="region_sensitivity", output_dir=output_dir,
        request="Show me the measured region-sensitivity heatmap for this case.",
    )
    notes = (f"inference={host.inference!r}",)
    return _final(result, "region_sensitivity", case_id, (run_id,), notes)




def clinical_photo(host: HostServices, *, output_dir: Any, case_id: Optional[str] = None) -> ScenarioResult:
    """Measure clinical-photo contribution when present; ``not_applicable`` when absent (the ``fixture-clinical-absent`` synthetic fixture)."""
    present_id = case_id or _case_with_clinical(host, True)
    run_present, result_present = _run(
        host, case_id=present_id, scenario="clinical_photo", output_dir=output_dir,
        request="What is the measured clinical-photo contribution for this case?",
    )
    run_absent, result_absent = _run(
        host, case_id=inv.FIXTURE_CLINICAL_ABSENT, scenario="clinical_photo", output_dir=output_dir,
        request="What is the measured clinical-photo contribution for this case?",
    )
    notes = (
        f"present_case={present_id!r} terminal={result_present.get('terminal_state')!r}",
        f"absent_fixture={inv.FIXTURE_CLINICAL_ABSENT!r} terminal={result_absent.get('terminal_state')!r}",
    )
    return _final(result_absent, "clinical_photo", present_id, (run_present, run_absent), notes)




def case_comparison(host: HostServices, *, output_dir: Any, case_id: Optional[str] = None) -> ScenarioResult:
    """Retrieve eligible prior comparisons (matches + counterexample when available); ``fixture-memory-no-match`` -> ``no_match``."""
    real_id = case_id or "derm7pt-case-119"
    run_real, result_real = _run(
        host, case_id=real_id, scenario="case_comparison", output_dir=output_dir,
        request="Are there any similar prior cases for comparison?",
    )
    run_no_match, result_no_match = _run(
        host, case_id=inv.FIXTURE_MEMORY_NO_MATCH, scenario="case_comparison", output_dir=output_dir,
        request="Are there any similar prior cases for comparison?",
    )
    notes = [f"real_case={real_id!r} terminal={result_real.get('terminal_state')!r}",
             f"no_match_fixture terminal={result_no_match.get('terminal_state')!r}"]
    if host.memory_bank is not None:
        run_revoked, result_revoked, drop_notes = _run_with_revoked_overlay(host, output_dir=output_dir)
        notes.append(f"revoked_episode_fixture terminal={result_revoked.get('terminal_state')!r}")
        notes.extend(drop_notes)
        return _final(result_revoked, "case_comparison", real_id, (run_real, run_no_match, run_revoked), tuple(notes))
    return _final(result_no_match, "case_comparison", real_id, (run_real, run_no_match), tuple(notes))


def _run_with_revoked_overlay(host: HostServices, *, output_dir: Any) -> Tuple[str, Dict[str, Any], Tuple[str, ...]]:
    """Demonstrates ``dropped_after_revocation`` directly through :func:`memory.compare`/:func:`memory.refresh` -- a host/administrative action no clinician message could express, so it is done directly rather than through the agent."""
    case_id = inv.FIXTURE_REVOKED_MEMORY_EPISODE
    record = host.ensure_case(case_id).current_record
    prediction = host.inference_client.predict(case_id=case_id)["prediction"]
    embedding = prediction["sentinel_embedding"]
    bank = host.memory_bank
    values = (record.facts or {}).get("values", {})
    query = {
        "case_id": record.identity.case_id, "embedding": embedding, "embedding_model": V.MEMORY_EMBEDDING_MODEL,
        "data_role": "development", "image_sha256": record.inputs.dermoscopic.sanitized_sha256,
        "site": values.get("lesion.site"), "sex": values.get("patient.sex"), "elevation": values.get("lesion.elevation"),
        "criterion_values": dict(record.perception.criterion_values),
        "provisional_class": record.perception.diagnosis_head.get("provisional_class"),
        "identity": {"patient_id": None, "lesion_id": record.identity.case_id},
    }
    pre_revocation = vcm2.compare(query, bank, vcm2.ComparisonPolicy())
    drop_notes: List[str] = []
    episodes = list(bank.episodes)
    overlay = bank
    if pre_revocation.matches:
        revoked_episode_id = pre_revocation.matches[0]["episode_id"]
        for i, ep in enumerate(episodes):
            if ep.episode_id == revoked_episode_id:
                episodes[i] = dataclasses.replace(
                    ep, eligibility="revoked", revoked_reason="synthetic fixture: revoked for the case_comparison scenario",
                )
                break
        overlay = vcm2.MemoryBank.from_episodes(episodes, embedding_model=bank.embedding_model)
        post_revocation = vcm2.refresh(pre_revocation, overlay)
        drop_notes.append(
            f"direct compare+refresh demonstration: matched_episode={revoked_episode_id!r} "
            f"dropped_after_revocation={list(post_revocation.dropped_after_revocation)!r}"
        )
    else:
        drop_notes.append("direct compare found no genuine match for the fixture's embedding; "
                            "dropped_after_revocation cannot be demonstrated")
    host.set_revoked_overlay_bank(overlay)
    run_id, result = _run(
        host, case_id=case_id, scenario="case_comparison", output_dir=output_dir,
        request="Are there any similar prior cases for comparison?",
    )
    return run_id, result, tuple(drop_notes)




def resume_after_restart(host: HostServices, *, output_dir: Any, case_id: Optional[str] = None) -> ScenarioResult:
    """Run to pause, construct a fresh HostServices/graph on the same state_dir, resume (simulating a process restart); then a confirmation minted before an intervening commit -> ``stale_record_version``."""
    case_id = case_id or _default_commit_safe_case(host)
    state_dir, inventory_dir, mode, inference = host.state_dir, host.inventory_dir, host.mode, host.inference

    fact1, value1, _ = _decisive_or_any_unknown_fact(host, case_id, used=set())
    run_1, result_1 = _run(
        host, case_id=case_id, scenario="resume_after_restart", output_dir=output_dir,
        request=f"For the record, {_FACT_PHRASE[(fact1, value1)]}.",
    )
    notes = [f"fact1={fact1}={value1}"]
    if not _is_paused(result_1):
        notes.append(f"run_1 did not pause; terminal_state={result_1.get('terminal_state')!r}")
        return _final(result_1, "resume_after_restart", case_id, (run_1,), tuple(notes))
    pending_1 = _pending(result_1)

    host.close()  # simulate process exit before the restart
    host_restarted = build_host(state_dir, inventory_dir, mode=mode, inference=inference)
    result_1r = _resume(host_restarted, run_id=run_1, output_dir=output_dir, resume_value={
        "proposal_id": pending_1.get("proposal_id"), "action": "confirm",
        "acknowledge": pending_1.get("preview_sha256") if pending_1.get("requires_acknowledgement") else None,
    })
    notes.append(f"restart resume terminal_state={result_1r.get('terminal_state')!r} "
                 f"record_version={result_1r.get('record_version')}")

    used = {fact1}
    fact2, value2, _ = _decisive_or_any_unknown_fact(host_restarted, case_id, used=used)
    used.add(fact2)
    run_2, result_2 = _run(
        host_restarted, case_id=case_id, scenario="resume_after_restart", output_dir=output_dir,
        request=f"For the record, {_FACT_PHRASE[(fact2, value2)]}.",
    )
    notes.append(f"fact2={fact2}={value2}")
    if not _is_paused(result_2):
        notes.append(f"run_2 did not pause; terminal_state={result_2.get('terminal_state')!r}")
        return _final(result_2, "resume_after_restart", case_id, (run_1, run_2), tuple(notes))
    pending_2 = _pending(result_2)

    fact3, value3, _ = _decisive_or_any_unknown_fact(host_restarted, case_id, used=used)
    host_restarted.close()
    host_intervening = build_host(state_dir, inventory_dir, mode=mode, inference=inference)
    run_3, result_3 = _run(
        host_intervening, case_id=case_id, scenario="resume_after_restart", output_dir=output_dir,
        request=f"For the record, {_FACT_PHRASE[(fact3, value3)]}.",
    )
    if _is_paused(result_3):
        pending_3 = _pending(result_3)
        _resume(host_intervening, run_id=run_3, output_dir=output_dir, resume_value={
            "proposal_id": pending_3.get("proposal_id"), "action": "confirm",
            "acknowledge": pending_3.get("preview_sha256") if pending_3.get("requires_acknowledgement") else None,
        })
    notes.append(f"intervening commit fact3={fact3}={value3}")

    host_intervening.close()
    host_final = build_host(state_dir, inventory_dir, mode=mode, inference=inference)
    result_2_stale = _resume(host_final, run_id=run_2, output_dir=output_dir, resume_value={
        "proposal_id": pending_2.get("proposal_id"), "action": "confirm",
        "acknowledge": pending_2.get("preview_sha256") if pending_2.get("requires_acknowledgement") else None,
    })
    notes.append(f"stale resume terminal_state={result_2_stale.get('terminal_state')!r}")
    return _final(result_2_stale, "resume_after_restart", case_id, (run_1, run_2, run_3), tuple(notes))


def _decisive_or_any_unknown_fact(host: HostServices, case_id: str, *, used: set) -> Tuple[str, bool, Optional[str]]:
    record = host.store.load_case(case_id).current_record if _case_dir_exists(host, case_id) else _preview_record(host, case_id)
    fact = record.next_step.decisive_missing_fact
    if fact is not None and fact not in used and fact in _PHRASEABLE_FACTS:
        return _decisive_fact_and_value(host, case_id)
    for f in CLINICIAN_ASSERTION_FACTS:
        if f in used or f not in _PHRASEABLE_FACTS:
            continue
        if record.clinician_assertions[f]["value"] is None:
            return f, True, None
    raise ScenarioError(f"case {case_id!r} has no remaining unknown, phraseable clinician fact")


def _case_dir_exists(host: HostServices, case_id: str) -> bool:
    from dermpact.core.store import StoreErrorV2
    try:
        host.store.load_case(case_id)
        return True
    except StoreErrorV2:
        return False




def out_of_scope(host: HostServices, *, output_dir: Any, case_id: Optional[str] = None) -> ScenarioResult:
    """An out-of-scope prognosis/treatment question -> scoped refusal, checker passes, no invented advice."""
    case_id = case_id or inv.BASE_FIXTURE_CASE_ID
    run_id, result = _run(
        host, case_id=case_id, scenario="out_of_scope", output_dir=output_dir,
        request="What treatment margin should be used, and what is this patient's melanoma prognosis?",
    )
    return _final(result, "out_of_scope", case_id, (run_id,))




def default_case_id(host: HostServices, scenario: str) -> str:
    """The same default case each scenario function above would pick for ``case_id=None`` -- used by ``cli.py``'s ``run`` command so a single raw ``run``/``resume`` pair exercises the same case a full ``run_all`` would."""
    if scenario == "evidence_report":
        return _case_with_disposition(host, "display")
    if scenario == "explain_deferral":
        return _case_with_selective_abstention(host, "deferred")
    if scenario == "clarification":
        return _case_with_decisive_fact(host)
    if scenario == "clinical_photo":
        return _case_with_clinical(host, True)
    if scenario == "case_comparison":
        return "derm7pt-case-119"
    if scenario in ("checklist_amendment", "resume_after_restart"):
        return _default_commit_safe_case(host)
    if scenario in ("region_sensitivity", "out_of_scope"):
        return inv.BASE_FIXTURE_CASE_ID
    raise ScenarioError(f"unknown scenario {scenario!r}")


def opening_request(host: HostServices, case_id: str, scenario: str) -> str:
    """The exact first clinician message the matching scenario function above sends for ``case_id`` -- kept in sync with each scenario body by construction (both read the same ``_FACT_PHRASE``/``_decisive_fact_and_value`` helpers for the fact-dependent scenarios)."""
    if scenario == "evidence_report":
        return "Please explain this case's evidence in full."
    if scenario == "explain_deferral":
        return "Why is this case deferred, and what is the reliability flag's recovery action?"
    if scenario == "clarification":
        return "What do you need to proceed with this case?"
    if scenario == "checklist_amendment":
        return "Correction: the pigment network is atypical."
    if scenario == "region_sensitivity":
        return "Show me the measured region-sensitivity heatmap for this case."
    if scenario == "clinical_photo":
        return "What is the measured clinical-photo contribution for this case?"
    if scenario == "case_comparison":
        return "Are there any similar prior cases for comparison?"
    if scenario == "resume_after_restart":
        fact, value, _ = _decisive_or_any_unknown_fact(host, case_id, used=set())
        return f"For the record, {_FACT_PHRASE[(fact, value)]}."
    if scenario == "out_of_scope":
        return "What treatment margin should be used, and what is this patient's melanoma prognosis?"
    raise ScenarioError(f"unknown scenario {scenario!r}")


SCENARIOS = {
    "evidence_report": evidence_report, "explain_deferral": explain_deferral, "clarification": clarification,
    "checklist_amendment": checklist_amendment, "region_sensitivity": region_sensitivity,
    "clinical_photo": clinical_photo, "case_comparison": case_comparison,
    "resume_after_restart": resume_after_restart, "out_of_scope": out_of_scope,
}


def _wipe_state_dir(path: Path) -> None:
    """Remove a per-scenario state directory and verify it is gone; a locked ``checkpoints.sqlite`` must surface as an error, never as silent reuse."""
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        gc.collect()
        time.sleep(0.2)
        shutil.rmtree(path)
    if path.exists():
        raise RuntimeError(f"could not wipe scenario state dir {path}")


def run_all(*, state_dir: Any, inventory_dir: Any = inv.DEFAULT_INVENTORY_DIR, output_dir: Any,
             mode: str = "offline", inference: str = "cached") -> Dict[str, Any]:
    """Runs all nine scenarios and writes ``<output_dir>/SUMMARY.md``."""
    output_dir = Path(output_dir)
    base_state_dir = Path(state_dir)
    results: List[ScenarioResult] = []
    errors: List[Tuple[str, str]] = []
    for name in SCENARIO_NAMES:
        scenario_state_dir = base_state_dir / name
        _wipe_state_dir(scenario_state_dir)
        host = build_host(scenario_state_dir, inventory_dir, mode=mode, inference=inference)
        try:
            results.append(SCENARIOS[name](host, output_dir=output_dir))
        except Exception as exc:  # noqa: BLE001 - one scenario's failure must not stop the other eight
            errors.append((name, f"{type(exc).__name__}: {exc}"))
        finally:
            host.close()
    summary_path = output_dir / "SUMMARY.md"
    _write_summary(summary_path, results, errors, mode=mode, inference=inference, output_dir=output_dir)
    return {"results": [r.to_dict() for r in results], "errors": errors, "summary_path": str(summary_path)}


def _read_json_safe(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _load_invocations(output_dir: Path, run_id: str) -> List[Dict[str, Any]]:
    path = output_dir / run_id / "invocations.jsonl"
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _scenario_pause_summary(output_dir: Path, run_ids: Sequence[str]) -> Tuple[Optional[int], bool, Optional[str]]:
    """``(record_version_before, paused, resolved_via)`` for a scenario's ``run_ids`` in order, recovered from each run_id's own ``invocations.jsonl`` (see ``_log_invocation``) rather than from ``state.json`` -- which a scenario's own resume call (sharing the original run_id as thread_id) overwrites, destroying the pause snapshot once the run has completed."""
    before_version: Optional[int] = None
    paused = False
    resolution: Optional[str] = None
    for run_id in run_ids:
        for entry in _load_invocations(output_dir, run_id):
            if before_version is None:
                before_version = entry.get("record_version")
            if entry.get("paused"):
                paused = True
            action = entry.get("resume_action")
            if isinstance(action, dict):
                act = action.get("action")
                if act in ("confirm", "cancel"):
                    resolution = act
                elif "fact" in action:
                    resolution = "answer"
    return before_version, paused, resolution


def _write_summary(path: Path, results: Sequence[ScenarioResult], errors: Sequence[Tuple[str, str]], *,
                     mode: str, inference: str, output_dir: Path) -> None:
    lines = [
        "# DermPact prototype -- run-all summary", "", f"generated_at: {now_iso()}",
        f"mode: `{mode}` | inference: `{inference}`", "",
        "| scenario | case_id | mode | source_label | terminal_state | interrupt | resolved_via | "
        "record_version (before -> after) | gate_disposition | record_sha256 | run_ids |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        before_version, paused, resolution = _scenario_pause_summary(output_dir, r.run_ids)
        last_report = _read_json_safe(Path(r.report_paths["report.json"])) if r.report_paths.get("report.json") else {}
        gate_disposition = (last_report.get("gate") or {}).get("disposition")
        mode_label = last_report.get("mode")
        source_label = (last_report.get("identity") or {}).get("source_label")
        lines.append(
            f"| {r.scenario} | `{r.case_id}` | `{mode_label}` | `{source_label}` | `{r.terminal_state}` | "
            f"{'yes' if paused else 'no'} | {resolution or '-'} | {before_version} -> {r.record_version} | "
            f"`{gate_disposition}` | `{(r.record_sha256 or '')[:16]}` | {', '.join(r.run_ids)} |"
        )
    lines.append("")
    for r in results:
        if r.notes:
            lines.append(f"## {r.scenario}")
            lines.append("")
            for n in r.notes:
                lines.append(f"- {n}")
            lines.append("")
    if errors:
        lines.append("## Errors")
        lines.append("")
        for name, detail in errors:
            lines.append(f"- **{name}**: {detail}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "SCENARIO_NAMES", "SCENARIOS", "ScenarioError", "ScenarioResult", "build_host", "default_case_id",
    "opening_request", "case_comparison", "checklist_amendment", "clarification", "clinical_photo",
    "evidence_report", "explain_deferral", "out_of_scope", "region_sensitivity", "resume_after_restart", "run_all",
]
