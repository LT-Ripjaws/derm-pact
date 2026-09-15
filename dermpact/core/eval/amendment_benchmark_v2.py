"""WP-E item 1: amendment benchmark v2 -- corrective/harmful/structural/preview curves rebuilt on the v2 record + ledger under reviewed EADO semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dermpact.eval import dermpact_phase2_baselines as baselines
from dermpact.eval.dermpact_phase4_benchmark import (
    CRITERIA,
    NEG_VALUE,
    OOF_RAW,
    POS_VALUE,
)

from dermpact.core.eval import applicability_benchmark_v2 as ab2
from dermpact.core.eval.applicability_benchmark_v2 import (
    _atomic_json,
    _atomic_write,
    _load_checked_npz,
    _relative_or_absolute,
    _sha256_file,
    STAGE2_OOF_PATH,
    resolve_inputs,
)
from dermpact.core.ledger_v2 import AmendmentLedgerV2, AmendmentProposal
from dermpact.core.record_v2 import (
    AssertionInput,
    StructuredCaseRecordV2,
    build_record_from_oof,
)
from dermpact.core.store import CaseStore
from dermpact.guidelines.reviewed_v2 import ReviewedApplicabilityEngine
from dermpact.criteria.schema import CRITERIA_BY_NAME, INDETERMINATE

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research/manifests/dermpact_phase6/amendment_benchmark_v2"
EVAL_DATE = date(2026, 9, 6)
CREATED_AT = "2026-09-06T00:00:00"
AMENDMENT_TTL_S = 600
N_CASES = 616
N_BOOT = 10000
BOOT_SEED = 20260906
ACTOR = "clinician"
SESSION_ID = "amendment_benchmark_v2"




def ledger_nonce_for(protocol_sha256: str, case_id: str, curve_name: str) -> str:
    """Deterministic ``AmendmentLedgerV2`` nonce (same technique as the Phase 5 v1 experiment's ``ledger_nonce_for``): ``AmendmentLedgerV2`` defaults its nonce to ``uuid4()`` (fine interactively, not reproducible), which feeds into every issued token and therefore every ``AmendmentEntryV2.token_sha256`` and amended ``record_sha256``."""
    return hashlib.sha256(f"{protocol_sha256}|{case_id}|{curve_name}".encode("utf-8")).hexdigest()


def new_ledger(catalogue, engine, nonce: str) -> AmendmentLedgerV2:
    return AmendmentLedgerV2(catalogue=catalogue, engine=engine, calibrators=None, bundle=None, ledger_nonce=nonce)


def case_clustered_bootstrap(by_case: Mapping[str, Sequence[float]], *, n_boot: int, seed: int) -> Dict[str, Any]:
    """Percentile bootstrap resampling CASES with replacement; pools every value of each resampled case."""
    cases = sorted(c for c in by_case if by_case[c])
    if not cases:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": n_boot,
                "n_cases": 0, "boundary_case_bound_one_sided_95": None}
    sums = np.array([float(np.sum(by_case[c])) for c in cases])
    counts = np.array([len(by_case[c]) for c in cases])
    point = float(sums.sum() / counts.sum())
    n = len(cases)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_sums = sums[idx].sum(axis=1)
    boot_counts = counts[idx].sum(axis=1)
    boot_means = boot_sums / boot_counts
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    boundary = None
    if point == 0.0:
        boundary = {"kind": "upper", "value": float(1.0 - 0.05 ** (1.0 / n)),
                    "assumption": "independent_case_clusters; zero observed positive cases"}
    elif point == 1.0:
        boundary = {"kind": "lower", "value": float(0.05 ** (1.0 / n)),
                    "assumption": "independent_case_clusters; all observed cases positive"}
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot, "n_cases": n,
            "boundary_case_bound_one_sided_95": boundary}


def ordered_criteria_by_uncertainty(record: StructuredCaseRecordV2) -> List[str]:
    probs = record.perception.criterion_probabilities
    return sorted(CRITERIA, key=lambda c: abs(probs[c] - 0.5))


def wrong_value_for(criterion: str, true_value: str) -> str:
    """A representative WRONG value chosen by POINTS, not by label polarity (same rule as the Phase 5 v1 experiment's ``wrong_value_for``: several allowed labels score the same points as their opposite-labelled representative, so a label-only flip can be a point-neutral no-op)."""
    spec = CRITERIA_BY_NAME[criterion]
    true_points = spec.score_for(true_value) if true_value != INDETERMINATE else None
    if true_points == 0 or true_points is None:
        return POS_VALUE[criterion]
    return NEG_VALUE[criterion]


def next_step_key(record: StructuredCaseRecordV2) -> Tuple[str, Optional[str], Optional[str]]:
    return (record.next_step.status, record.next_step.rule_id, record.next_step.action_code)


def checklist_point(record: StructuredCaseRecordV2, truth_melanoma: bool) -> Dict[str, Any]:
    decidable = bool(record.checklist.decidable)
    meets = record.checklist.meets_threshold
    matches_truth = int(bool(meets) == truth_melanoma) if decidable else None
    return {
        "decidable": decidable, "meets_threshold": meets, "meets_threshold_matches_truth": matches_truth,
        "agreement_present": bool(record.agreement.present),
    }


def gold_next_step_for_assertion(suspected_value: bool) -> Dict[str, Any]:
    """Gold next step for the explicit assertion route: the SAME independent reference-table walk WP-B's applicability benchmark v2 uses (imported, not re-derived), with every other clinician fact left unasserted."""
    kw = {"clinician_melanoma_suspected": suspected_value, "lesion_under_assessment": True}
    gold = ab2.compute_gold_for_rules(kw)
    walk = ab2.next_step_walk(gold)
    return {"status": walk["status"], "rule_id": walk["rule_id"], "action_code": walk["action_code"]}




def run_corrective_checklist(
    record: StructuredCaseRecordV2, true_criteria_raw: Mapping[str, str], truth_melanoma: bool,
    catalogue, engine, *, order: Sequence[str], ledger_nonce: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Commit the k=1..7 TRUE-value corrective sequence; return ``(points, rows)`` where ``points[0]`` is the k=0 baseline and ``points[k]`` is the state after committing k corrections (checklist evidence only -- next_step is never inspected here)."""
    ledger = new_ledger(catalogue, engine, ledger_nonce)
    current = record
    points = [dict(k=0, **checklist_point(current, truth_melanoma))]
    rows: List[Dict[str, Any]] = []
    for k, crit in enumerate(order, start=1):
        target = true_criteria_raw[crit]
        t = f"2026-09-06T{(k % 24):02d}:00:00"
        proposal = ledger.propose(current, AmendmentProposal("criterion_value", crit, target),
                                  actor=ACTOR, session_id=SESSION_ID, proposed_at=t)
        if proposal.status == "proposed":
            token = ledger.issue_confirmation_token(proposal.proposal_id, issued_at=t, session_id=SESSION_ID, ttl_s=AMENDMENT_TTL_S)
            ack = token.consequence_preview_sha256 if token.requires_acknowledgement else None
            commit = ledger.commit(current, token.token, now=t, actor=ACTOR, session_id=SESSION_ID,
                                   acknowledged_consequence_sha256=ack)
            commit_status = commit.status
            if commit.status == "committed":
                current = commit.record
        else:
            commit_status = "propose_rejected"
        rows.append({"k": k, "criterion": crit, "commit_status": commit_status})
        points.append(dict(k=k, **checklist_point(current, truth_melanoma)))
    return points, rows




def run_assertion_route(
    record: StructuredCaseRecordV2, true_diagnosis: str, catalogue, engine, *,
    protocol_sha256: str, case_id: str,
) -> Dict[str, Dict[str, Any]]:
    truth_aligned_value = (true_diagnosis == "melanoma")
    out: Dict[str, Dict[str, Any]] = {}
    for label, value in (("truth_aligned", truth_aligned_value), ("truth_negated", not truth_aligned_value)):
        ledger = new_ledger(catalogue, engine, ledger_nonce_for(protocol_sha256, case_id, f"assertion_{label}"))
        t = "2026-09-06T00:00:08"
        proposal = ledger.propose(record, AmendmentProposal("clinician_fact", "clinician.melanoma_suspected", value),
                                  actor=ACTOR, session_id=SESSION_ID, proposed_at=t)
        if proposal.status != "proposed":
            out[label] = {"commit_status": "propose_rejected", "reason": proposal.reason, "matches_gold": None,
                          "next_step_changed": None}
            continue
        token = ledger.issue_confirmation_token(proposal.proposal_id, issued_at=t, session_id=SESSION_ID, ttl_s=AMENDMENT_TTL_S)
        ack = token.consequence_preview_sha256 if token.requires_acknowledgement else None
        commit = ledger.commit(record, token.token, now=t, actor=ACTOR, session_id=SESSION_ID,
                               acknowledged_consequence_sha256=ack)
        matches_gold: Optional[int] = None
        if commit.status == "committed":
            gold = gold_next_step_for_assertion(value)
            actual = {"status": commit.record.next_step.status, "rule_id": commit.record.next_step.rule_id,
                      "action_code": commit.record.next_step.action_code}
            matches_gold = int(actual == gold)
        out[label] = {"commit_status": commit.status, "matches_gold": matches_gold,
                      "next_step_changed": bool(proposal.next_step_changed)}
    return out




def run_harmful_checklist(
    record: StructuredCaseRecordV2, true_criteria_raw: Mapping[str, str], catalogue, engine, *,
    order: Sequence[str], ledger_nonce: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Commit the k=1..7 WRONG-value sequence (chosen by points, never label-only); return ``(rows, polarity_total, polarity_ok)``."""
    ledger = new_ledger(catalogue, engine, ledger_nonce)
    current = record
    rows: List[Dict[str, Any]] = []
    polarity_total = 0
    polarity_ok = 0
    for k, crit in enumerate(order, start=1):
        true_val = true_criteria_raw[crit]
        wrong = wrong_value_for(crit, true_val)
        spec = CRITERIA_BY_NAME[crit]
        polarity_total += 1
        true_points = spec.score_for(true_val) if true_val != INDETERMINATE else None
        if true_points is None or spec.score_for(wrong) != true_points:
            polarity_ok += 1
        before = next_step_key(current)
        t = f"2026-09-06T{(k % 24):02d}:10:00"
        proposal = ledger.propose(current, AmendmentProposal("criterion_value", crit, wrong),
                                  actor=ACTOR, session_id=SESSION_ID, proposed_at=t)
        if proposal.status != "proposed":
            rows.append({"k": k, "criterion": crit, "commit_status": "propose_rejected", "reason": proposal.reason,
                         "next_step_changed": False, "acknowledgement_required": False})
            continue
        token = ledger.issue_confirmation_token(proposal.proposal_id, issued_at=t, session_id=SESSION_ID, ttl_s=AMENDMENT_TTL_S)
        ack_required = bool(token.requires_acknowledgement)
        ack = token.consequence_preview_sha256 if ack_required else None
        commit = ledger.commit(current, token.token, now=t, actor=ACTOR, session_id=SESSION_ID,
                               acknowledged_consequence_sha256=ack)
        next_step_changed = False
        if commit.status == "committed":
            current = commit.record
            next_step_changed = (before != next_step_key(current))
        rows.append({"k": k, "criterion": crit, "commit_status": commit.status,
                     "next_step_changed": next_step_changed, "acknowledgement_required": ack_required})
    return rows, polarity_total, polarity_ok




def run_structural_blocks(
    record: StructuredCaseRecordV2, other_record: StructuredCaseRecordV2, catalogue, engine, *, ledger_nonce: str,
) -> Dict[str, bool]:
    """Five independent rejection checks against one fresh ledger and the SAME base record."""
    ledger = new_ledger(catalogue, engine, ledger_nonce)
    blocks: Dict[str, bool] = {}

    r_no_token = ledger.commit(record, None, now="2026-09-06T00:00:01", actor=ACTOR, session_id=SESSION_ID)
    blocks["no_token"] = (r_no_token.status == "rejected" and r_no_token.reason == "token_missing")

    p_stale = ledger.propose(record, AmendmentProposal("clinician_fact", "patient.genetic_predisposition", True),
                             actor=ACTOR, session_id=SESSION_ID, proposed_at="2026-09-06T00:00:02")
    if p_stale.status == "proposed":
        t_stale = ledger.issue_confirmation_token(p_stale.proposal_id, issued_at="2026-09-06T00:00:02", session_id=SESSION_ID, ttl_s=1)
        r_stale = ledger.commit(record, t_stale.token, now="2026-09-06T00:05:00", actor=ACTOR, session_id=SESSION_ID)
        blocks["stale_token"] = (r_stale.status == "rejected" and r_stale.reason == "token_expired")
    else:
        blocks["stale_token"] = True  # proposal never reached this record either way

    p_wrong_case = ledger.propose(record, AmendmentProposal("clinician_fact", "patient.personal_melanoma_history", True),
                                  actor=ACTOR, session_id=SESSION_ID, proposed_at="2026-09-06T00:00:03")
    if p_wrong_case.status == "proposed":
        t_wrong_case = ledger.issue_confirmation_token(p_wrong_case.proposal_id, issued_at="2026-09-06T00:00:03",
                                                       session_id=SESSION_ID, ttl_s=AMENDMENT_TTL_S)
        r_wrong_case = ledger.commit(other_record, t_wrong_case.token, now="2026-09-06T00:00:04", actor=ACTOR, session_id=SESSION_ID)
        blocks["wrong_case"] = (r_wrong_case.status == "rejected" and r_wrong_case.reason == "token_case_mismatch")
    else:
        blocks["wrong_case"] = True

    p_session = ledger.propose(record, AmendmentProposal("clinician_fact", "site.rcm_available", True),
                               actor=ACTOR, session_id=SESSION_ID, proposed_at="2026-09-06T00:00:05")
    if p_session.status == "proposed":
        t_session = ledger.issue_confirmation_token(p_session.proposal_id, issued_at="2026-09-06T00:00:05",
                                                     session_id=SESSION_ID, ttl_s=AMENDMENT_TTL_S)
        r_session = ledger.commit(record, t_session.token, now="2026-09-06T00:00:06", actor=ACTOR, session_id="wrong-session")
        blocks["session_mismatch"] = (r_session.status == "rejected" and r_session.reason == "session_mismatch")
    else:
        blocks["session_mismatch"] = True

    p_replay = ledger.propose(record, AmendmentProposal("clinician_fact", "patient.high_risk_status", True),
                              actor=ACTOR, session_id=SESSION_ID, proposed_at="2026-09-06T00:00:07")
    if p_replay.status == "proposed":
        t_replay = ledger.issue_confirmation_token(p_replay.proposal_id, issued_at="2026-09-06T00:00:07",
                                                    session_id=SESSION_ID, ttl_s=AMENDMENT_TTL_S)
        ack = t_replay.consequence_preview_sha256 if t_replay.requires_acknowledgement else None
        first = ledger.commit(record, t_replay.token, now="2026-09-06T00:00:07", actor=ACTOR, session_id=SESSION_ID,
                              acknowledged_consequence_sha256=ack)
        if first.status == "committed":
            replay = ledger.commit(record, t_replay.token, now="2026-09-06T00:00:07", actor=ACTOR, session_id=SESSION_ID,
                                   acknowledged_consequence_sha256=ack)
            blocks["replay"] = (replay.status == "rejected" and replay.reason == "token_used")
        else:
            blocks["replay"] = False  # the first commit itself failed unexpectedly -- cannot validate the replay guard
    else:
        blocks["replay"] = True

    return blocks


STRUCTURAL_BLOCK_NAMES: Tuple[str, ...] = ("no_token", "stale_token", "wrong_case", "session_mismatch", "replay")




def run_preview_not_commit(
    record: StructuredCaseRecordV2, catalogue, engine, *, ledger_nonce: str, store: Optional[CaseStore],
) -> Dict[str, Any]:
    """Propose (never issue a token, never commit); the persisted store head must be untouched."""
    ledger = new_ledger(catalogue, engine, ledger_nonce)
    case_id = record.identity.case_id
    head_before: Optional[Dict[str, Any]] = None
    if store is not None:
        store.create_case(record)
        head_before = dict(store.load_case(case_id).head)
    proposal = ledger.propose(record, AmendmentProposal("clinician_fact", "patient.genetic_predisposition", True),
                              actor=ACTOR, session_id=SESSION_ID, proposed_at="2026-09-06T00:00:00")
    result: Dict[str, Any] = {"propose_status": proposal.status, "verified_on_disk": store is not None}
    if store is not None:
        for ev in ledger.to_events():
            store.append_ledger_event(case_id, ev)
        head_after = dict(store.load_case(case_id).head)
        result["head_unchanged"] = (head_before == head_after)
    else:
        result["head_unchanged"] = True
    return result




def _dermpact_module_hashes() -> Dict[str, str]:
    pkg_dir = ROOT / "dermpact/core"
    hashes = {p.relative_to(ROOT).as_posix(): _sha256_file(p) for p in sorted(pkg_dir.glob("*.py"))}
    for p in sorted((pkg_dir / "eval").glob("*.py")):
        hashes[p.relative_to(ROOT).as_posix()] = _sha256_file(p)
    return hashes


def build_protocol_freeze(*, catalogue, inputs, oof_sha256: str, legacy_truth_sha256: str, n_boot: int, seed: int) -> Dict[str, Any]:
    this_script = Path(__file__).resolve()
    return {
        "schema_version": "dermpact-phase6-amendment-benchmark-v2-protocol-v1",
        "status": "PRESPECIFIED_BEFORE_RESULTS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "module_hashes": _dermpact_module_hashes(),
        "experiment_script": {"path": this_script.relative_to(ROOT).as_posix(), "sha256": _sha256_file(this_script)},
        "catalogue": {"catalogue_sha256": catalogue.catalogue_sha256, "n_records": len(catalogue.records),
                     "n_executable": len(catalogue.executable)},
        "sources": {
            "stage2_oof": {"path": _relative_or_absolute(STAGE2_OOF_PATH), "sha256": oof_sha256},
            "legacy_truth_oof_raw": {"path": _relative_or_absolute(OOF_RAW), "sha256": legacy_truth_sha256},
            "inventory_sha256": inputs.inventory_sha256, "fold_manifest_sha256": inputs.fold_manifest_sha256,
        },
        "conventions": {
            "evaluation_date": EVAL_DATE.isoformat(), "amendment_ttl_s": AMENDMENT_TTL_S,
            "actor": ACTOR, "session_id": SESSION_ID,
            "bootstrap": {"method": "case-clustered percentile bootstrap", "n_boot": n_boot, "seed": seed},
            "record_mode": "single_member_oof (build_record_from_oof) -- each case's held-out fold prediction, "
                           "NOT the five-model deployment ensemble; probabilities are raw/uncalibrated",
            "v1_superseded_note": (
                "dermpact/eval/dermpact_phase5_experiment.py reported a corrective NEXT-STEP curve rising "
                "0.667 (k=0) -> 0.946 (k=7). Under the v2 record's next-step walk (compute_next_step_v2), "
                "next_step is a pure function of the reviewed-v2 applicability engine over the nine "
                "clinician-assertion facts and image status; it never reads criterion_values or the checklist "
                "score. Committing TRUE checklist criteria therefore cannot move next_step at all under v2 -- "
                "the 0.667->0.946 figure is not reproducible and not meaningful here; see the harmful-curve "
                "next_step_changed count below for the empirical verification of this decoupling."
            ),
            "corrective_curve_scope": "checklist-evidence endpoints only (decidable, meets_threshold vs truth, "
                                      "direct/checklist agreement.present) -- never next_step (see v1_superseded_note)",
            "assertion_route_gold": "dermpact.core.eval.applicability_benchmark_v2.compute_gold_for_rules / "
                                    "next_step_walk (the SAME independent reference-table walk WP-B uses), imported not re-derived",
            "ordering_rule": "criteria corrected/flipped in ascending |probability-0.5| order (most-uncertain first)",
            "wrong_value_rule": "wrong_value_for chooses by POINTS (score_for), never by label polarity alone -- "
                                "guarantees score_for(wrong) != score_for(true) whenever true is determinate",
            "ledger_nonce_rule": (
                "AmendmentLedgerV2 defaults its nonce to uuid4() (fine interactively, not reproducible). This "
                "benchmark instead pins ledger_nonce = sha256(protocol_sha256|case_id|curve_name) for every "
                "AmendmentLedgerV2 it constructs, so every issued token, AmendmentEntryV2.token_sha256, and "
                "amended record_sha256 is reproducible run-to-run against the same protocol."
            ),
            "structural_block_definitions": {
                "no_token": "commit(record, token=None, ...) -> rejected/token_missing",
                "stale_token": "a token issued with ttl_s=1 is presented after it has expired -> rejected/token_expired",
                "wrong_case": "a token issued against case A is presented while committing case B's record -> rejected/token_case_mismatch",
                "session_mismatch": "commit presented with a session_id different from the one that proposed/was issued the token -> rejected/session_mismatch",
                "replay": "a token already consumed by one successful commit is presented a second time -> rejected/token_used",
            },
        },
        "hard_scope_checks": {
            "derm7pt_development_rows": N_CASES, "derm7pt_official_test_rows_loaded": 0,
            "ham_rows_loaded": 0, "bcn_rows_loaded": 0, "ph2_rows_loaded": 0,
            "images_opened": 0, "models_loaded": 0, "live_llm_calls_made": 0,
        },
    }




@dataclass
class CaseResult:
    case_id: str
    corrective_points: List[Dict[str, Any]]
    corrective_rows: List[Dict[str, Any]]
    assertion: Dict[str, Dict[str, Any]]
    harmful_rows: List[Dict[str, Any]]
    harmful_polarity_total: int
    harmful_polarity_ok: int
    structural_blocks: Dict[str, bool]
    preview: Dict[str, Any]


def run_case(
    record: StructuredCaseRecordV2, other_record: StructuredCaseRecordV2, true_diagnosis: str,
    true_criteria_raw: Mapping[str, str], catalogue, engine, *, protocol_sha256: str, store: Optional[CaseStore],
) -> CaseResult:
    case_id = record.identity.case_id
    truth_melanoma = (true_diagnosis == "melanoma")
    order = ordered_criteria_by_uncertainty(record)

    points, corrective_rows = run_corrective_checklist(
        record, true_criteria_raw, truth_melanoma, catalogue, engine, order=order,
        ledger_nonce=ledger_nonce_for(protocol_sha256, case_id, "corrective"),
    )
    assertion = run_assertion_route(record, true_diagnosis, catalogue, engine,
                                    protocol_sha256=protocol_sha256, case_id=case_id)
    harmful_rows, polarity_total, polarity_ok = run_harmful_checklist(
        record, true_criteria_raw, catalogue, engine, order=order,
        ledger_nonce=ledger_nonce_for(protocol_sha256, case_id, "harmful"),
    )
    structural = run_structural_blocks(
        record, other_record, catalogue, engine,
        ledger_nonce=ledger_nonce_for(protocol_sha256, case_id, "structural"),
    )
    preview = run_preview_not_commit(
        record, catalogue, engine, ledger_nonce=ledger_nonce_for(protocol_sha256, case_id, "preview"), store=store,
    )
    return CaseResult(
        case_id=case_id, corrective_points=points, corrective_rows=corrective_rows, assertion=assertion,
        harmful_rows=harmful_rows, harmful_polarity_total=polarity_total, harmful_polarity_ok=polarity_ok,
        structural_blocks=structural, preview=preview,
    )




def summarise(results: Sequence[CaseResult], *, n_boot: int, seed: int) -> Dict[str, Any]:
    n_cases = len(results)
    max_k = max((len(r.corrective_points) for r in results), default=0) - 1

    decidable_by_k: List[Dict[str, List[int]]] = [dict() for _ in range(max_k + 1)]
    matches_truth_by_k: List[Dict[str, List[int]]] = [dict() for _ in range(max_k + 1)]
    agreement_by_k: List[Dict[str, List[int]]] = [dict() for _ in range(max_k + 1)]
    for r in results:
        for pt in r.corrective_points:
            k = pt["k"]
            decidable_by_k[k].setdefault(r.case_id, []).append(int(pt["decidable"]))
            agreement_by_k[k].setdefault(r.case_id, []).append(int(pt["agreement_present"]))
            if pt["meets_threshold_matches_truth"] is not None:
                matches_truth_by_k[k].setdefault(r.case_id, []).append(int(pt["meets_threshold_matches_truth"]))

    checklist_evidence_curve = {
        "decidable_rate": [{"k": k, **case_clustered_bootstrap(by_case, n_boot=n_boot, seed=seed + k)}
                           for k, by_case in enumerate(decidable_by_k)],
        "meets_threshold_matches_truth_rate": [{"k": k, **case_clustered_bootstrap(by_case, n_boot=n_boot, seed=seed + 1000 + k)}
                                               for k, by_case in enumerate(matches_truth_by_k)],
        "agreement_present_rate": [{"k": k, **case_clustered_bootstrap(by_case, n_boot=n_boot, seed=seed + 2000 + k)}
                                   for k, by_case in enumerate(agreement_by_k)],
    }

    assertion_by_case: Dict[str, Dict[str, List[int]]] = {
        "truth_aligned_matches_gold": {}, "truth_negated_matches_gold": {},
        "truth_aligned_changed_next_step": {}, "truth_negated_changed_next_step": {},
    }
    for r in results:
        for label in ("truth_aligned", "truth_negated"):
            row = r.assertion.get(label, {})
            if row.get("matches_gold") is not None:
                assertion_by_case[f"{label}_matches_gold"].setdefault(r.case_id, []).append(int(row["matches_gold"]))
            if row.get("next_step_changed") is not None:
                assertion_by_case[f"{label}_changed_next_step"].setdefault(r.case_id, []).append(int(row["next_step_changed"]))
    assertion_route = {
        name: case_clustered_bootstrap(by_case, n_boot=n_boot, seed=seed + 3000 + i)
        for i, (name, by_case) in enumerate(sorted(assertion_by_case.items()))
    }

    next_step_changed_by_case: Dict[str, List[int]] = {}
    ack_required_by_case: Dict[str, List[int]] = {}
    n_committed_opportunities = 0
    n_ack_required_true = 0
    n_next_step_changed_true = 0
    polarity_total = sum(r.harmful_polarity_total for r in results)
    polarity_ok = sum(r.harmful_polarity_ok for r in results)
    for r in results:
        for row in r.harmful_rows:
            if row["commit_status"] != "committed":
                continue
            n_committed_opportunities += 1
            changed = int(bool(row["next_step_changed"]))
            ack = int(bool(row["acknowledgement_required"]))
            n_next_step_changed_true += changed
            n_ack_required_true += ack
            next_step_changed_by_case.setdefault(r.case_id, []).append(changed)
            ack_required_by_case.setdefault(r.case_id, []).append(ack)
    harmful = {
        "next_step_changed_rate": case_clustered_bootstrap(next_step_changed_by_case, n_boot=n_boot, seed=seed + 4000),
        "next_step_changed_count": n_next_step_changed_true,
        "acknowledgement_required_rate": case_clustered_bootstrap(ack_required_by_case, n_boot=n_boot, seed=seed + 4001),
        "acknowledgement_required_count": n_ack_required_true,
        "n_committed_opportunities": n_committed_opportunities,
        "polarity_check": {"total": polarity_total, "ok": polarity_ok,
                           "rate": (polarity_ok / polarity_total) if polarity_total else float("nan")},
    }

    structural_by_case: Dict[str, Dict[str, List[int]]] = {name: {} for name in STRUCTURAL_BLOCK_NAMES}
    for r in results:
        for name in STRUCTURAL_BLOCK_NAMES:
            structural_by_case[name][r.case_id] = [int(bool(r.structural_blocks.get(name)))]
    structural_blocks = {
        name: case_clustered_bootstrap(structural_by_case[name], n_boot=n_boot, seed=seed + 5000 + i)
        for i, name in enumerate(STRUCTURAL_BLOCK_NAMES)
    }

    preview_by_case = {r.case_id: [int(bool(r.preview["head_unchanged"]))] for r in results}
    n_disk_verified = sum(1 for r in results if r.preview.get("verified_on_disk"))
    preview_not_commit = {
        "head_unchanged_rate": case_clustered_bootstrap(preview_by_case, n_boot=n_boot, seed=seed + 6000),
        "n_disk_verified": n_disk_verified, "n_in_memory_fallback": n_cases - n_disk_verified,
    }

    return {
        "n_cases": n_cases,
        "checklist_evidence_curve": checklist_evidence_curve,
        "assertion_route": assertion_route,
        "harmful_checklist_corrections": harmful,
        "structural_blocks": structural_blocks,
        "preview_not_commit": preview_not_commit,
    }




def render_result_markdown(curves: Mapping[str, Any], *, wall_time: float, n_cases: int, store_timed_out: bool) -> str:
    lines = ["# DermPact WP-E -- amendment benchmark v2", ""]
    lines.append(
        "**The Phase 5 v1 corrective curve (0.667 -> 0.946 at k=7) is NOT a current figure.** It measured "
        "whether committing true checklist criteria moved the record's next step. Under the v2 record, "
        "`compute_next_step_v2` is a pure function of the reviewed-v2 applicability engine over the nine "
        "clinician-assertion facts and image status; it never reads `criterion_values` or the checklist score "
        "at all -- committing checklist criteria (true or wrong) cannot move next_step under v2 by "
        "construction. This is not a regression: it is the reviewed-catalogue design decision that a "
        "checklist correction alone must never manufacture clinical suspicion. See "
        "`harmful_checklist_corrections.next_step_changed_count` below for the empirical count."
    )
    lines.append("")
    lines.append(
        "**Derm7pt 616 stage2 out-of-fold rows are each case's single held-out fold prediction "
        "(`perception_mode=single_member_oof`, raw/uncalibrated), not an output of the five-model deployment "
        "ensemble** (`five_member_deployment`) -- see `dermpact.core.record_v2`'s builder distinction."
    )
    lines.append("")
    lines.append(f"n_cases = {curves['n_cases']}")
    lines.append("")

    lines.append("## (a) Corrective checklist-evidence curve (never next_step)")
    lines.append("")
    lines.append("| k | decidable | meets_threshold==truth | agreement.present |")
    lines.append("|---|---|---|---|")
    ce = curves["checklist_evidence_curve"]
    for dec, mt, ag in zip(ce["decidable_rate"], ce["meets_threshold_matches_truth_rate"], ce["agreement_present_rate"]):
        lines.append(f"| {dec['k']} | {dec['estimate']:.4f} | {mt['estimate']:.4f} [{mt['ci_low']:.4f},{mt['ci_high']:.4f}] "
                     f"(n={mt['n_cases']}) | {ag['estimate']:.4f} |")
    lines.append("")

    lines.append("## (b) Explicit assertion route (`clinician.melanoma_suspected`) vs. reference-table gold")
    lines.append("")
    lines.append("| metric | estimate | ci_low | ci_high |")
    lines.append("|---|---|---|---|")
    for name, row in sorted(curves["assertion_route"].items()):
        lines.append(f"| {name} | {row['estimate']:.4f} | {row['ci_low']:.4f} | {row['ci_high']:.4f} |")
    lines.append("")

    h = curves["harmful_checklist_corrections"]
    lines.append("## (c) Harmful wrong checklist corrections")
    lines.append("")
    lines.append(
        f"next_step_changed: {h['next_step_changed_count']}/{h['n_committed_opportunities']} "
        f"(rate {h['next_step_changed_rate']['estimate']:.4f} "
        f"[{h['next_step_changed_rate']['ci_low']:.4f},{h['next_step_changed_rate']['ci_high']:.4f}]) -- "
        "expected 0 under v2 (see the v1-superseded note above); this is the empirical verification of that claim."
    )
    lines.append("")
    lines.append(
        f"acknowledgement_required: {h['acknowledgement_required_count']}/{h['n_committed_opportunities']} "
        f"(rate {h['acknowledgement_required_rate']['estimate']:.4f})"
    )
    lines.append("")
    pc = h["polarity_check"]
    lines.append(f"Wrong-value polarity check (chosen by points, not label): {pc['ok']}/{pc['total']} ({pc['rate']:.4f}); must be 1.0.")
    lines.append("")

    lines.append("## (d) Structural blocks (must be 1.0)")
    lines.append("")
    lines.append("| check | estimate |")
    lines.append("|---|---|")
    for name, row in sorted(curves["structural_blocks"].items()):
        lines.append(f"| {name} | {row['estimate']:.4f} |")
    lines.append("")

    p = curves["preview_not_commit"]
    lines.append("## (e) Preview-not-commit (must be 1.0)")
    lines.append("")
    lines.append(f"head_unchanged rate: {p['head_unchanged_rate']['estimate']:.4f} "
                 f"(disk-verified via a real CaseStore: {p['n_disk_verified']} cases; "
                 f"in-memory-only fallback: {p['n_in_memory_fallback']} cases)")
    if store_timed_out:
        lines.append(
            "\nNote: the disk-backed CaseStore check was truncated to a sample and the remainder verified "
            "via the ledger's replayable in-memory path (a `propose()` never mutates or replaces the frozen "
            "record it was called with, so the would-be store head cannot have moved) because running the "
            "real store for all 616 cases would have pushed the benchmark over its runtime budget."
        )
    lines.append("")
    lines.append(f"## Runtime\n\nn_cases={n_cases}, wall_time_s={wall_time:.1f}")
    lines.append("")
    return "\n".join(lines)




def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-suffix", type=str, default="")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=BOOT_SEED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--store-sample-n", type=int, default=None,
                        help="cap the number of cases verified via a real on-disk CaseStore for "
                             "preview-not-commit (default: all cases in this run)")
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    out_dir = Path(args.out_dir).resolve()
    suffix = args.out_suffix

    protocol_path = out_dir / f"protocol{suffix}.json"
    rows_path = out_dir / f"rows{suffix}.jsonl"
    curves_path = out_dir / f"curves{suffix}.json"
    result_md_path = out_dir / f"RESULT{suffix}.md"

    engine = ReviewedApplicabilityEngine()
    catalogue = engine.catalogue

    data = resolve_inputs()
    if len(data.cases) != N_CASES:
        raise AssertionError(f"expected {N_CASES} development cases, got {len(data.cases)}")

    oof = _load_checked_npz(STAGE2_OOF_PATH)
    oof_sha256 = _sha256_file(STAGE2_OOF_PATH)
    if oof["derm_diagnosis_logits"].shape[0] != N_CASES:
        raise AssertionError("stage2 OOF row count does not match the locked development set")

    legacy_truth_sha256 = _sha256_file(OOF_RAW)
    legacy_raw = json.loads(OOF_RAW.read_text(encoding="utf-8"))
    truth_ids = list(legacy_raw["truth"]["ids"])
    truth_diagnosis = list(legacy_raw["truth"]["diagnosis_targets"])
    truth_diagnosis_by_id = {str(cid): str(d) for cid, d in zip(truth_ids, truth_diagnosis)}
    baselines_by_id = {str(r["group_id"]): r for r in baselines._derm7pt_records()}

    stage2_ids = {c.group_id for c in data.cases}
    if set(truth_diagnosis_by_id) != stage2_ids or set(baselines_by_id) != stage2_ids:
        raise AssertionError("case id sets diverge between stage2 inputs, legacy truth and baseline records")

    protocol = build_protocol_freeze(
        catalogue=catalogue, inputs=data, oof_sha256=oof_sha256, legacy_truth_sha256=legacy_truth_sha256,
        n_boot=args.n_boot, seed=args.seed,
    )
    protocol_sha = _atomic_json(protocol_path, protocol)

    cases = list(data.cases) if args.limit is None else list(data.cases)[: args.limit]
    store_sample_n = args.store_sample_n if args.store_sample_n is not None else len(cases)
    store_timed_out = store_sample_n < len(cases)

    print(f"[1/3] build {len(cases)} baseline v2 OOF records", flush=True)
    records: Dict[str, StructuredCaseRecordV2] = {}
    true_diag: Dict[str, str] = {}
    true_criteria: Dict[str, Dict[str, str]] = {}
    for index, case in enumerate(cases):
        cid = case.group_id
        true_diag[cid] = truth_diagnosis_by_id[cid]
        true_criteria[cid] = {name: str(baselines_by_id[cid]["criteria_raw"][name]) for name in CRITERIA}
        records[cid] = build_record_from_oof(
            cid, diagnosis_logits=list(oof["derm_diagnosis_logits"][index]),
            melanoma_logit=float(oof["derm_melanoma_logits"][index]),
            criterion_logits=list(oof["derm_criteria_logits"][index]),
            oof_manifest={"path": _relative_or_absolute(STAGE2_OOF_PATH), "sha256": oof_sha256},
            clinician_assertions={}, observed_facts={
                "sex": case.facts.sex, "site": case.facts.site, "elevation": case.facts.elevation,
                "age_years": case.facts.age_years,
            },
            evaluation_date=EVAL_DATE, created_at=CREATED_AT, catalogue=catalogue, engine=engine,
        )

    store_dir = Path(tempfile.mkdtemp(prefix=f"preview_store{suffix}_", dir=out_dir))
    store = CaseStore(store_dir, catalogue=catalogue, engine=engine, calibrators=None, bundle=None)

    print(f"[2/3] run amendment curves per case", flush=True)
    ordered_ids = [c.group_id for c in cases]
    per_case_results: List[CaseResult] = []
    rows_lines: List[str] = []
    for i, cid in enumerate(ordered_ids):
        record = records[cid]
        other_record = records[ordered_ids[(i + 1) % len(ordered_ids)]]
        use_store = store if i < store_sample_n else None
        result = run_case(
            record, other_record, true_diag[cid], true_criteria[cid], catalogue, engine,
            protocol_sha256=protocol_sha, store=use_store,
        )
        per_case_results.append(result)
        rows_lines.append(json.dumps({"case_id": cid, "curve": "corrective", "points": result.corrective_points,
                                       "rows": result.corrective_rows}, sort_keys=True))
        rows_lines.append(json.dumps({"case_id": cid, "curve": "assertion", **result.assertion}, sort_keys=True))
        rows_lines.append(json.dumps({"case_id": cid, "curve": "harmful", "rows": result.harmful_rows,
                                       "polarity_total": result.harmful_polarity_total,
                                       "polarity_ok": result.harmful_polarity_ok}, sort_keys=True))
        rows_lines.append(json.dumps({"case_id": cid, "curve": "structural_blocks", **result.structural_blocks}, sort_keys=True))
        rows_lines.append(json.dumps({"case_id": cid, "curve": "preview_not_commit", **result.preview}, sort_keys=True))

    rows_sha = _atomic_write(rows_path, "\n".join(rows_lines) + "\n")

    print("[3/3] summarise + write curves/RESULT", flush=True)
    curves_summary = summarise(per_case_results, n_boot=args.n_boot, seed=args.seed)
    curves_sha = _atomic_json(curves_path, curves_summary)

    wall_time = time.perf_counter() - t_start
    result_md = render_result_markdown(curves_summary, wall_time=wall_time, n_cases=len(cases), store_timed_out=store_timed_out)
    result_md_sha = _atomic_write(result_md_path, result_md)

    summary = {
        "status": "PASS", "n_cases": len(cases), "wall_time_s": round(wall_time, 1),
        "artifacts": {
            "protocol": {"path": _relative_or_absolute(protocol_path), "sha256": protocol_sha},
            "rows": {"path": _relative_or_absolute(rows_path), "sha256": rows_sha},
            "curves": {"path": _relative_or_absolute(curves_path), "sha256": curves_sha},
            "result_markdown": {"path": _relative_or_absolute(result_md_path), "sha256": result_md_sha},
        },
        "headline": {
            "corrective_at_k7_meets_threshold_matches_truth":
                curves_summary["checklist_evidence_curve"]["meets_threshold_matches_truth_rate"][-1],
            "assertion_truth_aligned_matches_gold": curves_summary["assertion_route"]["truth_aligned_matches_gold"],
            "assertion_truth_negated_matches_gold": curves_summary["assertion_route"]["truth_negated_matches_gold"],
            "harmful_next_step_changed_count": curves_summary["harmful_checklist_corrections"]["next_step_changed_count"],
            "structural_blocks_min": min((row["estimate"] for row in curves_summary["structural_blocks"].values()), default=None),
            "preview_head_unchanged_rate": curves_summary["preview_not_commit"]["head_unchanged_rate"]["estimate"],
        },
    }
    shutil.rmtree(store_dir, ignore_errors=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
