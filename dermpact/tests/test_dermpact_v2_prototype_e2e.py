"""WP-D: the nine scenarios of the revised interactive-demo brief, end to end through the CLI entry function on a temporary state dir, against WP-X's already-computed cached inventory (``research/experiments/dermpact_phase6/demo_inventory_v1`` -- real five-member model output, no image opened here), plus one ``@pytest.mark.slow`` real-model smoke on ``derm7pt-case-22`` via :class:`~dermpact.core.inference_client.WorkerClient`."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import numpy as np

from dermpact.core import xai
from dermpact.core.inference_client import WorkerClient, WorkerCrashed, WorkerRequestError, WorkerTimeout
from dermpact.core.record_v2 import StructuredCaseRecordV2
from dermpact.core.prototype import cli
from dermpact.core.prototype import graph as graph_mod
from dermpact.core.prototype import inventory as inv
from dermpact.core.prototype import scenarios as scen
from dermpact.core.prototype.host import HostServices

INVENTORY_MANIFEST = inv.DEFAULT_INVENTORY_DIR / "inventory_manifest.json"
REQUIRES_INVENTORY = pytest.mark.skipif(
    not INVENTORY_MANIFEST.is_file(),
    reason=f"WP-X inventory manifest not found at {INVENTORY_MANIFEST}; build it before running the e2e suite",
)
pytestmark = REQUIRES_INVENTORY

FORBIDDEN_SECRET_MARKERS = ("api_key", "bearer ")




def _host(tmp_path: Path, **kwargs: Any) -> HostServices:
    return scen.build_host(tmp_path / "state", inv.DEFAULT_INVENTORY_DIR, **kwargs)


class FakeXaiClient:
    """Wraps a real ``predict``-capable client and adds fake ``xai_region``/``xai_clinical`` methods (see module docstring)."""

    def __init__(self, base: Any, host: HostServices):
        self._base = base
        self._host = host

    def predict(self, **kwargs: Any) -> Dict[str, Any]:
        return self._base.predict(**kwargs)

    def _prediction_for_sha(self, dermoscopic_sha256: str) -> Dict[str, Any]:
        case_ids = self._host.store.list_cases()
        if len(case_ids) == 1:
            return self._base.predict(case_id=case_ids[0])["prediction"]
        for case_id in self._host.store.list_cases():
            record = self._host.store.load_case(case_id).current_record
            if record.inputs.dermoscopic.sanitized_sha256 == dermoscopic_sha256:
                return self._base.predict(case_id=case_id)["prediction"]
        raise AssertionError("fake XAI could not bind the requested image to a stored case")

    def xai_region(self, *, progress_cb=None, expected_dermoscopic_sha256: str, **_ignored: Any) -> Dict[str, Any]:
        if progress_cb is not None:
            progress_cb({"fold": 4, "n_folds": 5, "stage": "region"})
        prediction = self._prediction_for_sha(expected_dermoscopic_sha256)
        no_op = {}
        for fold, raw in enumerate(prediction["raw_fold_outputs"]):
            no_op[str(fold)] = xai._calibrated_member(
                {"diagnosis": np.asarray([raw["diagnosis"]]), "melanoma": np.asarray([raw["melanoma"]]),
                 "criteria": np.asarray([raw["criteria"]])},
                fold, self._host.bundle.calibration,
            )
        return {
            "status": "complete", "n_members": 5, "members_measured": 5, "failures": {},
            "results": [{"region_index": 0, "rank_key_local_mean_melanoma_head_abs_delta": 0.0}],
            "heatmap": {"local_mean": {"melanoma_head": [[0.0]]}},
            "padding_only_regions": [], "no_op_member_raw": no_op,
        }

    def xai_clinical(self, *, progress_cb=None, **_ignored: Any) -> Dict[str, Any]:
        if progress_cb is not None:
            progress_cb({"fold": 4, "n_folds": 5, "stage": "clinical"})
        return {
            "status": "complete", "n_members": 5, "members_measured": 5, "failures": {},
            "targets": {target: {"aggregated_delta": 0.0, "per_member_delta": [0.0] * 5,
                                  "exact_zero": True} for target in xai.TARGETS},
        }

    def __getattr__(self, item: str) -> Any:
        return getattr(self._base, item)


def _install_fake_xai(host: HostServices) -> None:
    host.inference_client = FakeXaiClient(host.inference_client, host)


def _assert_no_secrets_in_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict").lower()
        except (UnicodeDecodeError, ValueError):
            continue  # a binary artefact (e.g. the optional PNG overlay) -- not text to scan
        for marker in FORBIDDEN_SECRET_MARKERS:
            assert marker not in text, f"found forbidden marker {marker!r} in {path}"


def _report(paths: Dict[str, str]) -> Dict[str, Any]:
    return json.loads(Path(paths["report.json"]).read_text(encoding="utf-8"))


def _trace(paths: Dict[str, str]) -> list:
    lines = Path(paths["trace.jsonl"]).read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def _tool_result(turn_result: Dict[str, Any], tool: str) -> Optional[Dict[str, Any]]:
    for entry in reversed(turn_result.get("tool_trace") or []):
        if entry.get("tool") == tool and entry.get("ok"):
            return entry.get("data") or {}
    return None




def test_evidence_report_writes_json_and_markdown_with_gate_and_labels(tmp_path):
    host = _host(tmp_path)
    result = scen.evidence_report(host, output_dir=tmp_path / "out")
    assert result.terminal_state == "answered"
    report = _report(result.report_paths)
    assert report["gate"]["disposition"] in ("display", "partial_report")
    assert report["mode"] == "offline"
    assert report["identity"]["source_label"] == "development_allowlisted_case"
    md = Path(result.report_paths["report.md"]).read_text(encoding="utf-8")
    assert "mode: `offline`" in md
    assert f"source_label: `{report['identity']['source_label']}`" in md
    nodes = [t["node"] for t in _trace(result.report_paths)]
    assert nodes == sorted(nodes, key=nodes.index)  # no duplicates
    assert len(nodes) == len(set(nodes))
    _assert_no_secrets_in_tree(Path(result.report_paths["report.json"]).parent)




def test_explain_deferral_shows_real_thresholds_and_recovery(tmp_path):
    host = _host(tmp_path)
    result = scen.explain_deferral(host, output_dir=tmp_path / "out")
    report = _report(result.report_paths)
    assert report["reliability"]["selective_abstention"] == "deferred"
    thresholds = report["reliability"]["thresholds"]
    assert isinstance(thresholds.get("ood_threshold"), float)
    assert thresholds.get("per_member_acceptance_thresholds")
    case_state = host.store.load_case(result.case_id)
    warnings = case_state.current_record.reliability.warnings
    assert warnings, "a deferred case must carry at least one reliability warning"
    deferral_warnings = [w for w in warnings if w.get("code") == "selective_deferral"]
    assert deferral_warnings, warnings
    assert deferral_warnings[0].get("recovery_action") == "clinician_review"
    turn_text = (report.get("turn_result") or {}).get("text", "")
    assert turn_text, "explain_deferral must produce agent text naming the deferral"
    assert report["turn_result"]["checker_verdict"]["ok"] is True




def test_clarification_pause_confirm_commits_r3(tmp_path):
    """Sub-run A (the ask -> answer -> preview/confirm flow through the ``clarify`` node) commits v1 -> v2; sub-run B (the original direct- assertion path, kept unchanged) commits v2 -> v3 on the same case."""
    host = _host(tmp_path)
    result = scen.clarification(host, output_dir=tmp_path / "out")
    assert result.terminal_state == "committed"
    assert result.record_version == 3
    joined_notes = " | ".join(result.notes)
    assert "question pause: fact=" in joined_notes
    assert "sub-run B" in joined_notes
    case_state = host.store.load_case(result.case_id)
    # sub-run A: v1 -> v2, through clarify's own question/confirm protocol.
    assert case_state.records[0].identity.record_version == 1
    assert case_state.records[1].identity.record_version == 2
    assert case_state.records[0].record_sha256 != case_state.records[1].record_sha256
    assert case_state.records[1].next_step.rule_id == "eado-mel-p1-2024:R3"
    assert case_state.records[1].next_step.status == "recommended"
    # sub-run B: v2 -> v3, through the original supply_assertion path.
    assert case_state.records[2].identity.record_version == 3
    assert case_state.records[1].record_sha256 != case_state.records[2].record_sha256


def test_clarification_resume_wrong_proposal_id_rejected_exit_3(tmp_path, capsys):
    """A resume is a one-shot answer to LangGraph's ``interrupt()`` -- a wrong proposal id does not leave the run paused for a retry, it finalises the run ``rejected`` with the head unchanged."""
    host = _host(tmp_path, mode="offline")
    case_id = scen._case_with_decisive_fact(host)
    out = tmp_path / "out"
    args = ["--state-dir", str(tmp_path / "state"), "--inventory-dir", str(inv.DEFAULT_INVENTORY_DIR)]

    code = cli.main(["run", "--case", case_id, "--scenario", "clarification", "--output", str(out), *args])
    assert code == cli.EXIT_OK
    printed = capsys.readouterr().out
    run_id = next(l.split("=", 1)[1] for l in printed.splitlines() if l.startswith("run_id="))

    code = cli.main(["resume", "--run-id", run_id, "--confirm", "not-a-real-proposal-id", "--output", str(out), *args])
    assert code == cli.EXIT_REJECTED
    assert host.store.load_case(case_id).head["record_version"] == 1


def test_clarification_resume_stale_after_intervening_commit_exit_3(tmp_path, capsys):
    """Two independent proposals against the same v1 head; committing the first moves the head to v2, so resuming the second (minted against v1) must be rejected with exit code 3 and must not move the head again."""
    host = _host(tmp_path, mode="offline")
    case_id = scen._default_commit_safe_case(host)
    out = tmp_path / "out"
    args = ["--state-dir", str(tmp_path / "state"), "--inventory-dir", str(inv.DEFAULT_INVENTORY_DIR)]

    fact1, value1, _ = scen._decisive_or_any_unknown_fact(host, case_id, used=set())
    code = cli.main([
        "run", "--case", case_id, "--scenario", "clarification", "--output", str(out),
        "--request", f"For the record, {scen._FACT_PHRASE[(fact1, value1)]}.", *args,
    ])
    assert code == cli.EXIT_OK
    run_id_1 = next(l.split("=", 1)[1] for l in capsys.readouterr().out.splitlines() if l.startswith("run_id="))
    snap1 = json.loads((out / run_id_1 / "state.json").read_text(encoding="utf-8"))

    fact2, value2, _ = scen._decisive_or_any_unknown_fact(host, case_id, used={fact1})
    code = cli.main([
        "run", "--case", case_id, "--scenario", "clarification", "--output", str(out),
        "--request", f"For the record, {scen._FACT_PHRASE[(fact2, value2)]}.", *args,
    ])
    assert code == cli.EXIT_OK
    run_id_2 = next(l.split("=", 1)[1] for l in capsys.readouterr().out.splitlines() if l.startswith("run_id="))
    snap2 = json.loads((out / run_id_2 / "state.json").read_text(encoding="utf-8"))

    ack1 = snap1["interrupt"].get("preview_sha256") if snap1["interrupt"].get("requires_acknowledgement") else None
    code = cli.main([
        "resume", "--run-id", run_id_1, "--confirm", snap1["interrupt"]["proposal_id"],
        *(["--acknowledge", ack1] if ack1 else []), "--output", str(out), *args,
    ])
    assert code == cli.EXIT_OK
    assert host.store.load_case(case_id).head["record_version"] == 2

    ack2 = snap2["interrupt"].get("preview_sha256") if snap2["interrupt"].get("requires_acknowledgement") else None
    code = cli.main([
        "resume", "--run-id", run_id_2, "--confirm", snap2["interrupt"]["proposal_id"],
        *(["--acknowledge", ack2] if ack2 else []), "--output", str(out), *args,
    ])
    assert code == cli.EXIT_REJECTED
    assert host.store.load_case(case_id).head["record_version"] == 2




def test_checklist_amendment_cancel_then_confirm(tmp_path):
    host = _host(tmp_path)
    result = scen.checklist_amendment(host, output_dir=tmp_path / "out")
    assert result.terminal_state == "committed"
    joined_notes = " | ".join(result.notes)
    assert "model_criterion_values preserved=True" in joined_notes
    assert "clinician_assertions unchanged=True" in joined_notes
    case_state = host.store.load_case(result.case_id)
    final = case_state.current_record
    assert final.perception.criterion_values["pigment_network"] == "atypical"
    assert final.perception.model_criterion_values["pigment_network"] == "typical"
    assert final.checklist.score_min > 0 or final.checklist.score_max > 0




def test_region_sensitivity_without_worker_is_honest_unavailable(tmp_path):
    host = _host(tmp_path)  # default inference="cached"; no xai_region attribute
    result = scen.region_sensitivity(host, output_dir=tmp_path / "out")
    report = _report(result.report_paths)
    turn_text = (report.get("turn_result") or {}).get("text", "") or ""
    xai_data = _tool_result(report["turn_result"], "request_measured_xai") or _tool_result(report["turn_result"], "read_measured_xai")
    assert report.get("xai") is None or report["xai"].get("status") != "complete"
    assert "heatmap" not in turn_text.lower() or "unavailable" in turn_text.lower() or "job" in turn_text.lower()


def test_region_sensitivity_with_fake_runner_stores_verified_receipt(tmp_path):
    host = _host(tmp_path, inference="cached")
    _install_fake_xai(host)
    result = scen.region_sensitivity(host, output_dir=tmp_path / "out")
    case_state = host.store.load_case(result.case_id)
    receipts = host.store.load_xai_receipts(result.case_id)
    assert receipts, "expected at least one persisted XAI receipt with the fake runner installed"
    receipt = receipts[-1]
    assert receipt["status"] == "complete"
    assert receipt["kind"] == "region"
    assert xai.verify_receipt(receipt, case_state.current_record.to_dict()) is True




def test_clinical_photo_present_measured_and_absent_not_applicable(tmp_path):
    host = _host(tmp_path)
    _install_fake_xai(host)
    result = scen.clinical_photo(host, output_dir=tmp_path / "out")
    assert "present_case=" in result.notes[0]
    assert "absent_fixture=" in result.notes[1]

    present_report = _report({"report.json": str(tmp_path / "out" / result.run_ids[0] / "report.json")})
    assert _tool_result(present_report["turn_result"], "read_measured_xai") is not None
    assert "available" in present_report["turn_result"]["text"].lower()

    absent_report = _report({"report.json": str(tmp_path / "out" / result.run_ids[1] / "report.json")})
    absent_text = absent_report["turn_result"]["text"].lower()
    assert "not applicable" in absent_text
    assert "clinical_availability='not_supplied'" in absent_text
    assert not absent_report["turn_result"]["tool_trace"], "the not-applicable path must never call an XAI tool"

    absent_state = host.store.load_case(inv.FIXTURE_CLINICAL_ABSENT)
    assert absent_state.current_record.inputs.clinical_availability == "not_supplied"




def test_case_comparison_matches_no_match_and_dropped_after_revocation(tmp_path):
    host = _host(tmp_path)
    result = scen.case_comparison(host, output_dir=tmp_path / "out")
    joined_notes = " | ".join(result.notes)
    assert "no_match_fixture terminal='answered'" in joined_notes or "no_match_fixture terminal=" in joined_notes

    real_run = result.run_ids[0]
    real_report = _report({"report.json": str(tmp_path / "out" / real_run / "report.json")})
    real_comparison = _tool_result(real_report["turn_result"], "retrieve_case_comparisons")
    assert real_comparison is not None
    assert real_comparison["status"] == "ok"
    assert real_comparison["matches"], "the real case must have at least one eligible match"

    no_match_run = result.run_ids[1]
    no_match_report = _report({"report.json": str(tmp_path / "out" / no_match_run / "report.json")})
    no_match_comparison = _tool_result(no_match_report["turn_result"], "retrieve_case_comparisons")
    assert no_match_comparison is not None
    assert no_match_comparison["status"] == "no_match"

    if host.memory_bank is not None:
        assert any("dropped_after_revocation=" in n for n in result.notes)
        drop_note = next(n for n in result.notes if "dropped_after_revocation=" in n)
        assert "dropped_after_revocation=[]" not in drop_note, (
            f"expected a genuine drop to be recorded, got: {drop_note}"
        )




def test_resume_after_restart_stale_record_version(tmp_path):
    host = _host(tmp_path)
    result = scen.resume_after_restart(host, output_dir=tmp_path / "out")
    assert result.terminal_state == "rejected"
    stale_run_id = result.run_ids[1]
    trace = _trace({"trace.jsonl": str(tmp_path / "out" / stale_run_id / "trace.jsonl")})
    await_entries = [t for t in trace if t["node"] == "await_confirmation"]
    assert await_entries, "resume_after_restart's stale run must have gone through await_confirmation"
    assert await_entries[-1]["outputs_summary"]["reason"] == "stale_record_version"




def test_out_of_scope_scoped_refusal_no_forbidden_phrases(tmp_path):
    host = _host(tmp_path)
    result = scen.out_of_scope(host, output_dir=tmp_path / "out")
    assert result.terminal_state == "refused_out_of_scope"
    report = _report(result.report_paths)
    tr = report["turn_result"]
    assert tr["checker_verdict"]["ok"] is True
    text_lower = tr.get("text", "").lower()
    for forbidden in ("verified diagnosis", "caused melanoma", "familiar image", "checklist applies"):
        assert forbidden not in text_lower
    for scope_word in ("margin", "prognosis", "survival"):
        # the refusal may name what it cannot do, but must never issue advice on it
        assert "should" not in text_lower or scope_word not in text_lower




def test_run_all_writes_summary_with_all_nine_terminal_states(tmp_path):
    from dermpact.core.prototype.scenarios import run_all
    summary = run_all(state_dir=tmp_path / "state", inventory_dir=inv.DEFAULT_INVENTORY_DIR, output_dir=tmp_path / "out")
    assert not summary["errors"], summary["errors"]
    assert len(summary["results"]) == len(scen.SCENARIO_NAMES)
    summary_path = Path(summary["summary_path"])
    assert summary_path.is_file()
    text = summary_path.read_text(encoding="utf-8")
    for name in scen.SCENARIO_NAMES:
        assert name in text
    _assert_no_secrets_in_tree(tmp_path / "out")


def test_run_all_is_isolated_across_scenarios_and_reruns(tmp_path):
    """WP-D fix: a shared state_dir across scenarios (and across repeated run_all calls) let a later scenario/call find an earlier one's committed clinician fact already on the case, so e.g. clarification could finish 'answered' without ever pausing."""
    from dermpact.core.prototype.scenarios import run_all
    state_dir = tmp_path / "state"

    summary1 = run_all(state_dir=state_dir, inventory_dir=inv.DEFAULT_INVENTORY_DIR, output_dir=tmp_path / "out1")
    assert not summary1["errors"], summary1["errors"]
    summary2 = run_all(state_dir=state_dir, inventory_dir=inv.DEFAULT_INVENTORY_DIR, output_dir=tmp_path / "out2")
    assert not summary2["errors"], summary2["errors"]

    terminal_1 = {r["scenario"]: r["terminal_state"] for r in summary1["results"]}
    terminal_2 = {r["scenario"]: r["terminal_state"] for r in summary2["results"]}
    assert terminal_1 == terminal_2

    for summary in (summary1, summary2):
        clarification_result = next(r for r in summary["results"] if r["scenario"] == "clarification")
        assert clarification_result["terminal_state"] == "committed"
        assert not any("did not pause as expected" in n for n in clarification_result["notes"]), clarification_result["notes"]
        assert clarification_result["record_version"] == 3


def test_run_all_real_cases_gate_display(tmp_path):
    """Per the orchestrator's note: record_v2/gate_v2 were fixed after the first prototype_runs_v1 generation so records built from the real WP-X inventory now gate 'display' instead of 'partial_report'."""
    from dermpact.core.prototype.scenarios import run_all
    out = tmp_path / "out"
    summary = run_all(state_dir=tmp_path / "state", inventory_dir=inv.DEFAULT_INVENTORY_DIR, output_dir=out)
    assert not summary["errors"], summary["errors"]
    checked = 0
    for r in summary["results"]:
        for run_id in r["run_ids"]:
            report_path = out / run_id / "report.json"
            if not report_path.is_file():
                continue  # a run left paused (no scenario here) never reaches persist
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["identity"]["source_label"] == "development_allowlisted_case":
                assert report["gate"]["disposition"] == "display", (r["scenario"], run_id, report["gate"])
                checked += 1
    assert checked > 0, "expected at least one development_allowlisted_case report across the nine scenarios"


def test_every_report_carries_mode_and_source_label(tmp_path):
    host = _host(tmp_path)
    result = scen.evidence_report(host, output_dir=tmp_path / "out")
    report = _report(result.report_paths)
    assert report.get("mode") in ("offline", "live-deferred")
    assert report["identity"].get("source_label") in (
        "development_allowlisted_case", "synthetic_workflow_fixture", "development_oof_benchmark",
    )




@pytest.mark.slow
def test_region_sensitivity_real_worker_smoke_derm7pt_case_22(tmp_path):
    """One real-model smoke: pings a real :class:`WorkerClient`, and only if it answers within 180s runs ``region_sensitivity --inference worker`` on ``derm7pt-case-22`` (the one allowlisted Phase-3 smoke case)."""
    client = WorkerClient(timeout_seconds=180.0)
    try:
        client.ping()
    except (WorkerTimeout, WorkerCrashed, WorkerRequestError, OSError) as exc:
        pytest.skip(f"real inference worker did not answer ping() within 180s: {exc}")
    finally:
        client.close()

    host = _host(tmp_path, inference="worker")
    result = scen.region_sensitivity(host, output_dir=tmp_path / "out", case_id=inv.BASE_FIXTURE_CASE_ID)
    assert result.case_id == inv.BASE_FIXTURE_CASE_ID
    report = _report(result.report_paths)
    assert report["mode"] == "offline"
    receipts = host.store.load_xai_receipts(inv.BASE_FIXTURE_CASE_ID)
    if receipts:
        receipt = receipts[-1]
        assert receipt["status"] in ("complete", "failed")
        if receipt["status"] == "complete":
            case_state = host.store.load_case(inv.BASE_FIXTURE_CASE_ID)
            assert xai.verify_receipt(receipt, case_state.current_record.to_dict()) is True
