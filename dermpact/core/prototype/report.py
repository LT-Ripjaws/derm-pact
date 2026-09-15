"""WP-D: JSON + Markdown report writer."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from dermpact.core import renderer_v2
from dermpact.core.episode import what_remains as episode_what_remains
from dermpact.core.record_v2 import StructuredCaseRecordV2
from dermpact.core.store import CaseState


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json(path: Path, payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    _atomic_write_text(path, text)
    import hashlib
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _atomic_write_text(Path(str(path) + ".sha256"), digest + "\n")
    return digest


def _write_text(path: Path, text: str) -> None:
    _atomic_write_text(path, text)


def _markdown(report: Mapping[str, Any], rendered: renderer_v2.RenderedOutput) -> str:
    lines = [
        f"# DermPact prototype report -- {report['run_id']}",
        "",
        f"- case_id: `{report['identity']['case_id']}`",
        f"- source_label: `{report['identity']['source_label']}` "
        + ("(SYNTHETIC WORKFLOW FIXTURE -- never a real case)" if report['identity']['source_label'] == "synthetic_workflow_fixture" else ""),
        f"- record_version: {report['identity']['record_version']} (record_sha256 `{report['record_sha256']}`)",
        f"- scenario: `{report['scenario']}` | mode: `{report['mode']}`"
        + (f" (live_status={report['live_status']})" if report.get("live_status") else ""),
        f"- terminal_state: `{report['terminal_state']}`",
        f"- gate disposition: `{report['gate']['disposition']}` (passed={report['gate']['passed']})",
        "",
        "## Top layer", "", rendered.layers.get("top", ""), "",
        "## Evidence layer", "", rendered.layers.get("evidence", ""), "",
        "## Audit layer", "", rendered.layers.get("audit", ""), "",
    ]
    warnings = report.get("warnings") or []
    lines += ["## Warnings", ""]
    if warnings:
        for w in warnings:
            lines.append(
                f"- [{w.get('code')}] {w.get('reason')} | {w.get('measurement')} | "
                f"affected outputs: {w.get('affected_outputs')} | "
                f"remaining usable output: {w.get('remaining_usable_output')} | "
                f"recovery: {w.get('recovery_action')}"
            )
            if w.get("deferral_note"):
                lines.append(f"    {w['deferral_note']}")
    else:
        lines.append("none")
    lines.append("")
    if report.get("clarification"):
        lines += ["## Clarification analysis", "", f"caveat: {report['clarification'].get('caveat')}", ""]
        for q in report["clarification"].get("questions", []):
            lines.append(f"- {q['fact']}: decision_relevant={q['decision_relevant']} -- {q['reason']}")
        lines.append("")
    if report.get("turn_result"):
        tr = report["turn_result"]
        lines += ["## Agent turn", "", f"mode: `{tr['mode']}`", "", tr.get("text", ""), "",
                  f"tool calls: {len(tr.get('tool_trace', []))}; checker ok: {tr['checker_verdict'].get('ok')}", ""]
    if report.get("xai"):
        lines += ["## Measured XAI", "", f"```json\n{json.dumps(report['xai'], indent=2, sort_keys=True)[:4000]}\n```", ""]
    if report.get("memory"):
        lines += ["## Comparison memory", "", f"status: `{report['memory'].get('status')}`", ""]
    lines += ["## What remains", "", report["episode"]["what_remains"]["sentence"], ""]
    return "\n".join(lines)


def build_report(
    *, state: Mapping[str, Any], record: StructuredCaseRecordV2, verdict: Any, rendered: renderer_v2.RenderedOutput,
    view: Dict[str, Any], episode_state: Any, case_state: CaseState, host: Any,
) -> Dict[str, Any]:
    pending_proposal = None
    interrupt_info = state.get("pending_interrupt")
    if interrupt_info and interrupt_info.get("kind") == "confirmation":
        pending_proposal = {"proposal_id": interrupt_info.get("proposal_id")}
    remains = episode_what_remains(episode_state, record, pending_proposal=pending_proposal)

    xai_receipts = case_state.xai_receipt_ids
    xai_summary = None
    if xai_receipts:
        latest = host.store.load_xai_receipts(case_state.case_id)[-1]
        xai_summary = {"receipt_id": latest.get("receipt_id"), "kind": latest.get("kind"), "status": latest.get("status")}

    return {
        "run_id": state.get("run_id"), "scenario": state.get("scenario"), "mode": state.get("mode"),
        "inference": state.get("inference"), "live_status": state.get("live_status"),
        "terminal_state": state.get("terminal_state"), "identity": view["identity"], "record_sha256": record.record_sha256,
        "heads": view["heads"], "checklist": view["checklist"], "clinician_assertions": view["clinician_assertions"],
        "reliability": view["reliability"], "image_status": view["image_status"], "next_step": view["next_step"],
        "applicability": view["applicability"], "sources": view["sources"], "versions": view["versions"],
        "amendments": view["amendments"], "warnings": view["warnings"],
        "gate": {"disposition": verdict.disposition, "passed": verdict.passed, "receipt_sha256": verdict.receipt_sha256,
                 "failures": [{"code": f.code, "detail": f.detail} for f in verdict.failures]},
        "clarification": state.get("clarification"), "turn_result": state.get("turn_result"), "xai": xai_summary,
        "memory": state.get("memory"),
        "episode": {"state": episode_state.to_dict(), "what_remains": {"action": remains.action, "sentence": remains.sentence}},
        "trace": list(state.get("trace") or []),
    }


def write_report(
    output_dir: Path, run_id: str, report: Dict[str, Any], *, record: StructuredCaseRecordV2, case_state: CaseState,
    rendered: renderer_v2.RenderedOutput,
) -> Dict[str, str]:
    """Write the full per-run artefact bundle."""
    run_dir = Path(output_dir) / run_id
    paths: Dict[str, str] = {}

    _write_json(run_dir / "report.json", report)
    paths["report.json"] = str(run_dir / "report.json")

    _write_text(run_dir / "report.md", _markdown(report, rendered))
    paths["report.md"] = str(run_dir / "report.md")

    _write_json(run_dir / "state.json", report)
    paths["state.json"] = str(run_dir / "state.json")

    trace_lines = "\n".join(json.dumps(t, sort_keys=True) for t in report["trace"])
    _write_text(run_dir / "trace.jsonl", trace_lines + ("\n" if trace_lines else ""))
    paths["trace.jsonl"] = str(run_dir / "trace.jsonl")

    _write_json(run_dir / "gate_receipt.json", report["gate"])
    paths["gate_receipt.json"] = str(run_dir / "gate_receipt.json")

    if report.get("turn_result"):
        _write_json(run_dir / "turn_result.json", report["turn_result"])
        paths["turn_result.json"] = str(run_dir / "turn_result.json")

    if report.get("clarification"):
        _write_json(run_dir / "clarification.json", report["clarification"])
        paths["clarification.json"] = str(run_dir / "clarification.json")

    if report.get("memory"):
        _write_json(run_dir / "memory_comparison.json", report["memory"])
        paths["memory_comparison.json"] = str(run_dir / "memory_comparison.json")

    _write_json(run_dir / "episode.json", report["episode"])
    paths["episode.json"] = str(run_dir / "episode.json")

    for i, rec in enumerate(case_state.records, start=1):
        p = run_dir / f"record_v{i}.json"
        _write_json(p, rec.to_dict())
        paths[f"record_v{i}.json"] = str(p)

    return paths


def write_pause_snapshot(output_dir: Path, run_id: str, *, state: Mapping[str, Any], interrupt_payload: Any) -> Dict[str, str]:
    """A run that paused at ``interrupt()`` never reaches ``persist`` in the same invoke -- this writes the smaller bundle the CLI can still show: the raw interrupt payload and the pre-pause state fields."""
    run_dir = Path(output_dir) / run_id
    paths: Dict[str, str] = {}
    snapshot = {"run_id": run_id, "paused": True, "interrupt": interrupt_payload,
                **{k: v for k, v in state.items() if k not in ("trace", "__interrupt__")}}
    _write_json(run_dir / "state.json", snapshot)
    paths["state.json"] = str(run_dir / "state.json")
    trace_lines = "\n".join(json.dumps(t, sort_keys=True) for t in (state.get("trace") or []))
    _write_text(run_dir / "trace.jsonl", trace_lines + ("\n" if trace_lines else ""))
    paths["trace.jsonl"] = str(run_dir / "trace.jsonl")
    return paths


__all__ = ["build_report", "write_pause_snapshot", "write_report"]
