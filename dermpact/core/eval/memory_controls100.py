"""Focused 100-case visual-only and deterministic memory-panel controls."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from dermpact.core import memory
from dermpact.core.agent.tools import ToolRegistry
from dermpact.core.eval import agent_eval
from dermpact.core.eval.agent_live_eval import load_frozen_tasks
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _atomic_write, _sha256_file

ROOT = Path(__file__).resolve().parents[3]
TASKS = ROOT / "research/manifests/dermpact_phase7/agent_confirmatory300_v2_grounded_reads/task_inventory_300.json"
CASES = ROOT / "research/experiments/dermpact_phase7/confirmatory_inventory100_v1"
FULL_MEMORY = ROOT / "research/manifests/dermpact_phase7/agent_memory_ablation100_v1/results.json"
OUT = ROOT / "research/manifests/dermpact_phase7/memory_controls100_v1"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a)) * float(np.linalg.norm(b)), 1e-12)
    return float(np.dot(a, b) / denom)


def _visual_only(host: Any, case_id: str, *, k: int = 3) -> dict[str, Any]:
    """Plain nearest-image retrieval: no threshold, field comparison or outcome arbitration."""
    record = host.ensure_case(case_id).current_record
    query = np.asarray(host.inference_client.predict(case_id=case_id)["prediction"]["sentinel_embedding"], dtype=np.float64)
    candidates = []
    seen_images: set[str] = set()
    for episode in host.memory_bank.effective():
        if episode.case_id == case_id or episode.lesion_id == case_id:
            continue
        if episode.data_role != memory.DATA_ROLE_DEVELOPMENT or episode.eligibility != "eligible":
            continue
        if episode.image_sha256 == record.inputs.dermoscopic.sanitized_sha256:
            continue
        if episode.image_sha256 in seen_images:
            continue
        seen_images.add(episode.image_sha256)
        candidates.append((_cosine(query, np.asarray(episode.embedding, dtype=np.float64)), episode))
    candidates.sort(key=lambda item: (-item[0], item[1].episode_id))
    matches = [
        {
            "case_id": episode.case_id,
            "episode_id": episode.episode_id,
            "similarity": round(similarity, 6),
            "dataset_label": (episode.outcome or {}).get("diagnosis"),
        }
        for similarity, episode in candidates[:k]
    ]
    current = memory.harmonise_diagnosis(record.perception.diagnosis_head.get("provisional_class"))
    concordant = [
        memory.harmonise_diagnosis(match["dataset_label"]) == current
        for match in matches
        if memory.harmonise_diagnosis(match["dataset_label"]) != memory.COMPARISON_UNMAPPED
        and current != memory.COMPARISON_UNMAPPED
    ]
    return {
        "case_id": case_id,
        "matches": matches,
        "n_candidates": len(candidates),
        "top1_label_concordant": (concordant[0] if concordant else None),
        "label_concordance_at_k": (sum(concordant) / len(concordant) if concordant else None),
        "field_comparison_available": False,
        "counterexample_available": False,
        "diagnostic_effect": "none",
    }


def run(out_dir: Path = OUT) -> dict[str, Any]:
    tasks = [dict(task) for task in load_frozen_tasks(TASKS) if task["family"] == "case_comparison"]
    if len(tasks) != 100 or len({task["case_id"] for task in tasks}) != 100:
        raise AssertionError("expected the frozen 100-case comparison subset")
    scratch = Path(tempfile.mkdtemp(prefix="dermpact_memory_controls100_"))
    host = agent_eval.build_host(scratch / "state", CASES)
    visual_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    try:
        if host.memory_bank is None or host.memory_policy is None:
            raise RuntimeError("comparison bank and bound policy are required")
        for task in tasks:
            case_id = str(task["case_id"])
            started = time.perf_counter()
            visual = _visual_only(host, case_id)
            visual["latency_s"] = time.perf_counter() - started
            visual_rows.append(visual)

            record = host.ensure_case(case_id).current_record
            ctx, _, _ = host.tool_context(
                case_id,
                session_id=f"memory-panel:{task['task_id']}",
                actor="deterministic_memory_panel",
                memory_request_explicit=True,
            )
            started = time.perf_counter()
            receipt = ToolRegistry(ctx).call("retrieve_case_comparisons", {
                "case_id": case_id,
                "record_version": record.identity.record_version,
            })
            latency = time.perf_counter() - started
            if not receipt.ok:
                raise RuntimeError(f"{case_id}: deterministic memory panel failed: {receipt.error}")
            data = dict(receipt.data or {})
            panel_rows.append({
                "case_id": case_id,
                "status": data.get("status"),
                "n_matches": len(data.get("matches") or []),
                "counterexample_available": data.get("counterexample") is not None,
                "counterexample_policy_permitted": data.get("status") == "ok" and data.get("counterexample") is not None,
                "field_comparison_available": bool(any(
                    (match.get("matched_fields") or match.get("different_fields"))
                    for match in (data.get("matches") or [])
                )),
                "descriptive_counts_present": bool((data.get("descriptive_reference_label_counts") or {}).get("counts")),
                "no_diagnosis_rewrite_disclosed": "not used to alter the current prediction" in str(data.get("grounded_summary_text") or "").lower(),
                "grounded_summary_text": data.get("grounded_summary_text"),
                "latency_s": latency,
            })
    finally:
        host.close()
        shutil.rmtree(scratch, ignore_errors=True)

    def mean(values: list[float]) -> float | None:
        finite = [value for value in values if math.isfinite(value)]
        return float(np.mean(finite)) if finite else None

    full = json.loads(FULL_MEMORY.read_text(encoding="utf-8"))["summaries"]["full_bounded_agent"]
    result = {
        "schema_version": "dermpact-phase7-memory-controls100-v1",
        "status": "PASS",
        "n_cases": 100,
        "arms": {
            "visual_only_nearest_case_retrieval": {
                "retrieval_coverage": sum(bool(row["matches"]) for row in visual_rows) / len(visual_rows),
                "top1_label_concordance_descriptive": mean([float(row["top1_label_concordant"]) for row in visual_rows if row["top1_label_concordant"] is not None]),
                "mean_label_concordance_at_k_descriptive": mean([float(row["label_concordance_at_k"]) for row in visual_rows if row["label_concordance_at_k"] is not None]),
                "field_comparison_available_rate": 0.0,
                "counterexample_available_rate": 0.0,
                "mean_latency_s": mean([row["latency_s"] for row in visual_rows]),
            },
            "deterministic_vcm_panel": {
                "request_completion_rate": 1.0,
                "comparison_available_rate": sum(row["n_matches"] > 0 for row in panel_rows) / len(panel_rows),
                "counterexample_candidate_available_rate": sum(row["counterexample_available"] for row in panel_rows) / len(panel_rows),
                "counterexample_available_rate": sum(row["counterexample_policy_permitted"] for row in panel_rows) / len(panel_rows),
                "field_comparison_available_rate": sum(row["field_comparison_available"] for row in panel_rows) / len(panel_rows),
                "descriptive_label_counts_present_rate": sum(row["descriptive_counts_present"] for row in panel_rows) / len(panel_rows),
                "no_diagnosis_rewrite_disclosure_rate": sum(row["no_diagnosis_rewrite_disclosed"] for row in panel_rows) / len(panel_rows),
                "mean_latency_s": mean([row["latency_s"] for row in panel_rows]),
            },
            "full_bounded_agent_reference": {
                "request_completion_rate": full["completion_rate"]["estimate"],
                "comparison_available_rate": full["comparison_available_rate"],
                "counterexample_available_rate": full["counterexample_display_rate"],
                "final_hazard_rate": full["final_display_ohr"]["estimate"],
                "mean_latency_s": full["latency"]["end_to_end_s"]["mean"],
            },
        },
        "sources": {
            "task_inventory_sha256": _sha256_file(TASKS),
            "case_inventory_sha256": _sha256_file(CASES / "inventory_manifest.json"),
            "full_memory_result_sha256": _sha256_file(FULL_MEMORY),
            "memory_bank_sha256": host.memory_bank.bank_sha256,
        },
        "scope": {
            "official_or_final_rows_loaded": 0,
            "llm_calls": 0,
            "memory_changes_diagnosis": False,
            "label_concordance_is_diagnosis_uplift": False,
        },
    }
    out_dir = Path(out_dir)
    _atomic_write(out_dir / "visual_only_rows.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in visual_rows))
    _atomic_write(out_dir / "deterministic_panel_rows.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in panel_rows))
    _atomic_json(out_dir / "results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
