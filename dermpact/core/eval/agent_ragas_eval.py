"""Three-pass RAGAS-style faithfulness evaluation for Phase-7 agent arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import httpx
import numpy as np

from dermpact import config
from dermpact.eval.ragas_judge import JudgeCache, RagasFaithfulnessJudge, judge_specification_sha256
from dermpact.core.eval.agent_live_eval import TASK_INVENTORY, load_frozen_tasks
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _sha256_file


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "research/manifests/dermpact_phase7/agent_ragas_v1"
FULL_TRANSCRIPTS = ROOT / "research/manifests/dermpact_phase7/agent_live_eval_ragas_dev12_promptv2/transcripts.jsonl"
ABLATION_TRANSCRIPTS = ROOT / "research/manifests/dermpact_phase7/agent_ablation_matrix_v2_frozen_no_postadjust/live_control_transcripts.jsonl"
JUDGE_MODEL = "openai/gpt-oss-20b"
ARMS = ("full_bounded_agent", "same_backbone_no_tools", "naive_static_rag")
ELIGIBLE_FAMILIES = frozenset({"evidence_report", "explain_deferral", "region_sensitivity", "clinical_photo", "case_comparison"})
N_PASSES = 3
CONTEXT_PROJECTION_VERSION = "provider-json-to-proposition-lines-v2-relational"


def baseline_rows_required(arms: Sequence[str]) -> bool:
    return any(arm != "full_bounded_agent" for arm in arms)


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _render_context(raw: str) -> str:
    """Readable projection of the same provider-visible JSON, with no added facts."""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return str(raw)
    if not isinstance(value, Mapping):
        return str(raw)
    if "tool" not in value:
        if value.get("text"):
            return f"Guideline passage {value.get('document_id', '')}: {value['text']}"
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    binding = value.get("binding") or {}
    lines = [
        f"Tool {value.get('tool')} returned ok={value.get('ok')} for case_id={binding.get('case_id')} "
        f"record_version={binding.get('record_version')}."
    ]
    error = value.get("error") or {}
    if error:
        lines.append(f"Tool error code={error.get('code')}: {error.get('detail')}")
    for proposition in value.get("propositions") or []:
        if "grounded_summary_text" in str(proposition.get("path_or_ref", "")):
            continue
        lines.append(
            f"Proposition {proposition.get('id')}: {proposition.get('path_or_ref')} = "
            f"{json.dumps(proposition.get('value'), ensure_ascii=False)}"
        )
    data = value.get("data")
    if value.get("tool") == "retrieve_case_comparisons" and isinstance(data, Mapping):
        label_counts = data.get("descriptive_reference_label_counts") or {}
        if isinstance(label_counts, Mapping):
            counts = label_counts.get("counts") or {}
            if isinstance(counts, Mapping) and counts:
                rendered_counts = ", ".join(
                    f"{label}={count}" for label, count in sorted(counts.items())
                )
                lines.append(
                    f"{label_counts.get('n_retrieved')} returned reference cases have "
                    f"dataset-label counts {rendered_counts}."
                )
                lines.append(
                    "Diagnostic effect: "
                    f"{label_counts.get('diagnostic_effect')}."
                )
        matches = data.get("matches") or []
        if isinstance(matches, list):
            for match in matches:
                if not isinstance(match, Mapping):
                    continue
                outcome = match.get("outcome") or {}
                diagnosis = match.get("diagnosis") or {}
                lines.append(
                    f"Retrieved case {match.get('case_id')} has similarity "
                    f"{match.get('similarity')} and dataset label "
                    f"{outcome.get('diagnosis')}; diagnosis relation "
                    f"{diagnosis.get('status')}; matched fields "
                    f"{match.get('matched_fields') or []}; different fields "
                    f"{match.get('different_fields') or []}."
                )
        counterexample = data.get("counterexample")
        if isinstance(counterexample, Mapping):
            outcome = counterexample.get("outcome") or {}
            lines.append(
                f"Counterexample case {counterexample.get('case_id')} has similarity "
                f"{counterexample.get('similarity')} and dataset label "
                f"{outcome.get('diagnosis')}."
            )
    if len(lines) == 1 and data is not None:
        lines.append("Tool data: " + json.dumps(value["data"], sort_keys=True, ensure_ascii=False))
    return "\n".join(lines)


class HttpxJudgeClient:
    """Minimal thread-safe client matching ``RagasFaithfulnessJudge._client``."""

    def __init__(self, model: str = JUDGE_MODEL, timeout_s: float = 90.0, retries: int = 5):
        config.load_env()
        self.model = model
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.call_log: list[Dict[str, Any]] = []

    def chat(self, messages: list[Dict[str, Any]], *, max_tokens: int, temperature: float, seed: int) -> str:
        if os.environ.get("DERMPACT_ALLOW_LIVE_LLM") != "1":
            raise RuntimeError("DERMPACT_ALLOW_LIVE_LLM=1 is required for judge calls")
        config.load_env()
        key = config.GATEWAY_API_KEY
        if not key:
            raise RuntimeError("configured gateway key is absent")
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "seed": seed}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(f"{config.GATEWAY_BASE_URL}/chat/completions", json=body, headers=headers)
                if response.status_code >= 500:
                    raise RuntimeError(f"gateway returned {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                usage = payload.get("usage") or {}
                content = str(payload["choices"][0]["message"]["content"] or "").strip()
                if not content:
                    raise RuntimeError("gateway returned an empty judge response")
                self.call_log.append({
                    "succeeded": True, "attempt": attempt + 1,
                    "latency_s": time.perf_counter() - started,
                    "usage": {k: v for k, v in usage.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
                })
                return content
            except Exception as exc:  # noqa: BLE001 - bounded transport retry
                last_error = exc
                self.call_log.append({"succeeded": False, "attempt": attempt + 1,
                                      "latency_s": time.perf_counter() - started,
                                      "error_type": type(exc).__name__, "usage": {}})
        raise RuntimeError(f"judge call failed after retries: {type(last_error).__name__}")


def build_samples(*, full_transcripts: Path = FULL_TRANSCRIPTS,
                  ablation_transcripts: Path = ABLATION_TRANSCRIPTS,
                  arms: Sequence[str] = ARMS,
                  task_inventory: Path = TASK_INVENTORY) -> list[Dict[str, Any]]:
    tasks = {task["task_id"]: task for task in load_frozen_tasks(task_inventory)}
    full = {row["run"]["task_id"]: row["run"] for row in _load_jsonl(full_transcripts)}
    baseline_rows = _load_jsonl(ablation_transcripts) if baseline_rows_required(arms) else []
    baselines = {(row["arm"], row["task_id"]): row for row in baseline_rows}
    eligible_ids = [
        task_id for task_id, task in tasks.items()
        if task["family"] in ELIGIBLE_FAMILIES
        and task.get("kind") == "paraphrase"
        and not str(task_id).startswith(("review-", "reacquisition-"))
        and full.get(task_id, {}).get("grounding_contexts")
        and all(baselines.get((arm, task_id), {}).get("grounding_contexts")
                for arm in arms if arm != "full_bounded_agent")
    ]
    samples = []
    for task_id in eligible_ids:
        task = tasks[task_id]
        rows = {"full_bounded_agent": full[task_id]}
        for baseline_arm in arms:
            if baseline_arm != "full_bounded_agent":
                rows[baseline_arm] = baselines[(baseline_arm, task_id)]
        for arm in arms:
            row = rows[arm]
            samples.append({
                "task_id": task_id, "case_id": task["case_id"], "family": task["family"], "arm": arm,
                "user_input": task["message"], "response": row["text"],
                "retrieved_contexts": [_render_context(context) for context in row["grounding_contexts"]],
            })
    return samples


def _usage(log: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    numeric: Dict[str, float] = {}
    for event in log:
        for key, value in (event.get("usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[key] = numeric.get(key, 0.0) + float(value)
    latency = [float(event.get("latency_s") or 0.0) for event in log]
    return {
        "calls": len(log), "failed_attempts": sum(not event.get("succeeded") for event in log),
        "numeric": numeric, "latency_s": {"sum": float(sum(latency)),
        "mean": float(np.mean(latency)) if latency else 0.0,
        "p95": float(np.percentile(latency, 95)) if latency else 0.0},
    }


def _aggregate(passes: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]],
               arms: Sequence[str] = ARMS, task_inventory: Path = TASK_INVENTORY) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for arm in arms:
        indices = [i for i, sample in enumerate(samples) if sample["arm"] == arm]
        per_pass = []
        task_values: Dict[str, list[float]] = {}
        for pass_row in passes:
            values = pass_row["scores"]["faithfulness"]["per_sample"]
            arm_values = [values[i] for i in indices]
            scored = [float(value) for value in arm_values if value is not None]
            per_pass.append({"mean": float(np.mean(scored)) if scored else None,
                             "n_scored": len(scored), "n_samples": len(arm_values)})
            for index, value in zip(indices, arm_values):
                if value is not None:
                    task_values.setdefault(str(samples[index]["task_id"]), []).append(float(value))
        pass_means = [row["mean"] for row in per_pass if row["mean"] is not None]
        result[arm] = {
            "per_pass": per_pass,
            "mean_across_passes": float(np.mean(pass_means)) if pass_means else None,
            "std_across_passes": float(np.std(pass_means)) if pass_means else None,
            "min_pass_mean": min(pass_means) if pass_means else None,
            "max_pass_mean": max(pass_means) if pass_means else None,
            "per_task_three_pass_mean": {task_id: float(np.mean(values)) for task_id, values in task_values.items()},
        }
    return result


def run(*, out_dir: Path = OUTPUT_DIR, max_workers: int = 4, n_passes: int = N_PASSES,
        full_transcripts: Path = FULL_TRANSCRIPTS,
        ablation_transcripts: Path = ABLATION_TRANSCRIPTS,
        arms: Sequence[str] = ARMS, task_inventory: Path = TASK_INVENTORY) -> Dict[str, Any]:
    arms = tuple(arms)
    if n_passes < 1:
        raise ValueError("n_passes must be positive")
    samples = build_samples(full_transcripts=full_transcripts, ablation_transcripts=ablation_transcripts,
                            arms=arms, task_inventory=task_inventory)
    if not samples or len(samples) % len(arms):
        raise AssertionError("matched RAGAS sample matrix is empty or unbalanced")
    pass_rows = []
    for pass_index in range(1, n_passes + 1):
        print(f"[RAGAS pass {pass_index}/{n_passes}] {len(samples)} arm-task samples", flush=True)
        client = HttpxJudgeClient(JUDGE_MODEL)
        cache = JudgeCache(out_dir / "judge_cache" / f"pass_{pass_index}", session_id=f"phase7-ragas-pass-{pass_index}")
        judge = RagasFaithfulnessJudge(
            client=client, model=JUDGE_MODEL, max_tokens=2048, seed=0,
            max_workers=max_workers, cache=cache,
        )
        judge.assert_matches_specification()
        scores = judge.evaluate([{
            "user_input": sample["user_input"], "response": sample["response"],
            "retrieved_contexts": sample["retrieved_contexts"],
        } for sample in samples], ("faithfulness",))
        pass_row = {"pass": pass_index, "scores": scores, "usage": _usage(client.call_log),
                    "cache": {"hits": cache.hits, "misses": cache.misses,
                              "session_composition": cache.session_composition()}}
        _atomic_json(out_dir / f"pass_{pass_index}.json", pass_row)
        pass_rows.append(pass_row)
    result = {
        "schema_version": "dermpact-phase7-agent-ragas-v1", "status": "PASS",
        "judge_model": JUDGE_MODEL, "judge_specification_sha256": judge_specification_sha256(),
        "n_passes": n_passes, "n_tasks_per_arm": len(samples) // len(arms),
        "arms": list(arms),
        "n_arm_task_samples_per_pass": len(samples), "eligible_families": sorted(ELIGIBLE_FAMILIES),
        "context_projection_version": CONTEXT_PROJECTION_VERSION,
        "aggregate": _aggregate(pass_rows, samples, arms),
        "passes": [{"pass": row["pass"], "usage": row["usage"], "cache": row["cache"]} for row in pass_rows],
        "sources": {"full": _sha256_file(full_transcripts),
                    **({"ablations": _sha256_file(ablation_transcripts)} if baseline_rows_required(arms) else {}),
                    "tasks": _sha256_file(task_inventory)},
        "scope": {"raw_images_sent": 0, "patient_identifiers_sent": 0,
                  "official_or_final_rows_loaded": 0,
                  "interpretation": "supporting LLM-judge metric; deterministic checker remains primary"},
    }
    _atomic_json(out_dir / "results.json", result)
    _atomic_json(out_dir / "sample_manifest.json", {"samples": samples})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--n-passes", type=int, default=N_PASSES)
    parser.add_argument("--full-transcripts", type=Path, default=FULL_TRANSCRIPTS)
    parser.add_argument("--task-inventory", type=Path, default=TASK_INVENTORY)
    parser.add_argument("--ablation-transcripts", type=Path, default=ABLATION_TRANSCRIPTS)
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args(argv)
    selected_arms = ("full_bounded_agent",) if args.full_only else ARMS
    result = run(out_dir=args.out_dir, max_workers=args.max_workers, n_passes=args.n_passes,
                 full_transcripts=args.full_transcripts,
                 ablation_transcripts=args.ablation_transcripts, arms=selected_arms,
                 task_inventory=args.task_inventory)
    print(json.dumps({"status": result["status"], "judge": result["judge_model"],
                      "n_tasks_per_arm": result["n_tasks_per_arm"],
                      "aggregate": result["aggregate"], "out_dir": str(args.out_dir)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
