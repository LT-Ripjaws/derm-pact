"""Case-balanced live comparison of deterministic applicability and three LLM arms."""
from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dermpact.core.eval import applicability_benchmark_v2 as bench
from dermpact.eval import dermpact_phase4_arms as arms_mod
from dermpact.eval import dermpact_phase4_evaluation as eval_mod
from dermpact.eval import dermpact_phase4_rule_catalog as catalog_mod
from dermpact.guidelines.reviewed_v2 import ReviewedApplicabilityEngine

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/manifests/dermpact_phase7/applicability_live100_v1"
QUOTAS = {"recommended": 34, "conditional": 33, "withheld_no_coverage": 33}
ARMS = ("direct_llm", "naive_rag", "llm_traversal")
MAX_LIVE_WORKERS = 4


def _cache_filename(run_id: str) -> str:
    """Return a readable cache name that is valid on Windows and POSIX."""
    safe = re.sub(r'[^A-Za-z0-9._-]+', "_", run_id).strip("._")
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    return f"{safe[:96]}--{digest}.json"


def _run_transport_resilient(arm_runner: Any, arm: str, item: dict[str, Any], *, attempts: int = 6) -> dict[str, Any]:
    """Retry transport failures only; never rescore or replace a received answer."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return arm_runner.run_arm(arm, item).to_dict()
        except (ConnectionError, TimeoutError, RuntimeError) as exc:
            last = exc
            if "gateway call failed" not in str(exc).lower() or attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"unreachable transport retry state: {last}")


def select_runs() -> list[dict[str, Any]]:
    runs = bench.build_runs().runs
    selected = []
    used_cases: set[str] = set()
    for status, quota in QUOTAS.items():
        candidates = [run for run in runs if run["gold_next_step"]["status"] == status]
        candidates.sort(key=lambda run: hashlib.sha256(
            f"dermpact-applicability-live100-v1:{status}:{run['run_id']}".encode()
        ).hexdigest())
        for run in candidates:
            if run["case_id"] in used_cases:
                continue
            selected.append(run)
            used_cases.add(run["case_id"])
            if sum(item["gold_next_step"]["status"] == status for item in selected) == quota:
                break
    if len(selected) != 100 or len(used_cases) != 100:
        raise AssertionError("could not build the 100-case applicability sample")
    return selected


def run(out_dir: Path = OUT, model: str = "qwen/qwen3-vl-32b-instruct") -> dict[str, Any]:
    selected = select_runs()
    out_dir = Path(out_dir)
    sample_path = out_dir / "sample.json"
    sample_sha = bench._atomic_json(sample_path, {
        "schema_version": "dermpact-applicability-live100-sample-v1",
        "status": "FROZEN_BEFORE_LIVE_CALLS", "quotas": QUOTAS,
        "n_cases": 100, "runs": selected, "official_or_final_rows_loaded": 0,
    })
    engine = ReviewedApplicabilityEngine()
    pages = catalog_mod.load_source_pages()["pages"]
    llm = arms_mod.LiveGatewayClient(model=model)
    arm_runner = arms_mod.Arms(engine.catalogue, source_pages=pages, llm=llm)
    outputs: dict[str, list[dict[str, Any]]] = {
        "deterministic_reviewed_v2": [bench.run_deterministic_reviewed_v2(engine, item) for item in selected]
    }
    for arm in ARMS:
        by_run_id: dict[str, dict[str, Any]] = {}
        pending: list[tuple[int, dict[str, Any], Path, str]] = []
        for index, item in enumerate(selected, start=1):
            cache_path = out_dir / "task_cache" / arm / _cache_filename(item["run_id"])
            key = hashlib.sha256(json.dumps({"sample_sha256": sample_sha, "arm": arm,
                                             "run_id": item["run_id"], "model": model}, sort_keys=True).encode()).hexdigest()
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text())
                if cached.get("cache_key") == key:
                    by_run_id[item["run_id"]] = cached["output"]
                    continue
            pending.append((index, item, cache_path, key))
        with ThreadPoolExecutor(max_workers=MAX_LIVE_WORKERS) as pool:
            futures = {
                pool.submit(_run_transport_resilient, arm_runner, arm, item): (index, item, cache_path, key)
                for index, item, cache_path, key in pending
            }
            for future in as_completed(futures):
                index, item, cache_path, key = futures[future]
                output = future.result()
                bench._atomic_json(cache_path, {"cache_key": key, "output": output})
                by_run_id[item["run_id"]] = output
                print(f"[{arm} {index}/100] {item['run_id']}", flush=True)
        outputs[arm] = [by_run_id[item["run_id"]] for item in selected]
    summaries = {}
    for arm, arm_outputs in outputs.items():
        scored = [bench.score_run(item, output) for item, output in zip(selected, arm_outputs)]
        summary = bench.summarise_arm(scored)
        summary["primary_ci"] = eval_mod.clustered_bootstrap_rate(scored, n_boot=10000, seed=20261400)
        summaries[arm] = summary
    result = {
        "schema_version": "dermpact-applicability-live100-v1", "status": "PASS",
        "model": model, "n_cases": 100, "sample_sha256": sample_sha, "summaries": summaries,
        "scope": {"official_or_final_rows_loaded": 0, "images_opened": 0,
                  "interpretation": "development-only matched applicability comparison"},
    }
    bench._atomic_json(out_dir / "results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
