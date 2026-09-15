"""Paired case-level inference for the frozen 100-case, five-judge-pass RAGAS study."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "research/manifests/dermpact_phase7/agent_ragas_matched100_v2/results.json"
OUT = ROOT / "research/manifests/dermpact_phase7/agent_ragas_matched100_v2/paired_statistics.json"
CONTROLS = ("naive_static_rag", "same_backbone_no_tools")


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def analyze(source: Path = SOURCE, out_path: Path = OUT, *, seed: int = 20260909) -> dict[str, Any]:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    full = aggregate["full_bounded_agent"]["per_task_three_pass_mean"]
    rng = np.random.default_rng(seed)
    comparisons: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for control in CONTROLS:
        baseline = aggregate[control]["per_task_three_pass_mean"]
        ids = sorted(set(full) & set(baseline))
        if len(ids) != 100 or set(full) != set(baseline):
            raise AssertionError(f"{control}: expected 100 exactly paired tasks")
        differences = np.asarray([float(full[key]) - float(baseline[key]) for key in ids], dtype=np.float64)
        boot = differences[rng.integers(0, len(differences), size=(10000, len(differences)))].mean(axis=1)
        # Randomization test under exchangeability of the paired difference signs.
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(100000, len(differences)))
        permuted = (signs * differences).mean(axis=1)
        observed = float(differences.mean())
        p_value = float((np.count_nonzero(np.abs(permuted) >= abs(observed)) + 1) / (len(permuted) + 1))
        raw_p[control] = p_value
        comparisons[control] = {
            "n_paired_cases": len(ids),
            "mean_difference": observed,
            "median_difference": float(np.median(differences)),
            "sd_difference": float(differences.std(ddof=1)),
            "paired_standardized_mean_difference_dz": float(observed / differences.std(ddof=1)),
            "bootstrap_95_ci_mean_difference": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "two_sided_sign_flip_p": p_value,
            "n_randomizations": 100000,
        }
    adjusted = _holm(raw_p)
    for control, value in adjusted.items():
        comparisons[control]["holm_adjusted_p"] = value
    result = {
        "schema_version": "dermpact-phase7-ragas-paired-statistics-v1",
        "status": "PASS",
        "primary_system": "full_bounded_agent",
        "n_cases": 100,
        "judge_passes_per_case": 5,
        "unit_of_inference": "case-level mean across the five prespecified judge passes",
        "comparisons": comparisons,
        "multiplicity": "Holm correction across the two prespecified full-system comparisons",
        "scope": {"official_or_final_rows_loaded": 0},
    }
    _atomic_json(Path(out_path), result)
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
