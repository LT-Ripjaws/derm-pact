"""Deterministic adversarial audit of paper 135's reproducible GMM-ATKS node.

This audit covers only DynaRAG equations 3--6.  It is intentionally not a
reimplementation or evaluation of the paper's QAF, AAM, training losses, or
LLaVA-Med system.  Synthetic scores expose the selector's design boundary
before spending model calls or admitting it to a real-data node tournament.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from medguard.pipeline.adaptive_selector import GMMAdaptiveTopKSelector


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "research" / "experiments" / "retrieval"
    / "dynarag_gmm_component_audit_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scenario_suite() -> List[Dict[str, Any]]:
    """Predeclared score-only probes; indices encode independent relevance labels."""
    return [
        {
            "id": "clean_bimodal",
            "kind": "clean",
            "scores": [0.95, 0.90, 0.85, 0.20, 0.15, 0.10],
            "relevant": [0, 1, 2],
            "poisoned": [],
            "purpose": "Can the selector remove a separated low-score distractor mode?",
        },
        {
            "id": "clean_near_flat",
            "kind": "clean",
            "scores": [0.510, 0.505, 0.500, 0.495, 0.490, 0.485],
            "relevant": [0, 1, 2],
            "poisoned": [],
            "purpose": "Does adaptive top-k still reduce context without separable modes?",
        },
        {
            "id": "clean_high_score_outlier",
            "kind": "clean",
            "scores": [0.99, 0.62, 0.60, 0.58, 0.20, 0.18, 0.15],
            "relevant": [0, 1, 2, 3],
            "poisoned": [],
            "purpose": "Can highest-mean selection discard useful lower-score evidence?",
        },
        {
            "id": "single_high_score_poison",
            "kind": "attack",
            "scores": [0.99, 0.82, 0.78, 0.22, 0.18, 0.10],
            "relevant": [1, 2],
            "poisoned": [0],
            "purpose": "Can score-only mixture selection reject one optimized poison?",
        },
        {
            "id": "high_score_false_consensus",
            "kind": "attack",
            "scores": [0.99, 0.97, 0.95, 0.85, 0.82, 0.20, 0.15, 0.10],
            "relevant": [3, 4],
            "poisoned": [0, 1, 2],
            "purpose": "Can score-only mixture selection reject a poisoned top mode?",
        },
    ]


def _fixed_top_k(scores: Sequence[float], k: int = 3) -> List[int]:
    ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return sorted(ranked[: min(k, len(scores))])


def _metrics(selected: Iterable[int], scenario: Mapping[str, Any]) -> Dict[str, Any]:
    chosen = set(selected)
    relevant = set(scenario["relevant"])
    poisoned = set(scenario["poisoned"])
    return {
        "selected_count": len(chosen),
        "selected_fraction": len(chosen) / len(scenario["scores"]),
        "relevant_precision": len(chosen & relevant) / len(chosen) if chosen else 0.0,
        "relevant_recall": len(chosen & relevant) / len(relevant) if relevant else 1.0,
        "poison_selected_count": len(chosen & poisoned),
        "poison_fraction_of_selection": (
            len(chosen & poisoned) / len(chosen) if chosen else 0.0
        ),
        "attack_contained": not bool(chosen & poisoned),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def audit() -> Dict[str, Any]:
    selector = GMMAdaptiveTopKSelector(
        gamma=0.5, max_components=4, min_selected=1,
        covariance_floor=1e-6, n_init=5, random_state=0,
    )
    rows: List[Dict[str, Any]] = []
    for scenario in scenario_suite():
        adaptive = selector.select(scenario["scores"])
        repeated = selector.select(scenario["scores"])
        fixed_indices = _fixed_top_k(scenario["scores"])
        rows.append({
            "id": scenario["id"],
            "kind": scenario["kind"],
            "purpose": scenario["purpose"],
            "scores": scenario["scores"],
            "relevant_indices": scenario["relevant"],
            "poisoned_indices": scenario["poisoned"],
            "adaptive": {
                "selected_indices": adaptive.selected_indices,
                "n_components": adaptive.n_components,
                "bic": adaptive.bic,
                "fallback_reason": adaptive.fallback_reason,
                "metrics": _metrics(adaptive.selected_indices, scenario),
            },
            "fixed_top3": {
                "selected_indices": fixed_indices,
                "metrics": _metrics(fixed_indices, scenario),
            },
            "deterministic": adaptive == repeated,
        })

    by_id = {row["id"]: row for row in rows}
    clean_adaptive = [row["adaptive"]["metrics"] for row in rows if row["kind"] == "clean"]
    clean_fixed = [row["fixed_top3"]["metrics"] for row in rows if row["kind"] == "clean"]
    attack_adaptive = [row["adaptive"]["metrics"] for row in rows if row["kind"] == "attack"]
    attack_fixed = [row["fixed_top3"]["metrics"] for row in rows if row["kind"] == "attack"]

    implementation_checks = {
        "deterministic_for_all_scenarios": all(row["deterministic"] for row in rows),
        "separated_low_score_mode_removed": (
            by_id["clean_bimodal"]["adaptive"]["selected_indices"] == [0, 1, 2]
        ),
        "flat_score_limitation_exposed": (
            by_id["clean_near_flat"]["adaptive"]["metrics"]["selected_fraction"] == 1.0
        ),
        "useful_lower_mode_underrecall_exposed": (
            by_id["clean_high_score_outlier"]["adaptive"]["metrics"]["relevant_recall"]
            < 1.0
        ),
        "single_high_score_poison_failure_exposed": (
            by_id["single_high_score_poison"]["adaptive"]["metrics"]
            ["poison_selected_count"] > 0
        ),
        "false_consensus_poison_failure_exposed": (
            by_id["high_score_false_consensus"]["adaptive"]["metrics"]
            ["poison_selected_count"] > 0
        ),
    }
    summaries = {
        "adaptive_clean_macro_relevant_recall": _mean(clean_adaptive, "relevant_recall"),
        "fixed_top3_clean_macro_relevant_recall": _mean(clean_fixed, "relevant_recall"),
        "adaptive_attack_containment_rate": _mean(attack_adaptive, "attack_contained"),
        "fixed_top3_attack_containment_rate": _mean(attack_fixed, "attack_contained"),
        "adaptive_attack_macro_relevant_recall": _mean(attack_adaptive, "relevant_recall"),
        "fixed_top3_attack_macro_relevant_recall": _mean(attack_fixed, "relevant_recall"),
    }
    robustness_gate = (
        summaries["adaptive_clean_macro_relevant_recall"] >= 0.90
        and summaries["adaptive_attack_containment_rate"]
        > summaries["fixed_top3_attack_containment_rate"]
        and summaries["adaptive_attack_macro_relevant_recall"]
        >= summaries["fixed_top3_attack_macro_relevant_recall"]
    )
    return {
        "protocol": "dynarag-gmm-atks-component-adversarial-audit-v1",
        "deterministic": True,
        "model_calls": 0,
        "development_only": True,
        "scope": "Paper 135 equations 3--6 only; not full DynaRAG H2H.",
        "implementation_checks": implementation_checks,
        "implementation_checks_passed": all(implementation_checks.values()),
        "summaries": summaries,
        "scenarios": rows,
        "robustness_gate_passed": robustness_gate,
        "decision": (
            "promote_to_real_data_retrieval_tournament"
            if robustness_gate
            else "retain_only_as_clean_retrieval_ablation_not_security_controller"
        ),
        "claim_boundary": (
            "Synthetic score probes validate implementation behavior, not clinical performance. "
            "GMM-ATKS can remove a separated low-score mode but has no provenance, semantic, or "
            "integrity signal and therefore selects optimized high-score poison by design."
        ),
    }


def run(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    implementation = Path(__file__)
    selector = ROOT / "medguard" / "pipeline" / "adaptive_selector.py"
    paper = ROOT / "papers" / "raw_texts" / "135.txt"
    manifest = {
        "protocol": "dynarag-gmm-atks-component-adversarial-audit-v1",
        "frozen_before_real_data_retrieval_evaluation": True,
        "scenario_count": len(scenario_suite()),
        "inputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
            for path in (implementation, selector, paper)
        ],
        "declared_deviations": [
            "The paper omits the BIC component search range; use K=1..min(4,floor(N/2)).",
            "The paper omits EM stability settings; use full covariance, reg_covar=1e-6, n_init=5, random_state=0.",
            "No QAF, AAM, L_quality, L_balance, generator training, or paper corpus is implemented.",
        ],
        "references": ["https://openreview.net/forum?id=7VaClU4BXo"],
        "candidate_acceptance_rule": (
            "Clean macro relevant recall >=0.90; attack containment strictly exceeds fixed top-3; "
            "attack relevant recall is no worse than fixed top-3."
        ),
    }
    manifest_path = output_dir / "protocol_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = audit()
    result["protocol_manifest_sha256"] = _sha256(manifest_path)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "implementation_checks_passed": result["implementation_checks_passed"],
        "robustness_gate_passed": result["robustness_gate_passed"],
        "decision": result["decision"],
        "summaries": result["summaries"],
    }, sort_keys=True))
    return 0 if result["implementation_checks_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
