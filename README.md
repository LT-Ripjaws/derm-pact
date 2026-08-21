# MedGuard

Code and evaluation artifacts for the manuscript *"MedGuard: An Evidence-Grounded Agentic Retrieval-Augmented Clinical Decision Support System for Dermoscopic Assessment"* (under review, IEEE Access).

MedGuard combines a frozen dermatology image encoder (DermLIP ViT-B/16) with lightweight diagnosis and seven-point-checklist models, an unfamiliar-image warning, a bounded clinical agent that retrieves source-linked guideline passages, and independent rule-based checks that verify every score, citation, and clinician correction before display.

This repository exists so that reviewers and readers can inspect the code that produced the reported numbers and the stored evaluation artifacts those numbers were read from. Every headline value in the paper traces to a JSON artifact here, and most evaluations can be replayed offline from the cached model outputs.

## Layout

| Path | Contents |
|---|---|
| `medguard_v2/` | The system evaluated in the paper: data loaders, frozen-encoder perception, checklist scoring, decision record, guideline binding and retrieval, clinical agent, rule-based verification, evaluation code, and the unit-test suite |
| `medguard/` | Support package from an earlier project phase that `medguard_v2` and some scripts import (LLM client, judge utilities, metrics) plus the guideline record collection (`medguard/data/corpus/`) |
| `scripts/dev/` | Scripts that ran the evaluations and built every table and figure in the paper |
| `scripts/sealed/` | One-shot evaluation runners for the held-out reads |
| `notebooks/` | End-to-end walkthrough notebook |
| `research/evidence/` | The evaluation artifacts the paper reports (see the map below) |
| `research/manifests/` | Frozen dataset split manifests (HAM10000, BCN20000) |
| `research/experiments/` | Cached DermLIP feature matrices for HAM10000 and BCN20000 |
| `research/*.json`, `*.jsonl` | Governance records: experiment specifications with acceptance receipts, preregistrations, results ledger, runtime lock, and append-only test-set access logs |

## Where each reported number lives

| Paper item | Artifact | Producing script |
|---|---|---|
| Diagnostic performance table (Derm7pt, HAM10000, BCN20000) | `research/evidence/derm7pt_v1/paper_metrics_summary.json`, `research/evidence/ham_v1/ham_evaluation.json`, `research/evidence/bcn_v1/bcn_test_evaluation.json`, `research/evidence/sealed_v2/sealed_v2_evaluation.json` | `scripts/dev/build_ham_evaluation.py`, `scripts/dev/build_bcn_test_evaluation.py`, `scripts/sealed/run_sealed_v2.py` |
| ROC curves and confusion matrices | `research/evidence/derm7pt_v1/roc_overlay_curves.json`, `promoted_two_view_ablation.json`, `confusion_matrices.json` | `scripts/dev/build_paper_figures.py`, `scripts/dev/build_roc_overlay_and_gate_fig.py` |
| Criterion profile and complementarity | `criteria_regrouped_kawahara.json`, `complementarity_venn_data.json` | `scripts/dev/build_criteria_regrouped_kawahara.py` |
| Unfamiliar-image warning | `domain_gate_validation.json`, `domain_gate_residual.json` | `scripts/dev/validate_domain_gate.py`, `scripts/dev/build_domain_gate_residual.py` |
| Agent endpoints (591 questions, 203 cases) | `agent_eval_report_v5_4.json` (+ `.PROVENANCE.md`) | `medguard_v2/eval/agent_eval.py` |
| Evidence faithfulness (RAGAS) | `agent_ragas_metrics_v5_4.json` | `scripts/dev/build_ragas_agent_metrics.py` |
| Adversarial robustness (132/40/29 trials) | `redteam_v1.json` | `medguard_v2/eval/` red-team modules |
| Component ablation | `ablation_table.json`, `promoted_two_view_ablation.json`, `agent_eval_report_v5_4.json` | `scripts/dev/build_ablation_table.py` |

All artifacts record their seeds, split hashes, and boundaries (paid calls, images encoded, which guarded set was read). `research/SEALED_ACCESS_LOG.jsonl` and `research/V2_TEST_ACCESS_LOG.jsonl` are the append-only logs of every access to a held-out partition.

## What is deliberately not here

- **Dataset images.** Derm7pt, BCN20000, and HAM10000 are distributed by their custodians under their own licenses (see the paper's Data Availability section). The loaders in `medguard_v2/data/` expect locally downloaded copies.
- **Derm7pt-derived feature matrices and image renders.** Derm7pt is licensed CC BY-NC-ND 4.0 (no derivatives), so its encoded feature matrices and any figure containing image crops are excluded. They regenerate deterministically from a downloaded copy of Derm7pt with the frozen public encoder (`medguard_v2/perception/encode.py`; model `redlessone/DermLIP_ViT-B-16` on Hugging Face). HAM10000 (CC BY-NC 4.0) and BCN20000 (CC BY 4.0) feature matrices are included with attribution to their sources.
- **Guideline PDFs and the third-party literature corpus.** The source documents are copyrighted; `research/evidence/guideline_scope_v1/` retains the inventory, coverage matrix, and content digests so the ingested collection can be verified against the originals.
- **The earlier head-to-head generation study.** This repository covers the evaluations reported in the manuscript.

## Reproducing

```
python -m venv venv && venv/Scripts/pip install -r requirements.txt
python -m unittest discover -s medguard_v2/tests -q
```

Without any dataset downloaded, the suite collects 479 tests: 478 pass or skip (34 skips), and one module (`test_derm7pt`) reports a collection error because it loads the Derm7pt metadata at import time; it runs once Derm7pt is placed at `release_v0/` in the repository root. Everything else runs from the artifacts in this repository. The agent and judge evaluations replay offline from the caches under `research/evidence/derm7pt_v1/` (`agent_cache/`, `ragas_judge_cache/`, `verifier_cache/`); each cache entry stores the model identifier and reply for one call, which is also direct evidence of the runs. Live generation, if you want to re-run rather than replay, needs an OpenRouter-compatible key in `MEDGUARD_API_KEY`.

Figure scripts assert their recomputed values against the stored artifacts before drawing (see `scripts/dev/build_roc_overlay_and_gate_fig.py`), so a successful figure build is itself a reproduction check.

## Licensing

The code in this repository (`medguard/`, `medguard_v2/`, `scripts/`, `notebooks/`) and the evaluation artifacts we produced (`research/evidence/*.json`, governance records) are released under the MIT License (see `LICENSE`).

Some files derive from third-party material and keep their upstream terms, which take precedence over the MIT license for those files:

- Feature matrices under `research/experiments/.../features/` are derived from HAM10000 (CC BY-NC 4.0; Tschandl et al., Sci. Data 2018) and BCN20000 (CC BY 4.0; Hernandez-Perez et al., Sci. Data 2024). The HAM10000-derived files are therefore for non-commercial use with attribution.
- `medguard/data/corpus/derm_guideline_recommendations_v1.jsonl` contains short passages quoted verbatim, with source identifiers, from published clinical guidelines (AAD, EADO, NICE, BAD). These excerpts remain the copyright of their publishers and are reproduced here only so that the reported evaluation can be verified; consult the original documents for clinical use.
- No dataset images and no Derm7pt-derived files are distributed in this repository (Derm7pt is CC BY-NC-ND 4.0).
