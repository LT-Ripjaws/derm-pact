# DERM-PACT

Code for **DERM-PACT Agent** (Dermoscopic Perception, Applicability, Confirmation, and Traceability), an agentic clinical decision support framework for dermoscopic melanoma assessment with selective referral and guideline-based support.

DERM-PACT connects criterion-aware perception with calibrated selective referral, a versioned case record, on-demand guideline support, and a bounded clinician-facing agent. A text-only large language model reads the structured record and retrieved evidence through read-only tools, chooses its own guideline searches, asks the clinician for a decisive fact that is not recorded, and drafts answers that link each statement to its evidence. A separate verifier call checks each draft against its sources and the request, allowing one repair before a fixed fallback response. Record changes are only proposed by the agent and written by the host after explicit clinician confirmation, and structured checkpoints let an interrupted interaction resume. The learned probabilities stay fixed during dialogue.

## Repository layout

```
dermpact/
  perception/     Multiscale hybrid ViT tail: LoRA-adapted blocks, regional and
                  cross-modal attention, criterion residuals, melanoma + 7 criterion heads
  eval/           Perception training, calibration, and benchmark modules
  reliability/    Temperature scaling, uncertainty and view-consistency signals,
                  image quality, OOD advisory, and selective referral
  criteria/       Weighted seven-point checklist scoring
  record/         Versioned case record, amendments, verification gate, brief template
  guidelines/     Executable guideline rule engine, facts, and derivation
  pipeline/       Frozen guideline retriever and the LLM client
  core/           Clinical decision support integration layer
    agent/redesign/   The DERM-PACT agent evaluated in the paper: read-only answer
                      workflow (workflow.py), evidence binding (evidence.py), the
                      separate verifier call (verification.py), on-demand guideline
                      answering with clarification (guidance_agent.py, clarification.py),
                      confirmed amendments (amendment.py), the two LLM controls
                      (baselines.py), and model access (provider.py, codex_provider.py)
    agent/        Earlier controller, kept for the command-line prototype
    memory.py     Case-based retrieval bank
    xai.py        Perturbation sensitivity (proof of concept)
    record_v2.py, ledger_v2.py, gate_v2.py   Case record, ledger, and gate
    prototype/    Interactive command-line demo
    eval/         Evaluation code; eval/study/ holds the suites behind the paper's
                  agent tables, codex_judge.py and kilo_judge.py the RAGAS judges
  tests/          Unit and property tests
```

## Datasets

DERM-PACT is evaluated on four public dermoscopy datasets. Source images are **not** redistributed here; obtain them from the original custodians under their current license and access terms.

- Derm7pt — https://derm.cs.sfu.ca/Download.html (CC BY-NC-ND 4.0)
- HAM10000 — https://doi.org/10.7910/DVN/DBW86T
- BCN20000 — https://doi.org/10.6084/m9.figshare.24140028 (CC BY 4.0)
- PH2 — from its original custodians

## Guideline data

The executable guideline rules and the retrieval library are derived from clinical practice guidelines that are licensed as no-derivatives and cannot be redistributed. They are **not** included. Supply the encoded rule catalogue and retrieval corpus locally to run the guideline engine and retriever.

## Setup

```bash
pip install -r requirements.txt
```

Perception requires the DermLIP--PanDerm encoder and the trained model checkpoints, which are subject to their own access and license terms. Configuration is read from the environment (see `dermpact/config.py`); the agent and judge models are configured through `DERMPACT_*` variables. A credential is loaded only via the opt-in `config.load_env`, and its value is never printed.

The agent reaches the LLM either through an OpenAI-compatible gateway (`DERMPACT_API_KEY`, `DERMPACT_GATEWAY_BASE_URL`) or through the Codex command-line tool on `PATH` (`DERMPACT_CODEX_GENERATION_DIR` for its artifacts). No live model call is made unless `DERMPACT_ALLOW_LIVE_LLM=1` is set.

## Agent evaluations

`python -m dermpact.core.eval.study --help` lists the commands (`prepare`, `plan`, `run`, `score`, `export`, `readiness`) for one suite at a time. The `read` suite produced the matched read-only comparison and the historical-evidence ablation, and the `safety` suite the robustness evaluation. The `guidelines` suite is an earlier conformance evaluation that the paper does not report. The evaluation inputs are built from Derm7pt cases and guideline passages, which cannot be redistributed, so they are not included.

## Tests

```bash
python -m pytest dermpact/tests -q
```

or the dependency-light runner:

```bash
python -m dermpact.tests.run_tests
```

## License

Code is released under the MIT License (see `LICENSE`). Datasets, model checkpoints, and guideline documents remain subject to their original terms.
