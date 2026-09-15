# DERM-PACT

Code for **DERM-PACT Agent** (Dermoscopic Perception, Applicability, Confirmation, and Traceability), an agentic clinical decision support framework for dermoscopic melanoma assessment with selective referral and guideline-based support.

DERM-PACT connects multiscale, criterion-aware perception, calibrated selective referral, a versioned case record, executable guideline rules, evidence retrieval, and a bounded clinician-facing agent. A large language model coordinates conversational access to the case; separate rule-based checks verify every response and record change before display, and the learned probabilities stay fixed during dialogue.

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
    agent/        Controller, tools, prompts, providers, and rule-based checker
    memory.py     Case-based retrieval bank
    xai.py        Perturbation sensitivity
    record_v2.py, ledger_v2.py, gate_v2.py   Case record, ledger, and gate
    prototype/    Interactive command-line demo
    eval/         Agent, guideline, grounding, and adversarial evaluations
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
