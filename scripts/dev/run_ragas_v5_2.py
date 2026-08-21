"""Score the v5.2 agent eval report with the RAGAS judge (faithfulness + answer_relevancy).

The v5.2 prompt adds a retrieval nudge on top of v5/v5.1's verification loop, intended
to push more `guideline`/`why` questions through `search_guidelines` instead of
answering off the contract alone. That should grow the guideline-scorable pool well
past v4's 34 samples.

This is a minimal driver around ``run_ragas()`` in ``rerun_agent_and_ragas.py``. It
does NOT call ``run_agent_eval()`` and does NOT regenerate the agent report --
``rerun_agent_and_ragas.py``'s own ``__main__`` always runs the agent first, and
that CLI has no flag to skip it (only --model/--limit/--cohort), so this driver
imports the judging function directly and calls only that (same pattern as
``run_ragas_v4.py``).

Input:  research/evidence/derm7pt_v1/agent_eval_report_v5_2.json (already generated;
        203 cases, 599 questions, prompt_version clinician-agent-v5.2 -- NOT
        regenerated here)
Output: research/evidence/derm7pt_v1/agent_ragas_metrics_v5_2.json (new file; does
        NOT touch agent_ragas_metrics_v3.json or agent_ragas_metrics_v4.json)

Judge model: medguard.config.JUDGE_MODEL (the same default the v3/v4 runs used --
run_ragas() ignores its own `model` parameter and always judges with
config.JUDGE_MODEL internally, so this driver reports that value rather than
inventing one).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Load rerun_agent_and_ragas.py by file path (not `import scripts.dev....`) so this
# works regardless of whether scripts/ and scripts/dev/ are set up as packages.
_spec = importlib.util.spec_from_file_location(
    "rerun_agent_and_ragas", REPO / "scripts" / "dev" / "rerun_agent_and_ragas.py"
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
run_ragas = _module.run_ragas

from medguard import config  # noqa: E402


def main() -> None:
    agent_results_path = REPO / "research/evidence/derm7pt_v1/agent_eval_report_v5_2.json"
    out = REPO / "research/evidence/derm7pt_v1/agent_ragas_metrics_v5_2.json"
    v4_path = REPO / "research/evidence/derm7pt_v1/agent_ragas_metrics_v4.json"

    if not agent_results_path.exists():
        raise SystemExit(f"missing agent report: {agent_results_path}")

    print(f"agent report : {agent_results_path}")
    print(f"output       : {out}")
    print(f"judge model  : {config.JUDGE_MODEL}  (config.JUDGE_MODEL; run_ragas's own "
          f"`model` arg is accepted but unused -- it always judges with config.JUDGE_MODEL)")

    result = run_ragas(agent_results_path, out, config.JUDGE_MODEL)

    reported_version = json.loads(agent_results_path.read_text(encoding="utf-8")).get("prompt_version")
    print(f"\n  prompt_version (from agent report): {reported_version}")

    if v4_path.exists():
        v4 = json.loads(v4_path.read_text(encoding="utf-8"))
        print("\n=== v4 -> v5.2 sample comparison ===")
        print(f"  n_guideline_scorable : {v4.get('n_guideline_scorable')} -> {result.get('n_guideline_scorable')}")
        print(f"  faithfulness         : {v4.get('faithfulness')} -> {result.get('faithfulness')}")
        print(f"  answer_relevancy     : {v4.get('answer_relevancy')} -> {result.get('answer_relevancy')}")
        v4_by_g = v4.get("faithfulness_by_grounding", {})
        v52_by_g = result.get("faithfulness_by_grounding", {})
        for g in ("guideline", "mixed"):
            print(f"  by_grounding[{g}]     : {v4_by_g.get(g, {}).get('mean', 'N/A')} -> {v52_by_g.get(g, {}).get('mean', 'N/A')}")


if __name__ == "__main__":
    main()
