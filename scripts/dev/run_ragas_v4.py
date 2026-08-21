"""Score the v4 agent eval report with the RAGAS judge (faithfulness + answer_relevancy).

This is a minimal driver around ``run_ragas()`` in ``rerun_agent_and_ragas.py``. It
does NOT call ``run_agent_eval()`` and does NOT regenerate the agent report --
``rerun_agent_and_ragas.py``'s own ``__main__`` always runs the agent first, and
that CLI has no flag to skip it (only --model/--limit/--cohort), so this script
imports the judging function directly and calls only that.

Input:  research/evidence/derm7pt_v1/agent_eval_report_v4.json (already generated;
        203 cases, 599 questions -- NOT regenerated here)
Output: research/evidence/derm7pt_v1/agent_ragas_metrics_v4.json (new file; does
        NOT touch agent_ragas_metrics_v3.json)

Judge model: medguard.config.JUDGE_MODEL (the same default the v3 run used --
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
    agent_results_path = REPO / "research/evidence/derm7pt_v1/agent_eval_report_v4.json"
    out = REPO / "research/evidence/derm7pt_v1/agent_ragas_metrics_v4.json"
    v3_path = REPO / "research/evidence/derm7pt_v1/agent_ragas_metrics_v3.json"

    if not agent_results_path.exists():
        raise SystemExit(f"missing agent report: {agent_results_path}")

    print(f"agent report : {agent_results_path}")
    print(f"output       : {out}")
    print(f"judge model  : {config.JUDGE_MODEL}  (config.JUDGE_MODEL; run_ragas's own "
          f"`model` arg is accepted but unused -- it always judges with config.JUDGE_MODEL)")

    result = run_ragas(agent_results_path, out, config.JUDGE_MODEL)

    if v3_path.exists():
        v3 = json.loads(v3_path.read_text(encoding="utf-8"))
        print("\n=== v3 -> v4 sample comparison ===")
        print(f"  n_guideline_scorable : {v3.get('n_guideline_scorable')} -> {result.get('n_guideline_scorable')}")
        print(f"  faithfulness         : {v3.get('faithfulness')} -> {result.get('faithfulness')}")
        print(f"  answer_relevancy     : {v3.get('answer_relevancy')} -> {result.get('answer_relevancy')}")
        v2_by_g = v3.get("faithfulness_by_grounding", {})
        v3_by_g = result.get("faithfulness_by_grounding", {})
        for g in ("guideline", "mixed"):
            print(f"  by_grounding[{g}]     : {v2_by_g.get(g, {}).get('mean', 'N/A')} -> {v3_by_g.get(g, {}).get('mean', 'N/A')}")


if __name__ == "__main__":
    main()
