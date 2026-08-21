"""Assemble notebooks/medguard_end_to_end.ipynb -- the light reviewer demo.

Structure follows the user's 2026-08-15 edit of the executed notebook: FOUR terse
sections, minimal prose, ONE agreeing case (derm7pt-12: classifier MEL 0.88,
checklist 4 >= 3 -- views agree) flowing through the whole notebook. Rebuild +
execute with:

    venv_win\\Scripts\\python.exe scripts/dev/build_reviewer_notebook.py
    venv_win\\Scripts\\python.exe -m nbconvert --to notebook --execute --inplace \\
        notebooks/medguard_end_to_end.ipynb

Zero paid API calls; offline; deterministic.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "notebooks" / "medguard_end_to_end.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip()))


md("## Environment check")
code("""
import json, os, sys
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
np.random.seed(20260815)

DATASET_AVAILABLE = (REPO / "release_v0" / "meta" / "meta.csv").exists()
EV = REPO / "research" / "evidence" / "derm7pt_v1"
print(f"repo: {REPO}")
print(f"Derm7pt dataset present: {DATASET_AVAILABLE}")
print(f"evidence dir present:    {EV.exists()}")
""")

md("## One image, end to end")
code("""
if DATASET_AVAILABLE:
    from IPython.display import Image as IPyImage, Markdown, display
    from medguard_v2.data.derm7pt import load_cases
    from medguard_v2.pipeline.predict_one import predict_one

    cases = {f"derm7pt-{c.case_num}": c for c in load_cases()}
    case = cases["derm7pt-12"]  # classifier and checklist AGREE: melanoma, score 4 >= 3
    display(IPyImage(filename=str(case.derm_path), width=360))
    pred = predict_one(case.derm_path, sex=case.sex, location=case.location,
                       elevation=case.elevation, case_id="derm7pt-12")
    print(f"diagnosis={pred.diagnosis}  confidence={pred.confidence:.2f}  "
          f"domain gated={pred.domain['gated']} (score {pred.domain['score']:.3f})")
    display(Markdown(pred.report_markdown))
else:
    print("Dataset absent - see the committed outputs of this cell.")
""")

md("## Generation under verification")
code("""
if DATASET_AVAILABLE:
    from medguard_v2.evidence.summary import generate_case_brief
    from medguard_v2.evidence.render import render_markdown
    from IPython.display import Markdown, display

    c = pred.contract  # same agreeing case as above
    good = (f"The 7-point score is {c.score.score_min}, at or above the excision "
            f"threshold of 3, so the checklist-implied action is {c.criteria_implied_action}. "
            "The pattern model and the checklist agree on this case.")
    def brief_chat(messages):
        return good

    brief = generate_case_brief(c, brief_chat)
    print(f"verified={brief.verified} attempts={brief.attempts} fallback={brief.fallback_used}")
    print("(measured on the real model: 203/203 briefs verified - case_brief_measurement.json)")
    display(Markdown(render_markdown(c, brief)))
else:
    print("Dataset absent - see committed outputs.")
""")

md("## The measured system")
code("""
import pandas as pd
S = json.loads((EV / "paper_metrics_summary.json").read_text(encoding="utf-8"))

perc = pd.DataFrame([
    ["Derm7pt test (395)", "5-class acc", S["perception_derm7pt"]["classifier_5class"]["argmax"]["accuracy"]],
    ["Derm7pt test", "MEL AUROC", S["perception_derm7pt"]["melanoma_auroc"]["point"]],
    ["Derm7pt test", "criteria binary-F1 (DermAgent 0.6506)", S["perception_derm7pt"]["criteria_aggregate"]["binary_concept_f1_macro"]],
    ["BCN test (712)", "MEL AUROC (promoted recipe)", S["perception_bcn"]["promoted_recipe_post_hoc"]["mel_auroc"]],
    ["HAM E3 (1595)", "7-class acc (promoted recipe)", S["perception_ham"]["promoted_recipe_post_hoc"]["e3_seven_class"]["accuracy"]],
    ["HAM sealed (1495)", "5-group acc (post-hoc)", S["perception_ham"]["promoted_recipe_post_hoc"]["sealed_five_group"]["accuracy"]],
], columns=["dataset", "metric", "value"])
display(perc)

ep = S["agent_eval"]["endpoints"]
agent_tbl = pd.DataFrame([
    ["counterfactual accuracy (exact)", ep["counterfactual_accuracy"]["correct"]],
    ["citation resolvability", ep["citation_resolvability"]["resolvability"]],
    ["refusal correctness", ep["refusal_correctness"]["overall"]],
    ["absence disclosure", ep["absence_disclosure"]["states_absence"]],
    ["management fabrication", ep["management_fabrication"]["rate"]],
    ["case briefs verified", S["agent_eval"]["case_briefs"]["verified_rate"]],
], columns=["agent endpoint", "value"])
display(agent_tbl)

rg = S["ragas"]
display(pd.DataFrame([
    ["faithfulness (median-of-3 judge)", rg["overall_faithfulness"]],
    ["guideline-grounded faithfulness", rg["by_grounding"]["guideline"]["mean"]],
    ["combined (deduplicated)", rg["combined_faithfulness"]],
    ["answer relevancy", rg["answer_relevancy"]],
], columns=["RAGAS", "value"]))
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3 (venv_win)", "language": "python", "name": "python3",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
