"""Direct Ragas rubric scoring of the 15 frozen Luna answers; no generation."""
import argparse
import asyncio
import inspect
import json
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, median

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
from ragas.metrics.collections import DomainSpecificRubrics
from .codex_judge import CodexJudge
from .prepare_read_evaluation import ROOT, sha, write
from dermpact.core.agent.redesign.evidence import digest

BASE = ROOT / "research/experiments/dermpact_agent_redesign"
SOURCE = BASE / "luna_fixed_pilot_v2"
OUT = BASE / "luna_direct_rubric_v1"
RUBRICS = {
    "score1_description": "The response is irrelevant, substantially incorrect, or provides no useful answer to the requested task.",
    "score2_description": "The response addresses part of the question but has a major factual error, unsupported clinical claim, or significant omission of a requested part or necessary qualification.",
    "score3_description": "The response is broadly correct but has a material gap in requested detail, ambiguous scope, or distracting unrelated information that substantially reduces its usefulness.",
    "score4_description": "The response correctly covers the essential requested parts and necessary qualifications, with only a minor omission, imprecise wording, or limited unnecessary detail.",
    "score5_description": "The response accurately and clearly covers all requested parts and necessary qualifications, agrees with the supplied evidence and the reference in meaning, and contains no material omission or unsupported claim. Equivalent paraphrases are valid; supported absence resolves conditional requests. Extra reference wording is not itself a requirement unless needed to answer the question."
}


async def main(phase):
    assert version("ragas") == "0.4.1"
    tasks = json.loads((SOURCE / "tasks.json").read_text(encoding="utf-8"))
    assert len(tasks) == 15 and all(t["role"] == "development" for t in tasks)
    files = [SOURCE / n for n in ("tasks.json", "protocol.json", "summary.json")]
    files += sorted((SOURCE / "answers").glob("*.answer.json"))
    files += sorted((SOURCE / "grades").glob("*.grade.json"))
    assert len(files) == 33
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in files}
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__), ROOT / "dermpact/core/eval/codex_judge.py", Path(inspect.getfile(DomainSpecificRubrics)))}
    judge = CodexJudge(OUT / "calls", effort="low")
    judge.reuse_claims = False
    metric = DomainSpecificRubrics(llm=judge, with_reference=True, rubrics=RUBRICS)
    expected = {"input_sha256": inputs, "source_sha256": sources, "rubrics": RUBRICS,
        "judge": judge.configuration, "ragas_version": version("ragas"),
        "metric": "DomainSpecificRubrics(with_reference=True)", "range": [1, 5],
        "scope": "User-authorized replacement of the lossy read-answer completion check; all 15 saved development answers. No agent regeneration, answer/reference editing, or changes to other scores.",
        "interpretation": "Direct reference/evidence-based answer-quality rubric, not binary goal accuracy or a percentage. Same-family judging remains a limitation.",
        "calls": "One call per answer, 15 total; first phase scores the first actual task and finish reuses it. No best-of retries or threshold tuning.",
        "criteria_policy": "Rubrics are frozen before the first judgment, applied identically to all answers. Inspect feedback for unsupported deductions; retain raw scores, including errors.",
        "documentation": "https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/general_purpose/"}
    if (OUT / "protocol.json").exists():
        assert json.loads((OUT / "protocol.json").read_text(encoding="utf-8")) == expected, "Input/scoring protocol changed"
    else:
        write(OUT / "protocol.json", expected)
    rows = []
    for task in (tasks[:1] if phase == "first" else tasks):
        stem = task["case_id"] + "--" + task["family"]
        answer = json.loads((SOURCE / "answers" / (stem + ".answer.json")).read_text(encoding="utf-8"))
        path = OUT / "grades" / (stem + ".json")
        if path.exists():
            grade = json.loads(path.read_text(encoding="utf-8"))
        else:
            events = [json.loads(m["content"]) for m in answer["messages"] if m["role"] == "user"]
            contexts = [json.dumps(e["receipt"], ensure_ascii=False) for e in events if e.get("host_event") == "tool_result"]
            start = len(judge.calls)
            grade = {"task_id": task["task_id"], "case_id": task["case_id"], "family": task["family"],
                "response_sha256": digest(answer["final_text"]), "reference_sha256": digest(task["reference"]),
                "contexts_sha256": digest(contexts), "rubrics_sha256": digest(RUBRICS)}
            try:
                result = await metric.ascore(user_input=task["question"], response=answer["final_text"],
                    reference=task["reference"], retrieved_contexts=contexts)
                value = float(result.value)
                if value not in (1, 2, 3, 4, 5):
                    raise ValueError("rubric_score_out_of_range")
                grade.update(status="scored", score=int(value), feedback=result.reason)
            except Exception as exc:
                grade.update(status="unscored", score=None, error_type=type(exc).__name__)
            grade["judge_calls"] = judge.calls[start:]
            write(path, grade)
        rows.append({k: v for k, v in grade.items() if k != "judge_calls"})
        print(task["task_id"], grade["status"], grade["score"], grade.get("feedback", grade.get("error_type")), flush=True)
        if phase == "first" and grade["status"] != "scored":
            raise RuntimeError("First actual item could not be scored; do not start remaining items")
    assert inputs == {str(p.relative_to(ROOT)): sha(p) for p in files}, "Frozen pilot artifacts changed"
    if phase == "finish":
        groups = {}
        for family in ("all", *sorted({r["family"] for r in rows})):
            subset = [r for r in rows if family == "all" or r["family"] == family]
            values = [r["score"] for r in subset if r["status"] == "scored"]
            groups[family] = {"attempted_n": len(subset), "scored_n": len(values),
                "mean": fmean(values) if values else None, "median": median(values) if values else None,
                "distribution": {str(i): values.count(i) for i in range(1, 6)}}
        write(OUT / "summary.json", {"rows": rows, "groups": groups, "range": [1, 5],
            "original_artifacts_unchanged": True, "new_agent_generations": 0, "feedback_review_pending": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=["first", "finish"])
    asyncio.run(main(parser.parse_args().phase))
