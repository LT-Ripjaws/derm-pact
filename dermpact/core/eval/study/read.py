"""Agreed read conditions with paired output-policy ablations and exact contexts."""
import copy
from dataclasses import asdict

from .common import BUNDLE, GENERATOR, VERIFIER, checked, load, write, result_path, call_cost
from ..read_reference_bundle import SnapshotEvidenceSource
from ..structured_read_generator import StructuredReadGenerator
from ..read_evaluation_context import read_evaluation_contexts as contexts
from dermpact.core.agent.redesign.baselines import ReadBaseline, common_case_input
from dermpact.core.agent.redesign.evidence import EvidenceError
from dermpact.core.agent.redesign.evidence import digest
from dermpact.core.agent.redesign.provider import JsonGatewayClient
from dermpact.core.agent.redesign.verification import SemanticVerifier, SemanticVerdict
from dermpact.core.agent.redesign.workflow import ReadAgent, ReadSettings

_SCORING = {}


class ExposureSource:
    def __init__(self, source, no_history=False):
        self.source, self.no_history = source, no_history

    def identity(self):
        return self.source.identity()

    def tools(self):
        available = self.source.tools()
        if self.no_history:
            available.pop("retrieve_case_comparisons", None)
        return available

    def read(self, tool, args):
        if tool not in self.tools():
            raise EvidenceError("tool_not_permitted_in_this_condition")
        return self.source.read(tool, args)


def paired_variant(full, arm):
    initial = full.get("initial_result")
    if not initial:
        return {**copy.deepcopy(full), "arm": arm, "paired_from": "full", "paired_initial_unavailable": True}
    result = {**copy.deepcopy(full), **copy.deepcopy(initial), "arm": arm, "paired_from": "full", "repairs": 0}
    result["verifications"] = []
    result["settings"] = {**result["settings"], "semantic_verification": arm == "check_only", "repair_allowed": False}
    if arm == "check_only":
        first = full.get("verifications", [{}])[0]
        providers = first.get("providers", [first["provider"]] if first.get("provider") else [])
        result["calls"] += [{"stage": "verify", **p} for p in providers]
        result["verifications"] = [first]
        result["elapsed_s"] = first.get("elapsed_until_check_s")
        if first.get("ok"):
            result["outcome"] = "answered"
        else:
            result.update(outcome="safe_fallback", final_text="A verified explanation could not be produced. Review the recorded evidence.")
    return result


def generate(root, task, arm, storage_suite="read"):
    path = result_path(root, storage_suite, task["split"], arm, task["task_id"])
    if path.exists():
        return checked(path)
    if arm in {"no_verification", "check_only"}:
        full = generate(root, task, "full", storage_suite)
        result = paired_variant(full, arm)
    else:
        source = SnapshotEvidenceSource(task["case_id"], role=task["split"], allow_evaluation=task["split"] == "evaluation", bundle=BUNDLE)
        source = ExposureSource(source, no_history=arm == "no_history")
        before = source.identity()
        case_input = common_case_input(source)
        if arm == "no_history":
            case_input["data"]["information_access"] = "Historical retrieval is disabled in this condition; missing access is not evidence that no matches exist."
        if arm in {"case_only", "standard_rag"}:
            docs = checked(root / "corpora" / task["split"] / (task["case_id"] + ".json")) if arm == "standard_rag" else ()
            result = ReadBaseline(source, arm=arm, documents=docs).run(task["question"], case_input=case_input)
        else:
            verifier = SemanticVerifier(JsonGatewayClient(VERIFIER, allow_live=True, response_schema=SemanticVerdict.model_json_schema()))
            settings = ReadSettings(comparison_representation="raw" if arm == "raw_history" else "typed")
            result = ReadAgent(source, StructuredReadGenerator(GENERATOR), verifier, settings).run(task["question"], case_input=case_input)
        result["record_unchanged"] = source.identity() == before
        if not result["record_unchanged"]:
            raise AssertionError("Read condition mutated its frozen source")
    result.update(task_id=task["task_id"], case_id=task["case_id"], split=task["split"], family=task["family"], arm=arm)
    result["tool_metrics_applicable"] = task.get("family") == "referral_and_comparison" and arm != "no_history"
    result["cost"] = call_cost(result)
    write(path, result)
    return result


async def score(root, task, arm):
    from importlib.metadata import version
    from ragas.metrics.collections import DomainSpecificRubrics
    from ..kilo_judge import KiloJudge
    from ..standard_read_metrics import StandardReadScorer
    from ..score_saved_read_rubric import RUBRICS
    path = result_path(root, "read", task["split"], arm, task["task_id"], "grades")
    if path.exists():
        return checked(path)
    result = checked(result_path(root, "read", task["split"], arm, task["task_id"]))
    if result["outcome"] not in {"answered", "answered_after_repair", "answered_unverified"}:
        grade = {"task_id": task["task_id"], "case_id": task["case_id"], "family": task["family"], "arm": arm,
                 "outcome": result["outcome"], "status": "not_answered", "scores": {}, "quality_rubric": None,
                 "reason": "Failure/fallback remains in completion denominator; do not assign perfect faithfulness to a claimless output."}
        write(path, grade)
        return grade
    assert version("ragas") == "0.4.1"
    signature=digest({"lock":checked(root/"lock.json")["source_sha256"],"task":task,"answer":result["final_text"],
        "contexts":contexts(result),"acquisition":result["settings"].get("acquisition"),
        "commands":[e.get("command") for e in result["trace"] if e.get("node") in {"generate","repair"}],
        "tool_metrics_applicable":result.get("tool_metrics_applicable",True)})
    cache=root/"grade_cache"/(signature+".json")
    if cache.exists():
        grade=checked(cache);grade.update(arm=arm,outcome=result["outcome"],agent_outcome=result["outcome"],score_reused_from=str(cache.relative_to(root)))
        write(path,grade);return grade
    session_key=str(root.resolve())
    if session_key not in _SCORING:
        judge=KiloJudge(root/"judge_calls"/"read",effort="low")
        scorer=StandardReadScorer(judge=judge,factual_judge=judge,atomicity="low")
        scorer.metrics.pop("agent_goal_accuracy")
        _SCORING[session_key]=(judge,scorer)
    judge,scorer=_SCORING[session_key]
    partial_dir=path.parent/(path.stem+"_metrics")
    completed={};previous_key=None
    if partial_dir.exists():
        for saved in partial_dir.glob("*.json"):
            item=checked(saved)
            if previous_key is not None and previous_key!=item["cache_key"]:raise ValueError("Mixed partial metric configurations")
            previous_key=item["cache_key"];completed[item["metric"]]=item["result"]
            for call in item["result"].get("judge_calls",[]):judge.cache[call["prompt_sha256"]]=call["output"]
    def save_metric(key,name,value):write(partial_dir/(name+".json"),{"cache_key":key,"metric":name,"result":value})
    grade = await scorer.score(result, task,completed_scores=completed,expected_cache_key=previous_key,metric_sink=save_metric)
    metric = DomainSpecificRubrics(llm=judge, with_reference=True, rubrics=RUBRICS)
    start = len(judge.calls)
    try:
        value = await metric.ascore(user_input=task["question"], response=result["final_text"],
                                   reference=task["reference"], retrieved_contexts=contexts(result))
        if float(value.value) not in (1, 2, 3, 4, 5):
            raise ValueError("rubric_out_of_range")
        rubric = {"status": "scored", "value": float(value.value), "reason": value.reason}
    except Exception as exc:
        rubric = {"status": "unscored", "value": None, "error_type": type(exc).__name__}
    grade.update(task_id=task["task_id"], case_id=task["case_id"], family=task["family"], arm=arm,
                 outcome=result["outcome"], quality_rubric=rubric, rubric_judge_calls=judge.calls[start:])
    write(cache,grade)
    write(path, grade)
    return grade
