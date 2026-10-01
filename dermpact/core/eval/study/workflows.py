"""Independently specified clinical-assertion/checklist episodes and real outcome scoring."""
import copy
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
from .common import ROOT, checked, load, write, result_path, slug, call_cost
from .. import applicability_benchmark_v2 as legacy
from dermpact.core.agent.redesign.clarification import ClarificationAgent, now
from dermpact.core.agent.redesign.amendment import AmendmentAgent, prediction_identity
from dermpact.core.agent.redesign.evidence import EvidenceError
from dermpact.core.episode import derive_state
from dermpact.core.prototype.host import HostServices
from dermpact.core.record_v2 import AssertionInput, StructuredCaseRecordV2, build_record_from_oof

M, E, R = "clinician.melanoma_suspected", "clinician.equivocal", "site.rcm_available"
H, C = "patient.high_risk_status", "patient.high_total_nevus_count"
RULE3, RULE7, RULE6 = ("eado-mel-p1-2024:R" + s for s in ("3", "7", "6"))
CLARIFICATION = [
    {"name":"yes", "rule":RULE3, "truth":{M:True}, "state":"applicable"},
    {"name":"no", "rule":RULE3, "truth":{M:False}, "state":"not_applicable"},
    {"name":"uncertain", "rule":RULE3, "truth":{M:None}, "state":"applicability_unknown", "stop":"uncertain"},
    {"name":"unavailable", "rule":RULE3, "truth":{M:None}, "state":"applicability_unknown", "stop":"unavailable"},
    {"name":"two_prerequisites", "rule":RULE7, "truth":{E:True,R:True}, "state":"applicable"},
    {"name":"short_circuit", "rule":RULE6, "truth":{H:True,C:False}, "state":"not_applicable", "sufficient":[[C]]},
]
CORRECTIONS = ["fact_true", "fact_false", "cancel", "uncertain", "criterion", "criterion_cancel"]
PHRASES = {M:"melanoma is clinically suspected for this lesion", E:"this lesion is clinically equivocal",
           R:"reflectance confocal microscopy is available at this site", H:"this patient belongs to a melanoma high-risk group",
           C:"this patient has a high total nevus count"}


def reply_text(task, fact):
    value = task["truth"][fact]
    if task.get("stop") == "uncertain":
        return "I am uncertain whether " + PHRASES[fact] + "."
    if task.get("stop") == "unavailable":
        return "That assessment is unavailable and cannot be obtained at present."
    return ("Yes, I confirm that " if value else "No, I confirm that it is not the case that ") + PHRASES[fact] + "."


class RecordFactory:
    def __init__(self):
        self.data = legacy.resolve_inputs()
        self.rows = {c.group_id:i for i,c in enumerate(self.data.cases)}
        self.oof = legacy._load_checked_npz(legacy.STAGE2_OOF_PATH)
        self.oof_sha = legacy._sha256_file(legacy.STAGE2_OOF_PATH)
        self.host = HostServices.build(ROOT / "research/experiments/dermpact_agent_redesign/study_preparation_host")

    def build(self, case_id, initial=None):
        i = self.rows[case_id]; source_case = self.data.cases[i]
        assertions = {key: AssertionInput(value=value, actor="scenario_author", session_id="frozen_preparation",
            asserted_at="2026-09-23T00:00:00+00:00", statement="Authored scenario input, not an observed clinical finding") for key,value in (initial or {}).items()}
        return build_record_from_oof(case_id, diagnosis_logits=self.oof["derm_diagnosis_logits"][i].tolist(),
            melanoma_logit=float(self.oof["derm_melanoma_logits"][i]), criterion_logits=self.oof["derm_criteria_logits"][i].tolist(),
            oof_manifest={"path":str(legacy.STAGE2_OOF_PATH.relative_to(ROOT)),"sha256":self.oof_sha},
            clinician_assertions=assertions, observed_facts={"sex":source_case.facts.sex,"site":source_case.facts.site,"elevation":source_case.facts.elevation},
            evaluation_date=date(2026,9,23), created_at="2026-09-23T00:00:00+00:00", source_label="synthetic_workflow_fixture",
            catalogue=self.host.catalogue, engine=self.host.engine)

    def close(self):
        self.host.close(); self.oof.close()


def prepare_tasks(root, case_ids, split, factory):
    tasks = []
    for i, case_id in enumerate(case_ids[:30]):
        spec = copy.deepcopy(CLARIFICATION[i % len(CLARIFICATION)])
        task_id = case_id + "::clarification::" + spec["name"]
        record = factory.build(case_id)
        fixture = Path("fixtures") / split / (slug(task_id) + ".json")
        write(root / fixture, record.to_dict())
        tasks.append({**spec, "task_id":task_id,"case_id":case_id,"split":split,"family":"clarification","fixture":str(fixture),
            "request":"Determine whether the requested guideline rule applies. Ask for its missing clinical facts and explain the recorded result after confirmation.",
            "required_facts":list(spec["truth"]),"sufficient":spec.get("sufficient",[list(spec["truth"])]),
            "expected_outcome":"awaiting_information" if spec.get("stop") else "answered"})
        kind = CORRECTIONS[i % len(CORRECTIONS)]
        initial = {M:True} if kind == "fact_false" else {M:False} if kind == "fact_true" else {}
        record = factory.build(case_id, initial)
        if kind.startswith("criterion"):
            before = record.perception.criterion_values["pigment_network"]
            value = "typical" if before == "atypical" else "atypical"
            request = f"As the reviewing clinician, correct the pigment network annotation to {value}."
            change = {"kind":"criterion_value","field":"pigment_network","value":value}
        else:
            value = kind != "fact_false"
            request = ("As the clinician I confirm melanoma is clinically suspected for this lesion. Update that clinical assertion." if value else
                       "As the clinician I do not suspect melanoma for this lesion. Correct the clinical suspicion assertion to no.")
            change = {"kind":"clinician_fact","field":M,"value":value}
        if kind == "uncertain": request = "The pigment network might be atypical, but I am not sure whether it should be changed."
        task_id = case_id + "::correction::" + kind
        fixture = Path("fixtures") / split / (slug(task_id) + ".json")
        write(root / fixture, record.to_dict())
        tasks.append({"task_id":task_id,"case_id":case_id,"split":split,"family":"correction","name":kind,"fixture":str(fixture),
            "request":request,"change":change,"cancel":"cancel" in kind,"uncertain":kind=="uncertain",
            "expected_outcome":"cancelled" if "cancel" in kind else "awaiting_information" if kind=="uncertain" else "answered"})
    return tasks


def open_task_host(root, task, state_dir):
    host = HostServices.build(state_dir)
    record = StructuredCaseRecordV2.from_dict(checked(root / task["fixture"]))
    host.store.create_case(record)
    verdict = host.gate.verify(record, expected_case_id=task["case_id"], now=date.today(), lineage=[record])
    if not verdict.passed:
        raise ValueError("Prepared workflow fixture failed source/lineage gate: " + ",".join(f.code for f in verdict.failures))
    return host, record


def providers(trace):
    values = [t["provider"] for t in trace if "provider" in t]
    for entry in trace:
        if entry.get("stage") == "explanation_check":
            values.extend(entry["verdict"].get("providers", []))
    return values


def score_state(host, initial, expected_values, expected_criteria, expected_commits, unavailable=()):
    from dermpact.criteria.schema import CRITERIA_BY_NAME, INDETERMINATE
    case = host.store.load_case(initial.identity.case_id); record = case.current_record
    low=sum(spec.points for key,spec in CRITERIA_BY_NAME.items() if expected_criteria[key] in spec.scoring_values)
    high=low+sum(spec.points for key,spec in CRITERIA_BY_NAME.items() if expected_criteria[key]==INDETERMINATE)
    checks = {"assertion_values":{k:v["value"] for k,v in record.clinician_assertions.items()}==expected_values,
        "criterion_values":record.perception.criterion_values==expected_criteria,
        "predictions_unchanged":prediction_identity(initial)==prediction_identity(record),
        "independent_checklist_arithmetic":record.checklist.score_min==low and record.checklist.score_max==high,
        "version_count":record.identity.record_version==len(case.records)==initial.identity.record_version+expected_commits,
        "commits":sum(e["type"]=="committed" for e in case.ledger_events)==expected_commits,
        "unavailable":sorted(derive_state("episode-"+initial.identity.case_id,initial.identity.case_id,case.episode_events).unavailable_facts)==sorted(unavailable),
        "source_gate":host.gate.verify(record,expected_case_id=initial.identity.case_id,now=date.today(),lineage=case.records).passed}
    return checks, record


def generate(root, task, arm="full"):
    path = result_path(root,"workflows",task["split"],arm,task["task_id"])
    if path.exists(): return checked(path)
    state_dir = path.parent / (path.stem+"_state")
    if state_dir.exists(): raise RuntimeError("Interrupted workflow state exists; inspect/recover it, never blindly repeat writes")
    host, initial = open_task_host(root,task,state_dir)
    cls = ClarificationAgent if task["family"]=="clarification" else AmendmentAgent
    agent = cls(host,case_id=task["case_id"],session_id=slug(task["task_id"]),actor="simulated_clinician")
    expected={k:v["value"] for k,v in initial.clinician_assertions.items()}; criteria=dict(initial.perception.criterion_values)
    steps, questions, commits, error = [], [], 0, None
    started=time.monotonic(); result={}
    def save(action, result, user_input=None):
        steps.append({"action":action,"input":user_input,"result":result})
        write(path.parent/(path.stem+f"_step_{len(steps):02d}.json"),{"step":steps[-1],"checkpoint":agent.snapshot()})
    try:
        result=agent.begin(task["request"],rule_id=task.get("rule",""));save("begin",result,task["request"])
        if task["family"]=="clarification":
            for _ in range(6):
                if result["outcome"]!="awaiting_information" or "fact" not in result: break
                fact=result["fact"];questions.append(fact)
                if fact not in task["truth"]: raise ValueError("Asked a fact outside the independently specified requested rule")
                text=reply_text(task,fact);before=host.store.load_case(task["case_id"]).current_record.record_sha256
                result=agent.respond(text);save("reply",result,text)
                if host.store.load_case(task["case_id"]).current_record.record_sha256!=before: raise ValueError("Reply committed without confirmation")
                if task.get("stop"): break
                if result.get("proposed_value") is not task["truth"][fact]: raise ValueError("Preview interpretation differs from scripted clinician intent")
                result=agent.confirm(proposal_id=result["proposal_id"],preview_sha256=result["preview_sha256"],approved=True)
                commits+=1;expected[fact]=task["truth"][fact];save("explicit_confirmation",result,True)
        elif result["outcome"]=="awaiting_confirmation" and not task["uncertain"]:
            before=host.store.load_case(task["case_id"]).current_record.record_sha256
            if before!=initial.record_sha256: raise ValueError("Correction request committed without confirmation")
            change=task["change"]
            if agent.pending.kind!=change["kind"] or agent.pending.field!=change["field"] or agent.pending.after!=change["value"]:
                raise ValueError("Correction preview differs from authored intent")
            result=agent.confirm(proposal_id=result["proposal_id"],preview_sha256=result["preview_sha256"],approved=not task["cancel"])
            if not task["cancel"]:
                commits=1
                (expected if change["kind"]=="clinician_fact" else criteria)[change["field"]]=change["value"]
            save("cancel" if task["cancel"] else "explicit_confirmation",result,not task["cancel"])
    except Exception as exc: error={"type":type(exc).__name__,"reason":str(exc)}
    unavailable=list(task["truth"]) if task.get("stop")=="unavailable" else []
    checks,record=score_state(host,initial,expected,criteria,commits,unavailable)
    if task["family"]=="correction":
        checks["cancellation_count"]=sum(e["type"]=="cancelled" for e in host.store.load_case(task["case_id"]).ledger_events)==int(task["cancel"])
    if task["family"]=="clarification":
        checks["rule_state"]=next(r["state"] for r in record.applicability["results"] if r["rule_id"]==task["rule"])==task["state"]
    elif not task["cancel"] and not task["uncertain"]:
        change=task["change"];checks["intended_change_saved"]=(record.clinician_assertions[change["field"]]["value"] if change["kind"]=="clinician_fact" else record.perception.criterion_values[change["field"]])==change["value"] and commits==1
    gold=set(task.get("required_facts",[]));asked=set(questions)
    precision=len(asked&gold)/len(asked) if asked else 0.0 if gold else None
    recall=max((len(asked&set(s))/len(s) for s in task.get("sufficient",[]) if s),default=None)
    calls=providers(agent.trace)
    output={"task_id":task["task_id"],"case_id":task["case_id"],"family":task["family"],"arm":arm,"split":task["split"],
        "outcome":result.get("outcome"),"task_success":error is None and result.get("outcome")==task["expected_outcome"] and all(checks.values()),
        "final_state_exact_match":all(checks.values()),"checks":checks,"error":error,"steps":steps,"trace":agent.trace,"calls":calls,
        "question_count":len(questions),"questions":questions,"question_precision":precision,"question_recall":recall,
        "question_f1":(2*precision*recall/(precision+recall) if precision+recall else 0.0) if precision is not None and recall is not None else None,
        "final_text":result.get("text",result.get("reason","")),"elapsed_s":time.monotonic()-started,"final_record_sha256":record.record_sha256}
    output["cost"]=call_cost(output);write(path,output);host.close();return output
