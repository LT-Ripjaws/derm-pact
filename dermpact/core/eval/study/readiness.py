"""Readiness is evidenced by operated development paths, never by a successful import."""
from pathlib import Path
from .common import CONFIG, checked, execution_guard, result_path, sha, write


def inspect_readiness(root):
    root=Path(root)
    execution_guard(root,"development")
    package=root
    reuse_path=package/"QUALIFICATION_REUSE.json"
    reuse=checked(reuse_path) if reuse_path.exists() else None
    if reuse:
        if reuse["lock_sha256"]!=sha(package/"lock.json"):
            raise ValueError("Qualification reuse belongs to a different execution freeze")
        for relative,expected in reuse["evidence_sha256"].items():
            if sha(package/relative)!=expected:
                raise ValueError("Qualification evidence changed: "+relative)
        root=package/"qualification"
        previous=checked(root/"READY.json")
        if previous["status"]!="ready_for_evaluation" or previous["lock_sha256"]!=sha(root/"lock.json"):
            raise ValueError("Previous qualification was not ready or its lock changed")
        for relative in previous["evidence_files"]:
            if str(Path("qualification")/relative) not in reuse["evidence_sha256"]:
                raise ValueError("Missing bound qualification artifact: "+relative)
        for path in (root/"inputs").glob("*.json"):
            if sha(path)!=sha(package/"inputs"/path.name):
                raise ValueError("Reused qualification has different task inputs")
        if checked(package/"qualification/scoring_audit/CONTEXT_REPAIR_CHECK_V2.json")["status"]!="passed":
            raise ValueError("Corrected scoring path has not passed its targeted check")
    failures=[];evidence=[]
    final_files=[p for location in {root,package} for base in (location/"answers",location/"grades",location/"episodes") if base.exists()
                 for p in base.rglob("*.json") if "evaluation" in p.relative_to(base).parts]
    if final_files:failures.append("Final-result artifacts already exist; this is not preparation-only readiness")
    for arm in CONFIG["read_arms"]:
        tasks=checked(root/"inputs/read_development.json")
        answers=[];grades=[]
        for task in tasks:
            if arm in {"no_history","raw_history"} and task["family"]!="referral_and_comparison":continue
            p=result_path(root,"read","development",arm,task["task_id"])
            if p.exists():answers.append((p,checked(p)))
            p=result_path(root,"read","development",arm,task["task_id"],"grades")
            if p.exists():grades.append((p,checked(p)))
        if not answers:failures.append("No operated read condition: "+arm)
        if any(not result.get("record_unchanged",True) for _,result in answers):failures.append("Read mutation observed: "+arm)
        if not grades:failures.append("No grading/export path exercised: "+arm)
        for p,g in grades:
            if g.get("status")=="not_answered":continue
            if any(g.get("scores",{}).get(m,{}).get("status")!="scored" for m in ("faithfulness","factual_correctness_f1","factual_correctness_precision","factual_correctness_recall","response_relevancy")):
                failures.append("Incomplete standard metric grade: "+str(p.relative_to(root)))
            if (g.get("quality_rubric") or {}).get("status")!="scored":failures.append("Missing direct rubric grade: "+str(p.relative_to(root)))
        evidence.extend(str(p.relative_to(root)) for p,_ in answers+grades)
    for arm in CONFIG["guideline_arms"]:
        rows=[(p,checked(p)) for t in checked(root/"inputs/guidelines_development.json")
              if (p:=result_path(root,"guidelines","development",arm,t["task_id"])).exists()]
        if not rows:failures.append("No operated guideline arm: "+arm)
        if arm in {"forms","executable"} and any(r["correct"]!=r["eligible_decisions"] or r["malformed"] for _,r in rows):
            failures.append("Deterministic guideline/reference disagreement: "+arm)
        evidence.extend(str(p.relative_to(root)) for p,_ in rows)
    for family,expected in (("clarification",6),("correction",6)):
        rows=[(p,checked(p)) for t in checked(root/"inputs/workflows_development.json") if t["family"]==family
              and (p:=result_path(root,"workflows","development","full",t["task_id"])).exists()]
        if len({r["task_id"].split("::")[-1] for _,r in rows})<expected:failures.append("Workflow branch coverage incomplete: "+family)
        if any(not r["task_success"] for _,r in rows):failures.append("Observed workflow defect: "+family)
        evidence.extend(str(p.relative_to(root)) for p,_ in rows)
    for arm in CONFIG["persistence_arms"]:
        rows=[(p,checked(p)) for t in checked(root/"inputs/persistence_development.json")
              if (p:=result_path(root,"persistence","development",arm,t["task_id"])).exists()]
        if len({r["boundary"] for _,r in rows})<6:failures.append("Recovery boundary coverage incomplete: "+arm)
        # A measured decoder failure may be a valid result; unsafe mutation or transport/scoring failure is not ready.
        if any(not r.get("checks",{}).get("source_gate") for _,r in rows):failures.append("Recovery source integrity failure: "+arm)
        evidence.extend(str(p.relative_to(root)) for p,_ in rows)
    safety_tasks=checked(root/"inputs/safety_development.json")
    action_rows=[];text_families=set();valid_scored=set()
    for task in safety_tasks:
        for arm in (["full"] if task["kind"]=="action_attack" else CONFIG["safety_read_arms"]):
            p=result_path(root,"safety","development",arm,task["task_id"])
            if not p.exists():continue
            evidence.append(str(p.relative_to(root)))
            if task["kind"]=="action_attack":action_rows.append(checked(p));continue
            g=result_path(root,"safety","development",arm,task["task_id"],"grades")
            if not g.exists():failures.append("Safety judge path missing: "+str(g.relative_to(root)));continue
            verdict=checked(g)["verdicts"]["final"]
            if verdict["status"]!="scored":failures.append("Safety judgment unscored: "+str(g.relative_to(root)))
            if arm=="full" and task["kind"]=="text_attack":text_families.add(task["family"])
            if task["kind"]=="valid_read":valid_scored.add(arm)
            evidence.append(str(g.relative_to(root)))
    if len(text_families)<6:failures.append("Not all six text attack families have been operated")
    if len({r["family"] for r in action_rows})<4:failures.append("Action confirmation-attack coverage incomplete")
    if any(r.get("error") or r.get("attack_succeeded") for r in action_rows):failures.append("Action boundary failure or setup error remains")
    if valid_scored!=set(CONFIG["safety_read_arms"]):failures.append("Missing benign safety controls")
    if not (root/"exports/development/summary.json").exists():failures.append("Development table exports not built")
    manual=package/"MANUAL_REVIEW.json"
    if not manual.exists() or checked(manual).get("status")!="accepted_for_evaluation":failures.append("Final source/content/scope review not signed off")
    elif checked(manual).get("lock_sha256")!=sha(package/"lock.json"):
        failures.append("Manual review belongs to a different execution freeze")
    if reuse:
        evidence=[str(Path("qualification")/p) for p in evidence]+list(reuse["evidence_sha256"])
    report={"status":"ready_for_evaluation" if not failures else "not_ready","failures":failures,
            "evidence_files":sorted(set(evidence)),"lock_sha256":sha(package/"lock.json"),"final_evaluation_generated":False,
            "interpretation":"Readiness for the frozen research protocol, not clinical validation or guaranteed model superiority."}
    if reuse:
        report["qualification_scope"]="Prior operated development results plus the bound scoring amendment and targeted repair check; no fresh aggregate scores."
        report["qualification_reuse_sha256"]=sha(reuse_path)
    if not failures:write(package/"READY.json",report)
    return report
