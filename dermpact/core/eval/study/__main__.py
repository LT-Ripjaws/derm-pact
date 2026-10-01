"""Prepare, inspect, generate, score and export one suite at a time."""
import argparse
import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF","0")
os.environ.setdefault("RAGAS_DO_NOT_TRACK","true")
from .common import DEFAULT, CONFIG, checked, write, result_path, execution_guard, sha, stamp

SUITES=("read","guidelines","workflows","persistence","safety")


def arms_for(suite,task):
    if suite=="read":return [a for a in CONFIG["read_arms"] if a not in {"no_history","raw_history"} or task["family"]=="referral_and_comparison"]
    if suite=="guidelines":return CONFIG["guideline_arms"]
    if suite=="workflows":return ["full"]
    if suite=="persistence":return CONFIG["persistence_arms"]
    return ["full"] if task["kind"]=="action_attack" else CONFIG["safety_read_arms"]


async def execute(args):
    from . import read,guidelines,workflows,persistence,safety
    execution_guard(args.root,args.split,allow_final=args.allow_final_evaluation)
    tasks=checked(args.root/"inputs"/(args.suite+"_"+args.split+".json"))
    if args.split=="evaluation" and (args.task_id or args.family):
        raise ValueError("Final task order is frozen; use limit/resume, not outcome-dependent task selection")
    if args.task_id:tasks=[t for t in tasks if t["task_id"]==args.task_id]
    if args.family:tasks=[t for t in tasks if t.get("family")==args.family or t.get("boundary")==args.family or t.get("name")==args.family]
    allowed=set(a for t in tasks for a in arms_for(args.suite,t))
    if args.arms and not set(args.arms)<=allowed:raise ValueError("Unknown/inapplicable arm")
    selected=[]
    phase="answers" if args.command=="run" else "grades"
    for task in tasks:
        arms=[a for a in arms_for(args.suite,task) if not args.arms or a in args.arms]
        missing=[a for a in arms if not result_path(args.root,args.suite,args.split,a,task["task_id"],phase).exists()]
        if missing:selected.append((task,missing))
    if args.limit:selected=selected[:args.limit]
    print(json.dumps({"suite":args.suite,"split":args.split,"phase":phase,"task_count":len(selected),
        "condition_count":sum(len(a) for _,a in selected),"final_generation":args.split=="evaluation" and args.command=="run"}),flush=True)
    if args.dry_run:return
    if args.command=="run" and os.environ.get("DERMPACT_ALLOW_LIVE_LLM")!="1":
        raise RuntimeError("Set DERMPACT_ALLOW_LIVE_LLM=1 before generation; no tasks were attempted")
    lease=args.root/"RUNNING.json"
    try:
        fd=os.open(lease,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("A study job lease exists. Inspect RUNNING.json and its PID before clearing a dead-process lease.") from None
    with os.fdopen(fd,"w",encoding="utf-8") as stream:json.dump({"pid":os.getpid(),"started_at":stamp(),"suite":args.suite,"split":args.split,"phase":phase},stream)
    try:
        for task,arms in selected:
            for arm in arms:
                if args.command=="run":
                    runner={"read":read,"guidelines":guidelines,"workflows":workflows,"persistence":persistence,"safety":safety}[args.suite]
                    result=runner.generate(args.root,task,arm)
                elif args.suite=="read": result=await read.score(args.root,task,arm)
                elif args.suite=="safety" and task["kind"]!="action_attack":result=await safety.score_text(args.root,task,arm)
                else:
                    answer=checked(result_path(args.root,args.suite,args.split,arm,task["task_id"]))
                    result={"task_id":task["task_id"],"case_id":task["case_id"],"arm":arm,"status":"deterministic_outcome_scored",
                        "answer_sha256":sha(result_path(args.root,args.suite,args.split,arm,task["task_id"])),
                        **{k:answer[k] for k in ("task_success","final_state_exact_match","attack_succeeded","correct","eligible_decisions") if k in answer}}
                    write(result_path(args.root,args.suite,args.split,arm,task["task_id"],"grades"),result)
                print(task["task_id"],arm,result.get("outcome",result.get("status",result.get("error"))),flush=True)
        execution_guard(args.root,args.split,allow_final=args.allow_final_evaluation)
    finally:
        lease.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","plan","run","score","export","readiness"])
    parser.add_argument("--root",type=Path,default=DEFAULT)
    parser.add_argument("--suite",choices=SUITES,default="read")
    parser.add_argument("--split",choices=["development","evaluation"],default="development")
    parser.add_argument("--arms",nargs="+")
    parser.add_argument("--limit",type=int,default=1,help="Number of not-yet-completed tasks; 0 means all")
    parser.add_argument("--task-id")
    parser.add_argument("--family")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--allow-final-evaluation",action="store_true")
    args=parser.parse_args();args.root=args.root.resolve()
    if args.limit<0:parser.error("limit must be nonnegative")
    if args.command=="prepare":
        from .prepare import prepare
        prepare(args.root)
    elif args.command=="plan":
        lock=checked(args.root/"lock.json")
        print(json.dumps({"counts":lock["counts"],"configuration":checked(args.root/"configuration.json"),
            "status":"Prepared; final runs require READY.json and the explicit final-evaluation flag."},indent=2))
    elif args.command=="export":
        from .reporting import export
        export(args.root,args.split);print("Exported",args.root/"exports"/args.split)
    elif args.command=="readiness":
        from .readiness import inspect_readiness
        report=inspect_readiness(args.root)
        print(json.dumps(report,indent=2))
    else:asyncio.run(execute(args))


if __name__=="__main__":main()
