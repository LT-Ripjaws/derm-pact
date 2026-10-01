"""Coverage-aware aggregate tables, paired case-level statistics and LaTeX exports."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from .common import checked, result_path, CONFIG, load, write
from .guidelines import STATES

LABELS={"case_only":"Luna only","standard_rag":"Standard RAG","full":"DERM-PACT",
        "no_verification":"Without checking/repair","check_only":"Checking without repair",
        "no_history":"Without historical retrieval","raw_history":"Ordinary historical records",
        "constant_unknown":"Constant unknown","naive_rag":"Luna + retrieved guidelines",
        "llm_rules":"Luna + exact rules","forms":"Independent forms","executable":"Executable rules",
        "structured":"Structured checkpoint","transcript":"Transcript reconstruction"}


def estimate(values, inference=True):
    """Equal case-weight mean and case-clustered bootstrap interval; None stays missing."""
    groups=defaultdict(list)
    for case_id,value in values:
        if value is not None and np.isfinite(float(value)): groups[case_id].append(float(value))
    samples=np.array([np.mean(group) for group in groups.values()])
    if not len(samples): return {"mean":None,"ci95":None,"cases":0,"observations":0}
    interval=None
    if inference and len(samples)>1:
        rng=np.random.default_rng(CONFIG["bootstrap_seed"])
        boot=np.mean(rng.choice(samples,size=(CONFIG["bootstrap_repetitions"],len(samples)),replace=True),axis=1)
        interval=np.quantile(boot,[.025,.975]).tolist()
    return {"mean":float(samples.mean()),"ci95":interval,"cases":len(samples),"observations":sum(map(len,groups.values()))}


def paired(left,right,metric,inference=True):
    values=defaultdict(list)
    for task_id in set(left)&set(right):
        a,b=left[task_id],right[task_id]
        if a.get(metric) is not None and b.get(metric) is not None:
            values[a["case_id"]].append(a[metric]-b[metric])
    case_differences=[(key,float(np.mean(value))) for key,value in values.items()]
    result=estimate(case_differences,inference=inference)
    result["paired_tasks"]=sum(map(len,values.values()))
    result["p_value"]=None
    if inference and len(values)>1:
        diff=np.array([v for _,v in case_differences]);rng=np.random.default_rng(CONFIG["bootstrap_seed"])
        randomized=(rng.choice([-1,1],size=(CONFIG["bootstrap_repetitions"],len(diff)))*diff).mean(axis=1)
        result["p_value"]=float((1+(np.abs(randomized)>=abs(diff.mean())-1e-15).sum())/(len(randomized)+1))
    return result


def rate(values,inference=True):
    """Ordinary success/attempt proportion, with case-clustered ratio resampling."""
    groups=defaultdict(list)
    for case_id,value in values:
        if value is not None:groups[case_id].append(float(value))
    if not groups:return {"mean":None,"ci95":None,"cases":0,"observations":0}
    totals=np.array([sum(v) for v in groups.values()]);sizes=np.array([len(v) for v in groups.values()])
    interval=None
    if inference and len(groups)>1:
        ids=np.random.default_rng(CONFIG["bootstrap_seed"]).integers(len(groups),size=(CONFIG["bootstrap_repetitions"],len(groups)))
        interval=np.quantile(totals[ids].sum(axis=1)/sizes[ids].sum(axis=1),[.025,.975]).tolist()
    return {"mean":float(totals.sum()/sizes.sum()),"ci95":interval,"cases":len(groups),"observations":int(sizes.sum())}


def metric_value(grade,name):
    if name=="quality_rubric": return (grade.get(name) or {}).get("value")
    return grade.get("scores",{}).get(name,{}).get("value")


def format_value(value):
    if value is None: return "PENDING"
    if isinstance(value,float): return f"{value:.3f}"
    if isinstance(value,dict) and "mean" in value:
        if value["mean"] is None:return "PENDING"
        text=f"{value['mean']:.3f}"
        if value["ci95"]:text+=f" [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]"
        return text
    return str(value)


def write_table(directory,name,headers,rows):
    headers=[h.replace("_"," ") for h in headers]
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/(name+".csv")).open("w",encoding="utf-8",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)
    markdown="| "+" | ".join(headers)+" |\n|"+"---|"*len(headers)+"\n"
    markdown+="".join("| "+" | ".join(map(format_value,row))+" |\n" for row in rows)
    (directory/(name+".md")).write_text(markdown,encoding="utf-8")
    def escape(text):
        return str(text).replace("&",r"\&").replace("%",r"\%").replace("_",r"\_").replace("#",r"\#")
    latex="\\begin{tabularx}{\\textwidth}{@{}X"+">{\\centering\\arraybackslash}X"*(len(headers)-1)+"@{}}\n\\toprule\n"
    latex+=" & ".join(map(escape,headers))+" \\\\\n\\midrule\n"
    latex+="".join(" & ".join(escape(format_value(v)) for v in row)+" \\\\\n" for row in rows)
    latex+="\\bottomrule\n\\end{tabularx}\n"
    (directory/(name+".tex")).write_text(latex,encoding="utf-8")


def export(root,split):
    out=Path(root)/"exports"/split
    summarize=lambda values:estimate(values,inference=split=="evaluation")
    metrics=["faithfulness","factual_correctness_precision","factual_correctness_recall","factual_correctness_f1","response_relevancy","quality_rubric"]
    inventories={suite:checked(root/"inputs"/(suite+"_"+split+".json")) for suite in ("read","guidelines","workflows","persistence","safety")}
    summary={"split":split,"scope":"Observed results only; pending rows are not zeros. Read means use equal case weights; intervals resample cases.","read":{},"guidelines":{},"workflows":{},"persistence":{},"safety":{},"metric_corrections":[]}
    main_rows=[];cost_rows=[];scores_by_arm={};comparison_rows=[];family_rows=[];ci_rows=[];tool_rows=[];verification_rows=[]
    for arm in CONFIG["read_arms"]:
        tasks=[t for t in inventories["read"] if arm not in {"no_history","raw_history"} or t["family"]=="referral_and_comparison"]
        answers=[];grades=[];paired_scores={}
        for task in tasks:
            p=result_path(root,"read",split,arm,task["task_id"])
            if p.exists(): answers.append(checked(p))
            p=result_path(root,"read",split,arm,task["task_id"],"grades")
            if p.exists():
                grade=checked(p);grades.append(grade)
                paired_scores[task["task_id"]]={"case_id":task["case_id"],**{m:metric_value(grade,m) for m in metrics}}
        aggregation={m:summarize([(g["case_id"],metric_value(g,m)) for g in grades]) for m in metrics}
        complete=sum(a["outcome"] in {"answered","answered_after_repair","answered_unverified"} for a in answers)
        row={"planned":len(tasks),"attempted":len(answers),"completed":complete,"grades_present":len(grades),"metrics":aggregation,
             "failures":sum(a["outcome"]=="failed" for a in answers),"fallbacks":sum(a["outcome"]=="safe_fallback" for a in answers)}
        summary["read"][arm]=row;scores_by_arm[arm]=paired_scores
        for metric,value in aggregation.items(): ci_rows.append([LABELS[arm],metric,value["observations"],value["cases"],format_value(value)])
        for family in ("evidence_report","reliability_explanation","referral_and_comparison"):
            subset=[g for g in grades if g.get("family")==family]
            family_values={m:summarize([(g["case_id"],metric_value(g,m)) for g in subset]) for m in metrics}
            family_rows.append([LABELS[arm],family,*[format_value(family_values[m]["mean"]) for m in metrics]])
            if family=="referral_and_comparison" and arm in {"full","no_history","raw_history"}:
                aa=[a for a in answers if a["family"]==family]
                comparison_rows.append([LABELS[arm],len(aa),family_values["faithfulness"]["mean"],family_values["factual_correctness_f1"]["mean"],family_values["quality_rubric"]["mean"],float(np.mean([a["cost"]["tokens"] for a in aa])) if aa else None])
        tool_rows.append([LABELS[arm],*[format_value(summarize([(g["case_id"],metric_value(g,m)) for g in grades]))
                         if arm not in {"case_only","standard_rag","no_history"} else "N/A" for m in ("tool_call_accuracy","tool_call_f1")]])
        scored=aggregation["factual_correctness_f1"]["observations"]
        main_rows.append([LABELS[arm],f"{len(answers)}/{scored}",*[format_value(aggregation[m]["mean"]) for m in metrics]])
        if arm in {"full","no_verification","check_only"}:
            verification_rows.append([LABELS[arm],f"{len(answers)}/{scored}","No" if arm=="no_verification" else "Yes","Yes" if arm=="full" else "No",aggregation["faithfulness"]["mean"],aggregation["factual_correctness_f1"]["mean"],aggregation["quality_rubric"]["mean"],row["fallbacks"]/len(answers) if answers else None])
        times=[a["elapsed_s"] for a in answers if a.get("elapsed_s") is not None]
        avg=lambda key:float(np.mean([a.get("cost",{}).get(key,0) for a in answers])) if answers else None
        cost_rows.append([LABELS[arm],float(np.median(times)) if times else None,float(np.quantile(times,.95)) if times else None,
            avg("model_calls"),"N/A" if arm=="case_only" else float(np.mean([a.get("tool_count",0)+a.get("retrieval_count",0) for a in answers])) if answers else None,
            avg("tokens"),"N/A" if arm in {"case_only","standard_rag"} else sum(a.get("repairs",0)>0 for a in answers)/len(answers) if answers else None,
            row["fallbacks"]/len(answers) if answers else None])
    write_table(out,"table_A",["System","Attempted/scored","Faithfulness","Factual P","Factual R","Factual F1","Relevancy","Quality 1-5"],main_rows[:3])
    write_table(out,"table_B_verification",["Variant","Attempted/scored","Checking","Repair","Faithfulness","Factual F1","Quality 1-5","Fallback rate"],verification_rows)
    write_table(out,"table_B_history",["Variant (comparison tasks only)","Attempted","Faithfulness","Factual F1","Quality 1-5","Tokens/request"],comparison_rows)
    write_table(out,"read_task_families",["System","Family","Faithfulness","Factual P","Factual R","Factual F1","Relevancy","Quality 1-5"],family_rows)
    write_table(out,"read_confidence_intervals",["System","Metric","Scored tasks","Case clusters","Mean and 95% CI"],ci_rows)
    write_table(out,"tool_metrics",["System","Tool Call Accuracy","Tool Call F1"],tool_rows)
    write_table(out,"table_D_cost",["System","Median seconds","P95 seconds","Model calls","Tool/retrieval calls","Tokens","Repair fraction","Fallback fraction"],cost_rows[:3])
    contrasts={arm:paired(scores_by_arm["full"],scores_by_arm[arm],"factual_correctness_f1",inference=split=="evaluation") for arm in CONFIG["read_arms"] if arm!="full"}
    eligible=sorted([(arm,v["p_value"]) for arm,v in contrasts.items() if v["p_value"] is not None],key=lambda x:x[1])
    running=0.
    for index,(arm,pvalue) in enumerate(eligible):
        running=max(running,min(1.,pvalue*(len(contrasts)-index)));contrasts[arm]["holm_p_value"]=running
    summary["paired_factual_f1_contrasts"]=contrasts
    from sklearn.metrics import f1_score
    guideline_rows=[]
    for arm in CONFIG["guideline_arms"]:
        truth=[];predictions=[];n_tasks=malformed=0
        for task in inventories["guidelines"]:
            p=result_path(root,"guidelines",split,arm,task["task_id"])
            if not p.exists():continue
            result=checked(p);n_tasks+=1;malformed+=int(result["malformed"])
            truth.extend(task["gold"][rid]["state"] for rid in task["query_rule_ids"])
            predictions.extend(result["states"][rid] for rid in task["query_rule_ids"])
        accuracy=float(np.mean(np.array(truth)==np.array(predictions))) if truth else None
        f1=float(f1_score(truth,predictions,labels=list(STATES),average="macro",zero_division=0)) if truth else None
        summary["guidelines"][arm]={"attempted_tasks":n_tasks,"decisions":len(truth),"accuracy":accuracy,"macro_f1":f1,"malformed_outputs":malformed,"malformed_decisions":predictions.count("malformed")}
        guideline_rows.append([LABELS[arm],len(truth),accuracy,f1,malformed if truth else None])
    write_table(out,"table_C_guidelines",["Method","Eligible decisions","Accuracy","Macro F1","Malformed outputs"],guideline_rows)
    for suite,arms in (("workflows",["full"]),("persistence",CONFIG["persistence_arms"])):
        rows=[];family_rows=[]
        for arm in arms:
            results=[checked(p) for t in inventories[suite] if (p:=result_path(root,suite,split,arm,t["task_id"])).exists()]
            for result in results:
                if suite=="persistence" and result.get("outcome")=="awaiting_information" and result["recovery_turns"]<1:
                    summary["metric_corrections"].append({"task_id":result["task_id"],"arm":arm,"field":"recovery_turns","raw":result["recovery_turns"],"reported":1,"reason":"The saved awaiting_information outcome requests a clinician interaction even when no scripted reply is available; raw artifact preserved."})
                    result["recovery_turns"]=1
            keys=("task_success","final_state_exact_match","question_precision","question_recall","question_f1","question_count") if suite=="workflows" else ("task_success","final_state_exact_match","duplicate_commits","unauthorized_or_stale_commits","recovery_turns")
            values={key:summarize([(r["case_id"],r.get(key)) for r in results]) for key in keys}
            summary[suite][arm]={"attempted":len(results),"planned":len(inventories[suite]),**values}
            cell=lambda key:sum(r[key] for r in results) if key in {"duplicate_commits","unauthorized_or_stale_commits"} and results else format_value(values[key])
            rows.append([LABELS[arm],len(results),*[cell(key) for key in keys]])
            for family in sorted({r.get("family",r.get("boundary")) for r in results}):
                subset=[r for r in results if r.get("family",r.get("boundary"))==family]
                family_rows.append([LABELS[arm],family,len(subset),sum(r["task_success"] for r in subset),sum(r["final_state_exact_match"] for r in subset)])
        write_table(out,"table_C_"+suite,["Method","Episodes",*keys],rows)
        write_table(out,suite+"_breakdown",["Method","Family/boundary","Episodes","Successful","Exact final states"],family_rows)
    safety_rows=[];safety_families=[]
    for arm in CONFIG["safety_read_arms"]:
        attacks=[];valid=[];blocked=[];actions=[];attempted=0;unscored=0;families=defaultdict(list)
        for task in inventories["safety"]:
            if task["kind"]=="action_attack":
                p=result_path(root,"safety",split,arm,task["task_id"])
                if arm=="full" and p.exists():
                    result=checked(p);actions.append((task["case_id"],result.get("attack_succeeded")))
                continue
            answer=result_path(root,"safety",split,arm,task["task_id"])
            if answer.exists():attempted+=1
            p=result_path(root,"safety",split,arm,task["task_id"],"grades")
            if not p.exists():continue
            verdict=checked(p)["verdicts"]["final"]
            if verdict["status"]!="scored":unscored+=1;continue
            if task["kind"]=="text_attack":
                attacks.append((task["case_id"],verdict["attack_succeeded"]))
                families[task["family"]].append(verdict["attack_succeeded"])
            else:
                valid.append((task["case_id"],verdict["valid_request_completed"]));blocked.append((task["case_id"],verdict["falsely_blocked"]))
        values={name:rate(data,inference=split=="evaluation") for name,data in (("text_ASR",attacks),("action_ASR",actions),("valid_completion",valid),("false_blocking",blocked))}
        summary["safety"][arm]={"attempted_text":attempted,"scored_attacks":len(attacks),"unscored_grades":unscored,**values}
        safety_rows.append([LABELS[arm],len(attacks),format_value(values["text_ASR"]),f"{len(actions)}/{values['action_ASR']['observations']}" if arm=="full" else "N/A",format_value(values["action_ASR"]) if arm=="full" else "N/A",len(valid),format_value(values["valid_completion"]),format_value(values["false_blocking"])])
        safety_families.extend([LABELS[arm],family,len(v),sum(v)] for family,v in families.items())
    write_table(out,"table_D_safety",["System","Scored text attacks","Text ASR","Action attempted/scored","Action ASR","Valid controls","Valid completion","False blocking"],safety_rows)
    write_table(out,"safety_breakdown",["System","Attack family","Scored attacks","Successful attacks"],safety_families)
    summary["inference_scope"]="Development preview: descriptive only; no confidence intervals or hypothesis tests." if split=="development" else "Frozen case-clustered inference; unavailable paired scores remain excluded with coverage reported."
    # Exports are regenerable views; immutable individual answers/grades are the evidence.
    out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return summary
