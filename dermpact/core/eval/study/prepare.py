"""Prepare/freeze inputs and empty report templates without evaluating reserved cases."""
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

from .common import ROOT, BUNDLE, CONFIG, load, checked, sha, write, stamp
from . import guidelines, workflows, persistence, safety, reporting
from ..read_reference_bundle import SnapshotEvidenceSource
from ..read_contract_v7 import QUESTIONS, reference
from dermpact.core.agent.redesign.baselines import BaselineSettings, retrieval_documents
from dermpact.pipeline.guideline_retriever import FrozenGuidelineRetriever


def source_files():
    files=list((ROOT/"dermpact/core/agent/redesign").glob("*.py"))
    files+=list((ROOT/"dermpact/core/eval/study").glob("*.py"))
    files += [ROOT/"dermpact/core/eval"/(name+".py") for name in
        ("read_reference_bundle","read_contract_v6","read_contract_v7","structured_read_generator","standard_read_metrics",
         "score_saved_read_rubric","codex_judge","applicability_benchmark_v2","read_evaluation_context","kilo_judge")]
    files += [ROOT/"dermpact/core"/(name+".py") for name in
        ("store","record_v2","ledger_v2","gate_v2","clarification","episode","versions","prototype/host","prototype/luna_clarification","prototype/luna_cli")]
    files += [ROOT/"dermpact/pipeline"/(name+".py") for name in ("llm","hybrid_retriever","guideline_retriever")]
    files += [ROOT/"dermpact/guidelines"/(name+".py") for name in ("reviewed_v2","rules","facts","engine","derivation")]
    files += [ROOT/"dermpact/eval/dermpact_phase4_reference_v2.py",ROOT/"dermpact/criteria/schema.py"]
    for directory in ("dermpact/core","dermpact/core/prototype","dermpact/guidelines","dermpact/reliability","dermpact/criteria"):
        files += list((ROOT/directory).glob("*.py"))
    files += [ROOT/"dermpact/data/corpus"/name for name in ("eado_melanoma_part1_rules_v2.jsonl","derm_guideline_recommendations_v1.jsonl")]
    return sorted(set(files))


def prepare(root):
    root=Path(root)
    if (root/"lock.json").exists():raise FileExistsError("Study already frozen; use a new output directory for a changed configuration")
    roles=checked(BUNDLE/"roles.json")["roles"]
    # Stable outcome-independent order; role membership is inherited without alteration.
    from dermpact.core.agent.redesign.evidence import digest
    role_ids={split:sorted(roles[split],key=lambda cid:digest({"seed":20260923,"case_id":cid})) for split in ("development","evaluation")}
    assert len(role_ids["development"])==25 and len(role_ids["evaluation"])==100
    assert not set(role_ids["development"])&set(role_ids["evaluation"])
    write(root/"configuration.json",CONFIG)
    write(root/"case_roles.json",role_ids)
    library=ROOT/"dermpact/data/corpus/derm_guideline_recommendations_v1.jsonl"
    retriever=FrozenGuidelineRetriever.load(library,expected_sha256=Path(str(library)+".sha256").read_text(encoding="ascii").strip())
    factory=workflows.RecordFactory()
    counts={}
    try:
        for split,ids in role_ids.items():
            reads=[]
            for cid in ids:
                source=SnapshotEvidenceSource(cid,role=split,allow_evaluation=split=="evaluation")
                for family,question in QUESTIONS.items():
                    text=reference(source.snapshot,family)
                    reads.append({"task_id":cid+"::"+family,"case_id":cid,"split":split,"family":family,"question":question,
                        "reference":text,"reference_goal":"The delivered answer communicates: "+text,
                        "tool_alternatives":[[{"name":"retrieve_case_comparisons","args":{}}],[{"name":"retrieve_case_comparisons","args":{"detail":"full"}}]]
                                            if family=="referral_and_comparison" else [[]],
                        "snapshot_sha256":source.snapshot["snapshot_sha256"]})
                write(root/"corpora"/split/(cid+".json"),retrieval_documents(source,retriever.records,BaselineSettings()))
            inventories={"read":reads,"guidelines":guidelines.prepare_tasks(ids,split),
                         "workflows":workflows.prepare_tasks(root,ids,split,factory),
                         "persistence":persistence.prepare_tasks(root,ids,split,factory),
                         "safety":safety.prepare_tasks(root,ids,split,factory)}
            counts[split]={name:len(tasks) for name,tasks in inventories.items()}
            for name,tasks in inventories.items():write(root/"inputs"/(name+"_"+split+".json"),tasks)
    finally:factory.close()
    files=source_files()
    for p in files:
        destination=root/"source"/p.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
    input_files=[p for folder in ("inputs","fixtures","corpora") for p in (root/folder).rglob("*.json")]
    input_files += [root/"configuration.json",root/"case_roles.json"]
    lock={"version":CONFIG["version"],"created_at":stamp(),"source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in files},
        "input_sha256":{str(p.relative_to(root)):sha(p) for p in input_files},
        "upstream_bindings":{str(BUNDLE.relative_to(ROOT)/name):sha(BUNDLE/name) for name in ("roles.json","manifest.json","reference_manifest.json","EVALUATION_LOCK.json")},
        "environment":{"python":platform.python_version(),**{p:importlib.metadata.version(p) for p in ("ragas","pydantic","openai","langgraph","langgraph-checkpoint-sqlite","numpy","scikit-learn")}},
        "counts":counts,"reserved_answers_generated":0,
        "protocol":"Prepared source/reference artifacts only. No model or evaluated deterministic arm is executed over final tasks by prepare()."}
    write(root/"lock.json",lock)
    reporting.export(root,"development");reporting.export(root,"evaluation")
    print(json.dumps({"prepared":str(root),"counts":counts,"reserved_answers_generated":0}),flush=True)
    return lock
