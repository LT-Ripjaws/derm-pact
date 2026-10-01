"""Build reference obligations from frozen facts, not from candidate answers."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from dermpact.core.agent.redesign.evidence import EvidenceError, TOOLS, digest
from dermpact.core.eval.prepare_read_evaluation import OUT, PROFILE, QUESTIONS, sha, write


def required_units(snapshot, family):
    p,r,m=snapshot["perception"],snapshot["reliability"],snapshot["comparison"]
    result=[]
    def add(key, expectation): result.append({"unit_id":key,"expectation":expectation})
    if family=="evidence_report":
        add("diagnosis",f"Identify the model's diagnosis as {p['diagnosis']}; do not present it as confirmed clinical truth.")
        add("melanoma_estimate",f"Report the separate melanoma probability {p['melanoma_probability']:.6f}, allowing reasonable percentage/decimal rounding.")
        highest=sorted(p["criterion_positive_probabilities"].items(),key=lambda x:(-x[1],x[0]))[:2]
        for name,value in highest:
            add("criterion_"+name,f"Identify {name.replace('_',' ')} as one of the two strongest model-predicted findings; probability {value:.6f} if quantified.")
        add("prediction_scope","Distinguish model predictions from confirmed clinician observations.")
    else:
        referred=r["disposition"]=="review"
        add("referral_status","Explain that the case WAS selectively referred for clinician review." if referred else
            "Correct the question's premise if necessary: the case was NOT selectively referred; display/answered is not referral.")
        if referred:
            for w in r["warnings"]: add("trigger_"+w["code"],w["reason"]+". Do not substitute checklist/guideline state for this trigger.")
        if family=="reliability_explanation":
            if r["ood_state"]=="flagged":add("ood_scope","Disclose the unfamiliarity warning as advisory only; it did not itself trigger referral.")
        elif family=="referral_and_comparison":
            matched=bool(m["matches"])
            add("comparison_available",f"State that {len(m['matches'])} eligible historical comparisons exist (an exact count is optional)." if matched else
                "Disclose that no eligible historical comparison is available; do not invent an example.")
            add("comparison_effect","Explain that historical comparisons cannot change the stored diagnosis, probabilities, or referral decision.")
            if matched:add("comparison_provenance","Treat historical labels as dataset references, not clinician-approved conclusions in this system or proof of the current diagnosis.")
        else: raise ValueError("unknown reference family")
    return result


def reference(snapshot, task):
    return {"schema":"dermpact-independent-read-reference-v1","task_id":task["task_id"],"case_id":snapshot["case_id"],
            "question":task["question"],"family":task["family"],"snapshot_sha256":snapshot["snapshot_sha256"],
            "evidence":{k:copy.deepcopy(snapshot[k]) for k in ("perception","reliability","comparison")},
            "required_units":required_units(snapshot,task["family"]),
            "scope_rules":["Stored model outputs are not clinically confirmed diagnoses.",
                           "Read-only evidence cannot mutate diagnosis, probabilities, or referral.",
                           "Only recorded reliability triggers explain selective referral; OOD is advisory.",
                           "A guideline recommendation is not evidence of selective referral.",
                           "No claim that dataset labels lack original clinical confirmation follows from absence of approval within this system."],
            "authorship":"Engineering reference builder from frozen source facts; no candidate answer or runtime verdict used; not clinician adjudication."}


def build():
    hashes={}
    for role in ("development","evaluation"):
        tasks=json.loads((OUT/role/"tasks.json").read_text(encoding="utf-8"))
        refs=[]
        for task in tasks:
            path=OUT/task["case_path"]
            if sha(path)!=task["case_sha256"]:raise ValueError("case artifact drift")
            snapshot=json.loads(path.read_text(encoding="utf-8"))
            refs.append(reference(snapshot,task))
        path=OUT/role/"references.json";write(path,refs);hashes[role]=sha(path)
    write(OUT/"reference_manifest.json",{"hashes":hashes,"builder_sha256":sha(Path(__file__)),
          "candidate_answers_loaded":0,"runtime_verdicts_loaded":0,"clinical_adjudication":False,
          "evaluation_task_count":300,"development_task_count":75})
    print("Frozen 300 evaluation and 75 development references without generating evaluation answers.")


class SnapshotEvidenceSource:
    """Actual agent read interface over frozen OOF evidence, explicit evaluation gate."""
    def __init__(self, case_id, *, role="development", allow_evaluation=False, bundle=OUT):
        if role not in {"development","evaluation"} or (role=="evaluation" and not allow_evaluation):
            raise EvidenceError("evaluation_access_requires_explicit_mode")
        bundle=Path(bundle).resolve()
        manifest=json.loads((bundle/"manifest.json").read_text(encoding="utf-8"))
        self.path=bundle/role/"cases"/(case_id+".json")
        if not self.path.resolve().is_relative_to(bundle/role/"cases"):
            raise EvidenceError("invalid_case_path")
        expected=manifest["artifact_hashes"].get(str(self.path.relative_to(bundle)))
        # Manifest uses the platform's serialized relative path convention.
        if expected is None or sha(self.path)!=expected:raise EvidenceError("snapshot_file_hash_mismatch")
        self.snapshot=json.loads(self.path.read_text(encoding="utf-8"))
        if self.snapshot["snapshot_sha256"]!=digest({k:v for k,v in self.snapshot.items() if k!="snapshot_sha256"}):
            raise EvidenceError("snapshot_content_hash_mismatch")
        if self.snapshot["case_id"]!=case_id or self.snapshot["role"]!=role:raise EvidenceError("snapshot_role_mismatch")
        self.expected_file_sha=expected

    def identity(self):
        if sha(self.path)!=self.expected_file_sha:raise EvidenceError("snapshot_changed")
        return {"case_id":self.snapshot["case_id"],"record_version":1,"record_sha256":self.snapshot["snapshot_sha256"],
                "source_label":"agent_"+self.snapshot["role"]+"_single_member_oof"}

    def tools(self):return dict(TOOLS)

    def read(self,tool,args):
        if tool not in TOOLS:raise EvidenceError("tool_not_permitted")
        identity=self.identity()
        for key in ("case_id","record_version"):
            if key in args and args[key]!=identity[key]:raise EvidenceError("wrong_case_or_version_requested")
        permitted={"case_id","record_version","detail","section"} if tool=="read_record_section" else {"case_id","record_version","detail"}
        if set(args)-permitted:raise EvidenceError("unexpected_tool_argument")
        if args.get("detail","summary") not in {"summary","full"}:raise EvidenceError("invalid_evidence_detail")
        if tool=="read_record_section" and args.get("section","overview")!="overview":raise EvidenceError("unsupported_record_section")
        data=(self.snapshot["reliability"] if tool=="read_reliability" else self.snapshot["comparison"] if tool=="retrieve_case_comparisons"
              else {"perception":self.snapshot["perception"],"reliability":self.snapshot["reliability"],
                    "prediction_scope":"Single held-out model with disjoint calibration; not the historical five-member deployment ensemble."})
        return {**identity,"tool":tool,"ok":True,"detail":args.get("detail","summary"),"data":copy.deepcopy(data),"error":None}


if __name__=="__main__":build()
