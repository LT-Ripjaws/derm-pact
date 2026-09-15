"""One-shot, resumable final evaluation for the selected DermPact configuration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dermpact.core import versions as V
from dermpact.core.eval.applicability_benchmark_v2 import _atomic_json, _atomic_write, _sha256_file
from dermpact.core.eval.final_metrics import binary_metrics, bootstrap_metrics, criterion_metrics, multiclass_metrics
from dermpact.core.inference_client import WorkerClient
from dermpact.eval.dermpact_phase2_baselines import CRITERIA, POSITIVE_VALUES
from dermpact.reliability.bundle import ReliabilityBundle
from dermpact.reliability.ham_ensemble import CLASSES as HAM_CLASSES, assess_ham_lesion, load_parameters
from scipy.special import expit, softmax

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research/manifests/dermpact_final_v1"
PROTOCOL = BASE / "protocol.json"
ACCESS_LOG = ROOT / "research/DERMPACT_SELECTED_FINAL_ACCESS_LOG.jsonl"
OUTPUT = ROOT / "research/experiments/dermpact_final_v1"
MANIFESTS = {
    "roles": ROOT / "research/manifests/dermpact_phase1_v2/dermpact_phase1_role_assignment_v2.json",
    "derm7pt": ROOT / "research/manifests/dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json",
    "ham": ROOT / "research/manifests/ham10000_split_v2.json",
    "bcn": ROOT / "research/manifests/bcn20000_external_v1.json",
    "ph2": ROOT / "research/manifests/dermpact_phase1_v2/dermpact_phase1_ph2_manifest_v2.json",
}
SELECTION = ROOT / "research/manifests/dermpact_phase2_final/selection_2026-09-06.json"
HAM_PARAMETERS = ROOT / "research/experiments/dermpact_phase3/ensemble_ham_calibration_v1/calibrators.json"
BCN_FEATURES = ROOT / "research/experiments/derm_perception/v2_tournament/external_bcn_v1/features/panderm/features.npy"
BCN_FEATURE_MANIFEST = BCN_FEATURES.parent / "feature_manifest.json"
BCN_COMBINED_CHECKPOINTS = tuple(
    ROOT / f"research/experiments/dermpact_phase2_2/dataset_composition_v1/derm_ham_bcn/seed_{seed}/fold_{fold}/epoch_12_state.pt"
    for seed in (20260827, 20260828, 20260829) for fold in range(5)
)
N_BOOT = 2000
SEED = 20260909
UNIVERSAL_CLASSES = [
    "melanoma", "nevus", "basal_cell_carcinoma", "squamous_cell_carcinoma",
    "actinic_keratosis_or_intraepithelial_carcinoma", "seborrheic_keratosis_or_benign_keratosis",
    "dermatofibroma", "vascular_lesion", "other",
]
DERM5_CLASSES = ["MEL", "NEV", "BCC", "SK", "MISC"]
DERM5_COLUMNS = [[0], [1], [2], [5], [3, 4, 6, 7, 8]]
BCN_CLASSES = ["akiec", "bcc", "bkl", "df", "melanoma", "nevus", "scc", "vasc"]
BCN_COLUMNS = [4, 2, 5, 6, 0, 1, 3, 7]


def _verify_sidecar(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    if not path.is_file() or not sidecar.is_file() or _sha256_file(path) != sidecar.read_text(encoding="ascii").strip():
        raise ValueError(f"artifact or sidecar verification failed: {path}")


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _access_entries() -> list[dict[str, Any]]:
    if not ACCESS_LOG.is_file():
        return []
    return [json.loads(line) for line in ACCESS_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify_access_log() -> str:
    previous = "0" * 64
    for entry in _access_entries():
        if entry.get("previous_sha256") != previous:
            raise ValueError("final access-log chain is broken")
        body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        if _canonical_hash(body) != entry.get("entry_sha256"):
            raise ValueError("final access-log entry digest is invalid")
        previous = entry["entry_sha256"]
    return previous


def _append_access(event: str, protocol_sha256: str, detail: str) -> dict[str, Any]:
    entries = _access_entries()
    previous = _verify_access_log()
    body = {
        "sequence": len(entries), "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event, "protocol_sha256": protocol_sha256, "detail": detail,
        "previous_sha256": previous,
    }
    body["entry_sha256"] = _canonical_hash(body)
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ACCESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(body, sort_keys=True) + "\n");handle.flush();os.fsync(handle.fileno())
    return body


def freeze_protocol(path: Path = PROTOCOL) -> dict[str, Any]:
    for manifest in MANIFESTS.values():
        _verify_sidecar(manifest)
    _verify_sidecar(SELECTION);_verify_sidecar(ROOT / V.RELIABILITY_BUNDLE);_verify_sidecar(HAM_PARAMETERS)
    feature_manifest=json.loads(BCN_FEATURE_MANIFEST.read_text(encoding="utf-8"))
    if _sha256_file(BCN_FEATURES)!=feature_manifest["feature_file_sha256"]:raise AssertionError("BCN feature matrix drift")
    if len(BCN_COMBINED_CHECKPOINTS)!=15 or any(not path.is_file() for path in BCN_COMBINED_CHECKPOINTS):raise AssertionError("15 frozen BCN-combined checkpoints required")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    checkpoints = selection["artifacts"]["stage2"]["fold_checkpoints"]
    if len(checkpoints) != 5:
        raise AssertionError("five selected checkpoints required")
    for binding in checkpoints:
        if _sha256_file(ROOT / binding["path"]) != binding["sha256"]:
            raise AssertionError("selected checkpoint drift")
    source_paths = [Path(__file__), ROOT / "dermpact/core/eval/final_metrics.py",
                    ROOT / "dermpact/reliability/selected_perception.py",
                    ROOT / "dermpact/core/final_inference.py",
                    ROOT / "dermpact/core/inference_worker.py",
                    ROOT / "dermpact/core/inference_client.py",
                    ROOT / "dermpact/reliability/bundle.py", ROOT / "dermpact/reliability/ham_ensemble.py",
                    ROOT / "dermpact/eval/dermpact_phase2_hybrid_screen.py",
                    ROOT / "dermpact/eval/dermpact_phase2_baselines.py"]
    payload = {
        "schema_version": "dermpact-selected-final-evaluation-v1.1",
        "status": "FROZEN_BEFORE_FINAL_ACCESS",
        "mechanical_amendment": "BCN final rows join the frozen Phase-1 bcn_final_test role set; v1 stopped before predictions after using the source manifest's non-role split field",
        "model": "phase2_5_stage2_cefix_v1_five_member",
        "routes": {"Derm7pt": "full_multimodal_criterion_with_reflection", "PH2": "full_criterion_with_reflection",
                   "HAM10000": "selected global_diagnosis_only", "BCN20000_transfer": "selected global_diagnosis_only",
                   "BCN20000_combined": "frozen Phase-2.2 Derm+HAM+BCN pooled-feature 15-model ensemble"},
        "grouping": {"Derm7pt": "lesion", "HAM10000": "mean image probabilities within lesion",
                     "BCN20000": "mean image probabilities within lesion", "PH2": "released lesion image"},
        "metrics": ["accuracy", "balanced_accuracy", "macro_f1", "mcc", "macro_ovr_auroc",
                    "melanoma_auroc", "sensitivity_at_90_and_95_specificity", "nll", "brier", "ece_10bin",
                    "criterion_raw_state_accuracy_macro_f1", "criterion_presence_f1_auroc",
                    "eight_output_mean_auroc", "selective_coverage_risk_error_capture", "iqa_ood_view_flags"],
        "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "lesion/source case"},
        "manifest_bindings": {name: {"path": p.relative_to(ROOT).as_posix(), "sha256": _sha256_file(p)}
                              for name, p in MANIFESTS.items()},
        "selection": {"path": SELECTION.relative_to(ROOT).as_posix(), "sha256": _sha256_file(SELECTION)},
        "reliability_bundle": {"path": V.RELIABILITY_BUNDLE, "sha256": _sha256_file(ROOT / V.RELIABILITY_BUNDLE)},
        "ham_parameters": {"path": HAM_PARAMETERS.relative_to(ROOT).as_posix(), "sha256": _sha256_file(HAM_PARAMETERS)},
        "bcn_combined": {"feature_matrix":{"path":BCN_FEATURES.relative_to(ROOT).as_posix(),"sha256":_sha256_file(BCN_FEATURES)},
                         "feature_manifest":{"path":BCN_FEATURE_MANIFEST.relative_to(ROOT).as_posix(),"sha256":_sha256_file(BCN_FEATURE_MANIFEST)},
                         "checkpoints":[{"path":p.relative_to(ROOT).as_posix(),"sha256":_sha256_file(p)} for p in BCN_COMBINED_CHECKPOINTS]},
        "checkpoints": checkpoints,
        "source_bindings": {p.relative_to(ROOT).as_posix(): _sha256_file(p) for p in source_paths},
        "prohibitions": ["no post-final tuning", "no threshold selection on final labels", "no fold selection on final results"],
        "historical_exposure": "Derm7pt/HAM/BCN were evaluated by the previous project; this is confirmation for a new frozen configuration. PH2 is prediction-unopened before this run.",
    }
    _atomic_json(Path(path), payload)
    return payload


def _load_protocol(path: Path = PROTOCOL) -> tuple[dict[str, Any], str]:
    _verify_sidecar(path);protocol=json.loads(path.read_text(encoding="utf-8"));sha=_sha256_file(path)
    for binding in protocol["manifest_bindings"].values():
        if _sha256_file(ROOT / binding["path"]) != binding["sha256"]:raise ValueError("final manifest drift")
    for rel, expected in protocol["source_bindings"].items():
        if _sha256_file(ROOT / rel) != expected:raise ValueError(f"final evaluator source drift: {rel}")
    for binding in protocol["checkpoints"]:
        if _sha256_file(ROOT / binding["path"]) != binding["sha256"]:raise ValueError("final checkpoint drift")
    for binding in protocol["bcn_combined"]["checkpoints"]:
        if _sha256_file(ROOT/binding["path"])!=binding["sha256"]:raise ValueError("BCN combined checkpoint drift")
    for key in ("feature_matrix","feature_manifest"):
        binding=protocol["bcn_combined"][key]
        if _sha256_file(ROOT/binding["path"])!=binding["sha256"]:raise ValueError("BCN combined feature drift")
    return protocol, sha


def _group(records: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:grouped[str(record[key])].append(record)
    return dict(grouped)


def resolve_final_cases() -> list[dict[str, Any]]:
    """This function opens final labels and must only be called after access logging."""
    cases: list[dict[str, Any]] = []
    derm = json.loads(MANIFESTS["derm7pt"].read_text(encoding="utf-8"))["records"]
    for record in derm:
        if record["phase1_role"] != "final_test":continue
        derm_view=record["modalities"]["dermoscopic"];clinical=record["modalities"]["clinical"]
        clinical_ok=bool(clinical.get("available") and clinical.get("distinct_from_dermoscopic"))
        cases.append({"dataset":"Derm7pt","group_id":record["group_id"],"route":"full",
            "images":[{"case_id":record["group_id"],"path":str(ROOT/"release_v0/images"/derm_view["path"]),"sha256":derm_view["sha256"],
                       "clinical_path":str(ROOT/"release_v0/images"/clinical["path"]) if clinical_ok else None,
                       "clinical_sha256":clinical["sha256"] if clinical_ok else None}],
            "label":record["diagnosis"]["harmonized5"],"raw_label":record["diagnosis"]["raw"],
            "criteria_raw":record["criteria_raw"],"clinical_available":clinical_ok})
    ham = json.loads(MANIFESTS["ham"].read_text(encoding="utf-8"))["records"]
    for group_id, rows in _group((r for r in ham if r["split"]=="test"), "cluster_id").items():
        labels={r["label"] for r in rows}
        if len(labels)!=1:raise AssertionError(f"HAM lesion label conflict: {group_id}")
        cases.append({"dataset":"HAM10000","group_id":f"ham10000:{group_id}","route":"diagnosis_only",
                      "images":[{"case_id":r["case_id"],"path":r["image_path"],"sha256":r["image_sha256"]} for r in rows],
                      "label":next(iter(labels)),"clinical_available":False})
    role_manifest=json.loads(MANIFESTS["roles"].read_text(encoding="utf-8"))
    bcn_final_groups={str(row["group_id"]).split(":",1)[1] for row in role_manifest["group_rows"]
                      if row["dataset"]=="BCN20000" and row["role"]=="bcn_final_test"}
    if len(bcn_final_groups)!=712:raise AssertionError("frozen BCN final role count changed")
    bcn = json.loads(MANIFESTS["bcn"].read_text(encoding="utf-8"))["records"]
    for group_id, rows in _group((r for r in bcn if str(r["cluster_id"]) in bcn_final_groups), "cluster_id").items():
        labels={r["label"] for r in rows}
        if len(labels)!=1:raise AssertionError(f"BCN lesion label conflict: {group_id}")
        cases.append({"dataset":"BCN20000","group_id":f"bcn20000:{group_id}","route":"diagnosis_only",
                      "images":[{"case_id":r["case_id"],"path":r["image_path"],"sha256":r["image_sha256"]} for r in rows],
                      "label":next(iter(labels)),"clinical_available":False})
    ph2 = json.loads(MANIFESTS["ph2"].read_text(encoding="utf-8"))["records"]
    for record in ph2:
        cases.append({"dataset":"PH2","group_id":record["group_id"],"route":"full",
                      "images":[{"case_id":record["case_id"],"path":str(ROOT/record["image"]["path"]),"sha256":record["image"]["sha256"],
                                 "clinical_path":None,"clinical_sha256":None}],
                      "label":record["diagnosis"]["harmonized5"],"criteria_targets":record["criterion_targets"],
                      "clinical_available":False})
    counts={dataset:sum(case["dataset"]==dataset for case in cases) for dataset in ("Derm7pt","HAM10000","BCN20000","PH2")}
    if counts!={"Derm7pt":395,"HAM10000":1495,"BCN20000":712,"PH2":200}:raise AssertionError(counts)
    return cases


def _cache_path(dataset: str, group_id: str, image_id: str) -> Path:
    digest=hashlib.sha256(f"{dataset}:{group_id}:{image_id}".encode()).hexdigest()
    return OUTPUT/"image_predictions"/dataset/f"{digest}.json"


def _prediction_key(protocol_sha: str, case: dict[str, Any], image: dict[str, Any]) -> str:
    return _canonical_hash({"protocol_sha256":protocol_sha,"dataset":case["dataset"],"group_id":case["group_id"],
                            "route":case["route"],"image_id":image["case_id"],"image_sha256":image["sha256"],
                            "clinical_sha256":image.get("clinical_sha256")})


def _run_predictions(cases: list[dict[str, Any]], protocol_sha: str) -> None:
    bundle=ReliabilityBundle(ROOT/V.RELIABILITY_BUNDLE);client=WorkerClient(timeout_seconds=1200)
    total=sum(len(case["images"]) for case in cases);done=0
    try:
        client.ping()
        for case in cases:
            for image in case["images"]:
                done+=1;path=_cache_path(case["dataset"],case["group_id"],image["case_id"]);key=_prediction_key(protocol_sha,case,image)
                if path.is_file():
                    cached=json.loads(path.read_text(encoding="utf-8"))
                    if cached.get("cache_key")==key:
                        print(f"[{done}/{total}] CACHE {case['dataset']} {case['group_id']} {image['case_id']}",flush=True);continue
                started=time.perf_counter()
                if case["route"]=="full":
                    prediction=client.predict_final_full(dermoscopic_path=image["path"],clinical_path=image.get("clinical_path"),
                        expected_dermoscopic_sha256=image["sha256"],expected_clinical_sha256=image.get("clinical_sha256"),check_view=True)
                    payload=Path(image["path"]).read_bytes();reliability=bundle.assess_selected_prediction(payload,prediction).to_dict()
                else:
                    prediction=client.predict_diagnosis_only(dermoscopic_path=image["path"],expected_dermoscopic_sha256=image["sha256"])
                    reliability=None
                _atomic_json(path,{"cache_key":key,"dataset":case["dataset"],"group_id":case["group_id"],
                    "image_id":image["case_id"],"prediction":prediction,"reliability":reliability,
                    "wall_seconds":time.perf_counter()-started})
                print(f"[{done}/{total}] DONE {case['dataset']} {case['group_id']} {image['case_id']}",flush=True)
    finally:client.shutdown()


def _mean_prob(predictions: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([p[key] for p in predictions],dtype=np.float64).mean(axis=0)


def _map_prob(probability: np.ndarray, columns: list[list[int]]) -> np.ndarray:
    mapped=np.asarray([probability[indexes].sum() for indexes in columns],dtype=np.float64)
    return mapped/np.clip(mapped.sum(),1e-12,None)


def _dataset_rows(cases: list[dict[str, Any]]) -> dict[str,list[dict[str,Any]]]:
    output:dict[str,list[dict[str,Any]]]=defaultdict(list);ham_params=load_parameters(HAM_PARAMETERS)
    for case in cases:
        cached=[json.loads(_cache_path(case["dataset"],case["group_id"],image["case_id"]).read_text(encoding="utf-8")) for image in case["images"]]
        predictions=[row["prediction"] for row in cached];row={"group_id":case["group_id"],"label":case["label"],
            "clinical_available":case["clinical_available"],"n_images":len(predictions),"route":case["route"]}
        if case["dataset"]=="HAM10000":
            diag=np.asarray([[p["raw_fold_outputs"][fold]["diagnosis"] for p in predictions] for fold in range(5)])
            mel=np.asarray([[p["raw_fold_outputs"][fold]["melanoma"] for p in predictions] for fold in range(5)])
            assessed=assess_ham_lesion(diag,mel,ham_params);row.update(probabilities=assessed["diagnosis_probabilities"],
                melanoma_probability=assessed["melanoma_probability"],selective_risk=assessed["selective_risk"],
                referred=assessed["selective_abstention"]=="deferred",reliability={"selective_abstention":assessed["selective_abstention"]})
        else:
            diagnosis=_mean_prob(predictions,"diagnosis_probabilities")
            if case["dataset"]=="Derm7pt":row["probabilities"]=_map_prob(diagnosis,DERM5_COLUMNS).tolist()
            elif case["dataset"]=="BCN20000":
                p=diagnosis[BCN_COLUMNS];row["probabilities"]=(p/np.clip(p.sum(),1e-12,None)).tolist()
            else:
                p=diagnosis[[0,1]];row["probabilities"]=(p/np.clip(p.sum(),1e-12,None)).tolist()
            row["universal_probabilities"]=diagnosis.tolist();row["melanoma_probability"]=float(_mean_prob(predictions,"binary_probabilities")[0])
            if case["route"]=="full":
                binary=_mean_prob(predictions,"binary_probabilities");row["binary_probabilities"]=binary.tolist()
                row["checklist_melanoma_probability"]=float(np.mean([p["checklist_melanoma_probability"] for p in predictions]))
                row["criterion_state_labels"]=predictions[0]["criterion_state_labels"]
                row["criterion_state_probabilities"]={name:np.asarray([p["criterion_state_probabilities"][name] for p in predictions]).mean(0).tolist() for name in CRITERIA}
                reliabilities=[item["reliability"] for item in cached];row["reliability_images"]=reliabilities
                row["referred"]=any(r["disposition"]=="review" for r in reliabilities)
                risks=[max((r.get("scores") or {}).get("member_uncertainty") or [0.0]) for r in reliabilities]
                row["selective_risk"]=max(risks)
        row["criteria_raw"]=case.get("criteria_raw");row["criteria_targets"]=case.get("criteria_targets")
        output[case["dataset"]].append(row)
    return dict(output)


def _multiclass_report(rows:list[dict[str,Any]],classes:list[str],melanoma_label:str,seed:int)->dict[str,Any]:
    index={label:i for i,label in enumerate(classes)};y=np.asarray([index[r["label"]] for r in rows]);p=np.asarray([r["probabilities"] for r in rows])
    mel=index[melanoma_label];point=multiclass_metrics(y,p,melanoma_index=mel)
    keys=("accuracy","balanced_accuracy","macro_f1","mcc","macro_ovr_auroc","melanoma_auroc","nll","multiclass_brier","ece_10bin")
    point["confidence_intervals"]=bootstrap_metrics(y,p,lambda yy,pp:multiclass_metrics(yy,pp,melanoma_index=mel),keys=keys,seed=seed,n_boot=N_BOOT)
    point["classes"]=classes;return point


def _selective(rows:list[dict[str,Any]],classes:list[str],melanoma_label:str)->dict[str,Any]:
    index={label:i for i,label in enumerate(classes)};y=np.asarray([index[r["label"]] for r in rows]);pred=np.asarray([np.argmax(r["probabilities"]) for r in rows]);referred=np.asarray([bool(r.get("referred")) for r in rows]);errors=pred!=y;answered=~referred
    risk=np.asarray([float(r.get("selective_risk",0.0)) for r in rows]);curve=[]
    for threshold in np.quantile(risk,np.linspace(0,1,21)):
        keep=risk<=threshold
        curve.append({"threshold":float(threshold),"coverage":float(keep.mean()),"selective_risk":float(errors[keep].mean()) if keep.any() else None,"selective_accuracy":float((~errors[keep]).mean()) if keep.any() else None})
    return {"n":len(rows),"coverage":float(answered.mean()),"selective_accuracy":float((~errors[answered]).mean()) if answered.any() else None,
        "selective_risk":float(errors[answered].mean()) if answered.any() else None,
        "error_capture":float(errors[referred].sum()/errors.sum()) if errors.sum() else None,
        "melanomas_referred":int(((y==index[melanoma_label])&referred).sum()),"non_melanomas_referred":int(((y!=index[melanoma_label])&referred).sum()),
        "n_referred":int(referred.sum()),"risk_coverage_curve":curve}


def _bcn_combined_report(seed:int)->tuple[dict[str,Any],list[dict[str,Any]]]:
    import torch
    from dermpact.eval.dermpact_phase2_hybrid_screen import HybridSharedHead
    records=json.loads(MANIFESTS["bcn"].read_text(encoding="utf-8"))["records"]
    role_manifest=json.loads(MANIFESTS["roles"].read_text(encoding="utf-8"))
    final_groups={str(row["group_id"]).split(":",1)[1] for row in role_manifest["group_rows"]
                  if row["dataset"]=="BCN20000" and row["role"]=="bcn_final_test"}
    indices=np.asarray([i for i,row in enumerate(records) if str(row["cluster_id"]) in final_groups],dtype=np.int64)
    features=np.load(BCN_FEATURES,mmap_mode="r")[indices]
    diagnosis=[];melanoma=[]
    with torch.inference_mode():
        for checkpoint in BCN_COMBINED_CHECKPOINTS:
            model=HybridSharedHead();model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True));model.eval()
            dz=[];mz=[]
            for start in range(0,len(features),512):
                out=model(torch.from_numpy(np.asarray(features[start:start+512],dtype=np.float32).copy()))
                dz.append(out["diagnosis"].numpy());mz.append(out["melanoma"].numpy())
            diagnosis.append(softmax(np.concatenate(dz),axis=1));melanoma.append(expit(np.concatenate(mz)))
    image_probability=np.asarray(diagnosis).mean(0);image_melanoma=np.asarray(melanoma).mean(0)
    grouped:dict[str,list[int]]=defaultdict(list)
    final_records=[records[i] for i in indices]
    for i,row in enumerate(final_records):grouped[str(row["cluster_id"])].append(i)
    rows=[]
    for group_id,positions in sorted(grouped.items()):
        labels={final_records[i]["label"] for i in positions}
        if len(labels)!=1:raise AssertionError("BCN combined lesion label conflict")
        p=image_probability[positions].mean(0)[BCN_COLUMNS];p=p/np.clip(p.sum(),1e-12,None)
        rows.append({"group_id":f"bcn20000:{group_id}","label":next(iter(labels)),"probabilities":p.tolist(),
                     "melanoma_probability":float(image_melanoma[positions].mean()),"n_images":len(positions),
                     "route":"phase2_2_derm_ham_bcn_combined_15_model"})
    report=_multiclass_report(rows,BCN_CLASSES,"melanoma",seed)
    y=np.asarray([row["label"]=="melanoma" for row in rows],dtype=int)
    report["dedicated_melanoma_head"]=binary_metrics(y,[row["melanoma_probability"] for row in rows])
    report["calibration_status"]="uncalibrated equal-probability average across 3 seeds x 5 frozen folds"
    return report,rows


def score(cases:list[dict[str,Any]],protocol_sha:str)->dict[str,Any]:
    rows=_dataset_rows(cases);reports={
        "Derm7pt":_multiclass_report(rows["Derm7pt"],DERM5_CLASSES,"MEL",SEED+1),
        "HAM10000":_multiclass_report(rows["HAM10000"],HAM_CLASSES,"melanoma",SEED+2),
        "BCN20000":_multiclass_report(rows["BCN20000"],BCN_CLASSES,"melanoma",SEED+3),
        "PH2":_multiclass_report(rows["PH2"],["MEL","NEV"],"MEL",SEED+4),
    }
    reports["Derm7pt"]["selective_referral"]=_selective(rows["Derm7pt"],DERM5_CLASSES,"MEL")
    reports["HAM10000"]["selective_referral"]=_selective(rows["HAM10000"],HAM_CLASSES,"melanoma")
    reports["PH2"]["selective_referral"]=_selective(rows["PH2"],["MEL","NEV"],"MEL")
    for dataset,classes,mel_label in (("HAM10000",HAM_CLASSES,"melanoma"),("BCN20000",BCN_CLASSES,"melanoma")):
        truth=np.asarray([row["label"]==mel_label for row in rows[dataset]],dtype=int)
        reports[dataset]["dedicated_melanoma_head"]=binary_metrics(truth,[row["melanoma_probability"] for row in rows[dataset]])
    derm=rows["Derm7pt"];criteria={}
    for j,name in enumerate(CRITERIA):
        labels=derm[0]["criterion_state_labels"][name];truth=[r["criteria_raw"][name] for r in derm]
        state=np.asarray([r["criterion_state_probabilities"][name] for r in derm]);binary=np.asarray([r["binary_probabilities"][j+1] for r in derm])
        result=criterion_metrics(truth,state,labels,set(POSITIVE_VALUES[name]),binary)
        truth_binary=np.asarray([value in POSITIVE_VALUES[name] for value in truth],dtype=int)
        result["presence_confidence_intervals"]=bootstrap_metrics(truth_binary,binary,
            lambda yy,pp:binary_metrics(yy,pp),keys=("accuracy","f1","auroc"),seed=SEED+100+j,n_boot=N_BOOT)
        criteria[name]=result
    mel_truth=np.asarray([r["label"]=="MEL" for r in derm],dtype=int)
    direct=np.asarray([r["probabilities"][0] for r in derm]);dedicated=np.asarray([r["melanoma_probability"] for r in derm]);checklist=np.asarray([r["checklist_melanoma_probability"] for r in derm])
    routes={"direct_diagnosis":binary_metrics(mel_truth,direct),"dedicated_melanoma_head":binary_metrics(mel_truth,dedicated),"checklist_derived":binary_metrics(mel_truth,checklist)}
    criterion_aucs=[criteria[name]["melanoma_indicative_auroc"] for name in CRITERIA]
    reports["Derm7pt"]["criteria"]=criteria;reports["Derm7pt"]["melanoma_routes"]=routes
    reports["Derm7pt"]["seven_criterion_mean_auroc"]=float(np.mean(criterion_aucs))
    reports["Derm7pt"]["eight_output_mean_auroc"]=float(np.mean([routes["dedicated_melanoma_head"]["auroc"],*criterion_aucs]))
    ph2=rows["PH2"];ph2_truth=np.asarray([r["label"]=="MEL" for r in ph2],dtype=int)
    reports["PH2"]["dedicated_melanoma_head"]=binary_metrics(ph2_truth,[r["melanoma_probability"] for r in ph2])
    reports["PH2"]["supported_criteria"]={source:binary_metrics([r["criteria_targets"][source] for r in ph2],[r["binary_probabilities"][CRITERIA.index(target)+1] for r in ph2])
        for source,target in {"atypical_pigment_network":"pigment_network","atypical_dots_and_globules":"dots_and_globules","blue_whitish_veil_presence":"blue_whitish_veil","regression_presence":"regression_structures"}.items()}
    for dataset in ("Derm7pt","PH2"):
        reliability=[r for row in rows[dataset] for r in row.get("reliability_images",[])];reports[dataset]["reliability_flags"]={
            "n_images":len(reliability),"review_disposition":sum(r["disposition"]=="review" for r in reliability),
            "iqa_review_required":sum(r["iqa_state"]=="review_required" for r in reliability),
            "ood_flagged":sum(r["ood_state"]=="flagged" for r in reliability),
            "view_unstable":sum(r["view_stability_state"]=="unstable" for r in reliability)}
    combined_report,combined_rows=_bcn_combined_report(SEED+5);reports["BCN20000_combined"]=combined_report
    output={"schema_version":"dermpact-selected-final-results-v1","status":"PASS","protocol_sha256":protocol_sha,
        "datasets":reports,"row_counts":{name:len(value) for name,value in rows.items()},
        "scope":{"post_final_tuning_permitted":False,"agent_final_holdout":"not_run_no_predeclared_holdout_inventory"}}
    _atomic_json(OUTPUT/"results.json",output)
    for dataset,values in rows.items():_atomic_write(OUTPUT/f"{dataset.lower()}_lesion_rows.jsonl","".join(json.dumps(row,sort_keys=True)+"\n" for row in values))
    _atomic_write(OUTPUT/"bcn20000_combined_lesion_rows.jsonl","".join(json.dumps(row,sort_keys=True)+"\n" for row in combined_rows))
    return output


def dev_smoke() -> dict[str,Any]:
    from dermpact.core.inference_client import _resolve_confirmatory_cases
    case=sorted(_resolve_confirmatory_cases().values(),key=lambda item:item.group_id)[0];client=WorkerClient(timeout_seconds=1200)
    try:
        reference=client.predict(dermoscopic_path=case.dermoscopic_path,clinical_path=case.clinical_path,
            expected_dermoscopic_sha256=case.dermoscopic_sha256,expected_clinical_sha256=case.clinical_sha256,check_view=False)
        full=client.predict_final_full(dermoscopic_path=case.dermoscopic_path,clinical_path=case.clinical_path,
            expected_dermoscopic_sha256=case.dermoscopic_sha256,expected_clinical_sha256=case.clinical_sha256,check_view=False)
        diagnosis=client.predict_diagnosis_only(dermoscopic_path=case.dermoscopic_path,expected_dermoscopic_sha256=case.dermoscopic_sha256)
    finally:client.shutdown()
    if set(full["criterion_state_probabilities"])!=set(CRITERIA) or full["checklist_melanoma_probability"] is None:raise AssertionError("full route missing final-table outputs")
    if diagnosis.get("route")!="diagnosis_only" or len(diagnosis["raw_fold_outputs"])!=5:raise AssertionError("diagnosis-only route incomplete")
    parity=max(float(np.max(np.abs(np.asarray(reference[key])-np.asarray(full[key])))) for key in ("diagnosis_probabilities","binary_probabilities","sentinel_embedding"))
    if parity>1e-10:raise AssertionError(f"reporting adapter changed frozen predictions: {parity}")
    result={"status":"PASS","case_id":case.group_id,"reporting_adapter_max_abs_core_difference":parity,
        "reference_elapsed_seconds":reference["elapsed_seconds"],"full_elapsed_seconds":full["elapsed_seconds"],
        "diagnosis_only_elapsed_seconds":diagnosis["elapsed_seconds"],"official_or_final_rows_loaded":0}
    _atomic_json(BASE/"dev_smoke.json",result);return result


def run(*,resume:bool=False)->dict[str,Any]:
    _,protocol_sha=_load_protocol();entries=_access_entries()
    if (OUTPUT/"results.json").is_file():raise RuntimeError("final evaluation already completed; refusing a second run")
    if entries and not resume:raise RuntimeError("final access already started; use --resume after verifying this is the same interrupted run")
    _append_access("resume" if entries else "start",protocol_sha,"selected configuration final evaluation across Derm7pt, HAM, BCN and PH2")
    cases=resolve_final_cases();_run_predictions(cases,protocol_sha);result=score(cases,protocol_sha)
    _append_access("complete",protocol_sha,"all planned final predictions and scores written")
    return result


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-protocol",action="store_true");group.add_argument("--dev-smoke",action="store_true");group.add_argument("--run",action="store_true")
    parser.add_argument("--resume",action="store_true");parser.add_argument("--confirm-open-final",action="store_true")
    args=parser.parse_args(argv)
    if args.freeze_protocol:result=freeze_protocol()
    elif args.dev_smoke:result=dev_smoke()
    else:
        if not args.confirm_open_final:raise SystemExit("--run requires --confirm-open-final")
        result=run(resume=args.resume)
    print(json.dumps({"status":result.get("status"),"output":str(OUTPUT)},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
