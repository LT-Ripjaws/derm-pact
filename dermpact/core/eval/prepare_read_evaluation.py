"""Freeze agent-only held-out cases, OOF evidence, and independent reference facts."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import expit, logsumexp, softmax

from dermpact.core import memory
from dermpact.reliability.calibration import Temperature, risk_score, fit_acceptance_threshold
from dermpact.reliability.quality import describe, score_detector
from dermpact.eval.dermpact_phase2_4b_inputs import parse_prediagnostic_facts
from dermpact.core.agent.redesign.evidence import digest


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/experiments/dermpact_agent_redesign/evaluation_v2"
META = ROOT / "friend-cuda/research/manifests"
INVENTORY = META / "dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json"
FOLDS = META / "dermpact_phase2_1/dermpact_phase2_1_development_folds_v1.json"
OOF = ROOT / "research/experiments/dermpact_phase2_5/stage2_cefix_v1/stage2_oof.npz"
CAL = ROOT / "research/experiments/dermpact_phase3/calibration_v1/assessment_predictions.npz"
FEATURE_DIR = ROOT / "research/experiments/dermpact_phase2_1/features/panderm/derm7pt"
REFLECTION_DIR = ROOT / "research/experiments/dermpact_phase3/reflection_all_development_v1"
QUALITY_DIR = ROOT / "research/experiments/dermpact_phase3/quality_detector_v1"
BANK_DIR = ROOT / "research/experiments/dermpact_phase6/memory_bank_dev_v1"
CLASSES = ("melanoma", "nevus", "basal_cell_carcinoma", "benign_keratosis", "miscellaneous")
CRITERIA = ("pigment_network", "streaks", "pigmentation", "regression_structures", "dots_and_globules", "blue_whitish_veil", "vascular_structures")
PROFILE = "dermpact-agent-oof-disjoint-calibration-v1"
QUESTIONS = {
    "evidence_report": "Summarize the model's diagnosis, separate melanoma estimate, and the two strongest predicted dermoscopic findings. Distinguish model predictions from confirmed clinical observations.",
    "reliability_explanation": "Was this lesion referred for clinician review? Explain each recorded referral trigger and distinguish any advisory-only warning.",
    "referral_and_comparison": "Why was this lesion referred, and does the similar historical case change that?",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked(path):
    actual = sha(path)
    sidecar = Path(str(path) + ".sha256")
    if sidecar.exists() and sidecar.read_text().strip() != actual:
        raise ValueError(f"source hash mismatch: {path}")
    return actual


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    Path(str(path) + ".sha256").write_text(sha(path) + "\n", encoding="ascii")


def exposure_audit():
    roots = ["research/manifests", "research/experiments/agentic_loops", "research/experiments/dermpact_phase6",
             "research/experiments/dermpact_phase7", "research/experiments/dermpact_agent_redesign"]
    proc = subprocess.run(["rg", "--files", *roots, "-g", "**/task_cache/**/*.json", "-g", "*transcript*.jsonl"],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    paths = sorted({ROOT / p for p in proc.stdout.splitlines() if p.strip()})
    rx = re.compile(r"derm7pt(?:-case)?-(\d+)")
    seen, sources = set(), []
    for path in paths:
        ids = {f"derm7pt-case-{int(n)}" for n in rx.findall(path.read_text(encoding="utf-8"))}
        seen.update(ids)
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), "case_ids": sorted(ids)})
    # Explicit prior inventory is excluded even if a generation log is absent.
    old = ROOT / "research/experiments/dermpact_phase7/confirmatory_inventory100_v1/inventory_manifest.json"
    seen.update(r["case_id"] for r in json.loads(old.read_text())["rows"])
    return seen, {"roots": roots, "patterns": ["task_cache JSON", "transcript JSONL"],
                  "identifier_formats": ["derm7pt-case-N", "derm7pt-N"], "sources": sources,
                  "prior_inventory_sha256": checked(old),
                  "limitation": "No known exposure in the audited local logs/caches; undocumented external use cannot be excluded. Perception model selection already used this development pool."}


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    rows = json.loads(INVENTORY.read_text())["records"]
    assert len(rows) == 616 and all(r["phase1_role"] == "development" for r in rows)
    ids = [r["group_id"] for r in rows]
    assert len(set(ids)) == 616
    index = {cid: i for i, cid in enumerate(ids)}
    folds = np.array([r["development_fold"] for r in rows])
    locked_folds = {r["group_id"]: r["fold"] for r in json.loads(FOLDS.read_text())["derm7pt"]["folds"]}
    assert all(locked_folds[cid] == int(folds[i]) for i, cid in enumerate(ids))
    exposed, audit = exposure_audit()
    bank_old = memory.MemoryBank.load(BANK_DIR)
    bank_ids = {e.case_id for e in bank_old.episodes}
    quality_rows = json.loads((QUALITY_DIR / "rows.json").read_text())
    quality_seen = {r["id"] for r in quality_rows if r.get("dataset") == "Derm7pt"}
    excluded = exposed | bank_ids | quality_seen
    excluded_hashes = {r["modalities"][m].get("sha256") for r in rows if r["group_id"] in excluded
                       for m in ("dermoscopic", "clinical")} - {None}
    duplicate_excluded = {r["group_id"] for r in rows if any(r["modalities"][m].get("sha256") in excluded_hashes
                                                          for m in ("dermoscopic", "clinical"))}
    excluded |= duplicate_excluded
    rank = lambda cid: hashlib.sha256((PROFILE + ":" + cid).encode()).hexdigest()
    roles = {"evaluation": [], "development": [], "calibration_fit": [], "threshold_selection": []}
    for fold in range(5):
        pool = sorted((cid for cid in ids if folds[index[cid]] == fold and cid not in excluded), key=rank)
        if len(pool) < 25:
            raise ValueError(f"insufficient unexposed candidates in fold {fold}")
        roles["evaluation"] += pool[:20]
        roles["development"] += pool[20:25]
        remaining = sorted((cid for cid in ids if folds[index[cid]] == fold and cid not in pool[:25]), key=rank)
        roles["calibration_fit"] += remaining[:30]
        roles["threshold_selection"] += remaining[30:50]
    sets = {k: set(v) for k, v in roles.items()}
    assert all(not a & b for i, a in enumerate(sets.values()) for b in list(sets.values())[i+1:])
    assert not sets["evaluation"] & excluded
    # Cross-role image identities, including paired photographs, must not overlap.
    image_roles = {}
    for role, members in roles.items():
        for cid in members:
            for modality in rows[index[cid]]["modalities"].values():
                h = modality.get("sha256")
                if h:
                    prior = image_roles.setdefault(h, role)
                    if prior != role:
                        raise ValueError("cross-role duplicate image; revise duplicate grouping before preparation")
    source_paths = [INVENTORY, FOLDS, OOF, CAL, FEATURE_DIR / "feature_manifest.json", FEATURE_DIR / "features.npy",
                    QUALITY_DIR / "parameters.json", QUALITY_DIR / "rows.json", BANK_DIR / "bank.jsonl", BANK_DIR / "manifest.json"]
    source_paths += [REFLECTION_DIR / f"fold_{f}.json" for f in range(5)]
    source_paths += [Path(__file__),ROOT/'dermpact/eval/dermpact_phase2_4b_inputs.py',
                     ROOT/'dermpact/eval/dermpact_phase2_5_stage2_training.py',ROOT/'dermpact/reliability/calibration.py',
                     ROOT/'dermpact/reliability/quality.py',ROOT/'dermpact/core/memory.py']
    bindings = {str(p.relative_to(ROOT)): checked(p) for p in source_paths}
    write(OUT / "exposure_audit.json", {**audit, "excluded_development_case_ids": sorted(set(ids) & excluded)})
    write(OUT / "roles.json", {"roles": roles, "bank_cases": sorted(bank_ids),
          "historically_exposed_development": sorted(set(ids) & exposed),
          "counts": {k: len(v) for k, v in roles.items()}, "perception_training": "Every case uses its original held-out fold; training used other folds.",
          "note": "Calibration roles may contain historically exposed cases; evaluation/development reservations exclude known exposure, bank membership, and IQA-source cases."})
    write(OUT / "protocol.json", {"profile": PROFILE, "preparation_version":"v2_normalized_metadata", "state": "ROLES_FROZEN_BEFORE_NEW_CALIBRATION_AND_GENERATION",
          "roles_sha256": sha(OUT / "roles.json"), "source_bindings": bindings,
          "selection": "outcome-independent SHA256 rank within original folds; 20 evaluation and 5 development cases per fold",
          "calibration": "30 disjoint cases per fold for temperature fitting; 20 others for risk/reflection thresholds; no evaluation fitting",
          "risk_quantile": 0.90, "view_shift_quantile": 0.95, "ood_quantile": 0.95,
          "question_families": QUESTIONS, "group_unit": "Derm7pt case/lesion proxy; patient identifiers unavailable",
          "scope": "agent-only held-out development cases, not external or model-selection-independent clinical validation",
          "generator_calls_performed": 0, "main_evaluation_execution": "not authorized by this preparation step"})
    raw = np.load(OOF, allow_pickle=False)
    assessed = np.load(CAL, allow_pickle=False)
    assert list(assessed["ids"].astype(str)) == ids
    assert np.array_equal(assessed["folds"], folds)
    z9 = raw["derm_diagnosis_logits"]
    z = np.column_stack([z9[:, 0], z9[:, 1], z9[:, 2], z9[:, 5], logsumexp(z9[:, [3,4,6,7,8]], axis=1)])
    binary = np.column_stack([raw["derm_melanoma_logits"], raw["derm_criteria_logits"]])
    y = np.array([("MEL", "NEV", "BCC", "SK", "MISC").index(r["diagnosis"]["harmonized5"]) for r in rows])
    binary_y = assessed["binary_labels"]
    reflected, checkpoints = {}, {}
    for fold in range(5):
        payload = json.loads((REFLECTION_DIR / f"fold_{fold}.json").read_text())
        checkpoints[str(fold)] = payload["checkpoint"]
        for r in payload["rows"]:
            i = index[r["id"]]
            assert int(folds[i]) == fold == r["fold"] and r["index"] == i
            clean = r["conditions"]["clean_cached"]
            assert np.allclose(clean["diagnosis"], z9[i], atol=1e-5)
            assert np.allclose([clean["melanoma"], *clean["criteria"]], binary[i], atol=1e-5)
            value = r["conditions"]["horizontal_reflection"]
            reflected[r["id"]] = [value["melanoma"], *value["criteria"]]
    assert len(reflected) == 616
    p = np.zeros((616,5)); bp = np.zeros((616,8)); shifts = np.zeros(616)
    params = {}
    for fold in range(5):
        fit = [index[c] for c in roles["calibration_fit"] if folds[index[c]] == fold]
        select = [index[c] for c in roles["threshold_selection"] if folds[index[c]] == fold]
        target = np.flatnonzero(folds == fold)
        t = Temperature.fit(z[fit], y[fit])
        ts = [Temperature.fit(binary[fit,j], binary_y[fit,j]) for j in range(8)]
        p[target] = t.probabilities(z[target])
        for j in range(8): bp[target,j] = ts[j].probabilities(binary[target,j])
        rb = np.array([reflected[ids[i]] for i in target])
        rp = np.column_stack([ts[j].probabilities(rb[:,j]) for j in range(8)])
        shifts[target] = np.max(np.abs(bp[target]-rp), axis=1)
        params[str(fold)] = {"temperature": asdict(t), "binary_temperatures": [asdict(ti) for ti in ts],
            "fit_cases": [ids[i] for i in fit], "threshold_cases": [ids[i] for i in select],
            "risk_threshold": fit_acceptance_threshold(risk_score(p[select],bp[select]),.90),
            "view_shift_threshold": float(np.quantile(shifts[select],.95,method="higher")),
            "checkpoint": checkpoints[str(fold)]}
    write(OUT / "calibrators.json", params)
    fm = json.loads((FEATURE_DIR / "feature_manifest.json").read_text())
    feature_identity = [{"id": r["group_id"], "image_sha256": r["modalities"]["dermoscopic"]["sha256"]} for r in rows]
    assert digest(feature_identity) == fm["record_identity_sha256"]
    assert sha(FEATURE_DIR / "features.npy") == fm["feature_file_sha256"] and fm["n_records"] == 616
    features = np.load(FEATURE_DIR / "features.npy", allow_pickle=False)
    assert features.shape == (616,512) and np.isfinite(features).all()
    features = features / np.linalg.norm(features,axis=1,keepdims=True)
    bank_model = fm["model_id"] + ":phase2-1-preprocessing:" + sha(FEATURE_DIR / "feature_manifest.json")[:16]
    episodes = [memory.MemoryEpisode.create(case_id=e.case_id, embedding=features[index[e.case_id]], embedding_model=bank_model,
                image_sha256=e.image_sha256, outcome=e.outcome, author="frozen_agent_evaluation_preparation",
                documented_criterion_values=e.documented_criterion_values, original_model_record=e.original_model_record,
                field_provenance=e.field_provenance, site=e.site, sex=e.sex, elevation=e.elevation,
                event_time="2026-09-20T00:00:00Z") for e in bank_old.episodes]
    bank = memory.MemoryBank.from_episodes(episodes, embedding_model=bank_model)
    bx = np.array([e.embedding for e in episodes]); similarity = bx @ bx.T; np.fill_diagonal(similarity,-np.inf)
    threshold = max(.5, round(float(np.quantile(similarity.max(axis=1),.25)),3))
    derivation = {"method": "bank-only leave-one-out nearest cosine, p25 rounded to 3 decimals, floor .5; counterexample minus .05",
                  "min_similarity": threshold, "counterexample_min_similarity": max(.5,threshold-.05)}
    bank.write(OUT / "bank", source={"old_bank_sha256": bank_old.bank_sha256, "feature_manifest_sha256": sha(FEATURE_DIR / "feature_manifest.json")},
               n_embedded=50,n_development_cases=616,threshold_derivation=derivation)
    bank = memory.MemoryBank.load(OUT / "bank")
    policy = memory.ComparisonPolicy.from_manifest(OUT / "bank")
    fit_ix = [index[c] for c in roles["calibration_fit"]]
    sel_ix = [index[c] for c in roles["threshold_selection"]]
    ood = 1 - (features @ features[fit_ix].T).max(axis=1)
    ood_threshold = float(np.quantile(ood[sel_ix],.95,method="higher"))
    quality_parameters = json.loads((QUALITY_DIR / "parameters.json").read_text())
    risks = risk_score(p,bp)
    image_root = (ROOT / "release_v0/images").resolve()
    outputs, summary = {}, {"roles": {k:len(v) for k,v in roles.items()}, "ood_threshold":ood_threshold,
                           "bank_threshold":threshold,"evaluation_generations":0,"images_opened":0}
    for role in ("development","evaluation"):
        tasks=[]
        for cid in roles[role]:
            i=index[cid]; row=rows[i]; fold=int(folds[i]); cp=params[str(fold)]
            image_path=(image_root / row["modalities"]["dermoscopic"]["path"]).resolve()
            if not image_path.is_relative_to(image_root) or sha(image_path)!=row["modalities"]["dermoscopic"]["sha256"]:
                raise ValueError("image path/hash not in the frozen development allowlist")
            with Image.open(image_path) as image:
                quality=float(score_detector(describe(image)[None,:],quality_parameters)[0])
            summary["images_opened"]+=1
            warnings=[]
            for code, value, cutoff, reason in [
                ("uncertainty",float(risks[i]),cp["risk_threshold"],"Held-out model uncertainty exceeded its selected threshold"),
                ("view_instability",float(shifts[i]),cp["view_shift_threshold"],"Horizontal-reflection probability change exceeded its selected threshold"),
                ("image_quality",quality,quality_parameters["threshold"],"The frozen synthetic-corruption detector exceeded its warning threshold")]:
                if value>cutoff: warnings.append({"code":code,"reason":reason,"score":value,"threshold":cutoff})
            referred=bool(warnings)
            reliability={"disposition":"review" if referred else "display","selective_abstention":"deferred" if referred else "answered",
                "warnings":warnings,"ood_state":"flagged" if ood[i]>ood_threshold else "not_flagged",
                "ood_score":float(ood[i]),"ood_threshold":ood_threshold,"ood_advisory_only":True,
                "iqa_state":"review_required" if quality>quality_parameters["threshold"] else "not_flagged",
                "image_quality_score":quality,"quality_meaning":"specified synthetic corruption, not clinician-rated usability",
                "risk":float(risks[i]),"risk_threshold":cp["risk_threshold"],"view_shift":float(shifts[i]),"view_threshold":cp["view_shift_threshold"],
                "prediction_mode":"single held-out fold; separate calibration-fit and threshold-selection cases"}
            predicted=CLASSES[int(np.argmax(p[i]))]
            facts=parse_prediagnostic_facts(row)
            perception={"diagnosis":predicted,"diagnosis_probabilities":dict(zip(CLASSES,map(float,p[i]))),
                        "melanoma_probability":float(bp[i,0]),"criterion_positive_probabilities":dict(zip(CRITERIA,map(float,bp[i,1:]))),
                        "evidence_kind":"model predictions, not confirmed clinician observations"}
            query={"case_id":cid,"data_role":memory.DATA_ROLE_DEVELOPMENT,"embedding":features[i],"embedding_model":bank_model,
                   "image_sha256":row["modalities"]["dermoscopic"]["sha256"],"provisional_class":predicted,
                   "site":facts.site,"sex":facts.sex,"elevation":facts.elevation,
                   "criterion_values":{},"identity":{"lesion_id":cid,"patient_id":None}}
            comparison=memory.compare(query,bank,policy).to_dict()
            comparison["criterion_comparison_scope"]="Categorical query criteria not supplied; these fields remain unknown."
            snapshot={"case_id":cid,"role":role,"profile":PROFILE,"perception":perception,"reliability":reliability,"comparison":comparison,
                      "provenance":{"fold":fold,"checkpoint":checkpoints[str(fold)],"oof_sha256":bindings[str(OOF.relative_to(ROOT))],
                      "fold_manifest_sha256":bindings[str(FOLDS.relative_to(ROOT))],"calibrators_sha256":sha(OUT/'calibrators.json'),
                      "calibration_fit_overlap":False,"threshold_selection_overlap":False,"known_llm_exposure":False,
                      "patient_disjointness_verified":False,"perception_selection_independent":False}}
            snapshot["snapshot_sha256"]=digest(snapshot)
            file=OUT/role/"cases"/(cid+".json");write(file,snapshot);outputs[str(file.relative_to(OUT))]=sha(file)
            for family,question in QUESTIONS.items():
                tasks.append({"task_id":cid+"::"+family,"case_id":cid,"family":family,"question":question,
                              "case_path":str(file.relative_to(OUT)),"case_sha256":sha(file)})
        write(OUT/role/"tasks.json",tasks);outputs[str((OUT/role/'tasks.json').relative_to(OUT))]=sha(OUT/role/'tasks.json')
    summary.update({"known_exposure_exclusions":len(set(ids)&excluded),"evaluation_bank_overlap":0,
                    "evaluation_calibration_overlap":0,"evaluation_threshold_overlap":0,"source_bindings":bindings,
                    "artifact_hashes":outputs,"calibration_profile_changed":True,
                    "interpretation":"100 agent-evaluation cases and 25 development cases reserved; no agent evaluation answers generated."})
    write(OUT/"manifest.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in {'source_bindings','artifact_hashes'}},indent=2))


if __name__=="__main__":
    prepare()
