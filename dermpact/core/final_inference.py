"""Reporting-only adapter around the frozen selected perception runtime."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from PIL import ImageOps
from scipy.special import expit, softmax

from dermpact.reliability.calibration import Temperature
from dermpact.reliability.inference import aggregate_fold_probabilities, calibrated_fold_probabilities, load_selected_fold
from dermpact.reliability.sentinels import sanitize_image


def predict_full(engine: Any, dermoscopy_bytes: bytes, clinical_bytes: bytes | None = None,
                 check_view: bool = True) -> dict[str, Any]:
    if not hasattr(engine, "visual"):
        result = engine.predict(dermoscopy_bytes, clinical_bytes, check_view)
        return {**result, "criterion_state_labels": {}, "criterion_state_probabilities": {},
                "checklist_melanoma_probability": None, "route": "full_reporting_adapter_fake"}
    started = time.perf_counter();image,_,receipt=sanitize_image(dermoscopy_bytes);clinical=None;clinical_receipt=None
    if clinical_bytes is not None:
        clinical,_,clinical_receipt=sanitize_image(clinical_bytes)
        if np.array_equal(np.asarray(image),np.asarray(clinical)):
            clinical=None;clinical_receipt={**clinical_receipt,"availability":"duplicate_dermoscopy_excluded"}
    outputs={};raws=[];views={};raw_views=[];bindings=[]
    state_probabilities={name:[] for name in engine.s1.CRITERIA};checklist=[]
    with torch.inference_mode():
        prefixes=engine._prefix(image,clinical);alternative=engine._prefix(ImageOps.mirror(image),clinical) if check_view else None
        for fold in range(5):
            model,_,binding=load_selected_fold(fold);result=model(*prefixes)
            outputs[fold]=calibrated_fold_probabilities(result,fold,engine.calibrators)
            raw_logits={name:result["raw_logits"][name].cpu().numpy()[0].tolist() for name in engine.s1.CRITERIA}
            for name,logits in raw_logits.items():state_probabilities[name].append(softmax(np.asarray(logits)).tolist())
            checklist.append(float(expit(result["checklist_melanoma"].cpu().numpy()[0])))
            raws.append({**{key:result[key].cpu().numpy()[0].tolist() for key in ("diagnosis","melanoma","criteria")},
                         "raw_logits":raw_logits,"checklist_melanoma":float(result["checklist_melanoma"].cpu().numpy()[0])})
            if alternative is not None:
                changed=model(*alternative);views[fold]=calibrated_fold_probabilities(changed,fold,engine.calibrators)
                raw_views.append({key:changed[key].cpu().numpy()[0].tolist() for key in ("diagnosis","melanoma","criteria")})
            bindings.append(binding);del model
        sentinel_prefix=engine.adaptation._prefix_tokens(engine.visual,engine.preprocess(image.resize((224,224))).unsqueeze(0))
        sentinel=engine.adaptation._base_tail_features(engine.visual,sentinel_prefix).cpu().numpy()
    diagnosis,binary=aggregate_fold_probabilities(outputs)
    state_labels={name:[label for label,_ in sorted(engine.s1.RAW_CLASS_MAPS[name].items(),key=lambda item:item[1])]
                  for name in engine.s1.CRITERIA}
    return {"diagnosis_probabilities":diagnosis[0].tolist(),"binary_probabilities":binary[0].tolist(),
        "criterion_state_labels":state_labels,
        "criterion_state_probabilities":{name:np.asarray(values).mean(0).tolist() for name,values in state_probabilities.items()},
        "checklist_melanoma_probability":float(np.mean(checklist)),
        "alternative_binary_probabilities":aggregate_fold_probabilities(views)[1][0].tolist() if views else None,
        "sentinel_embedding":sentinel[0].tolist(),"raw_fold_outputs":raws,"raw_view_fold_outputs":raw_views,
        "checkpoint_bindings":bindings,"calibration_sha256":engine.calibration_sha256,"image_receipt":receipt,
        "clinical_receipt":clinical_receipt,"clinical_present":clinical is not None,"elapsed_seconds":time.perf_counter()-started,
        "inference_policy":"frozen SelectedPerception computation with reporting-only raw-state/checklist exposure",
        "ensemble_calibration_status":"not independently validated for paired Derm7pt route",
        "coverage_status":"member-wise conservative deferral; ensemble coverage not established",
        "view_policy":"horizontal reflection of dermoscopy, same clinical input","route":"full_reporting_adapter"}


def predict_diagnosis_only(engine: Any, dermoscopy_bytes: bytes, check_view: bool = False) -> dict[str, Any]:
    if not hasattr(engine, "visual"):
        result=engine.predict(dermoscopy_bytes,None,check_view)
        return {**result,"binary_probabilities":result["binary_probabilities"][:1],"route":"diagnosis_only",
                "runtime_kind":"fake"}
    started=time.perf_counter();image,_,receipt=sanitize_image(dermoscopy_bytes)
    raw=[];view_raw=[];member_probabilities=[];member_melanoma=[];bindings=[]
    with torch.inference_mode():
        prefix=engine.adaptation._prefix_tokens(engine.visual,engine.preprocess(image).unsqueeze(0))
        view_prefix=(engine.adaptation._prefix_tokens(engine.visual,engine.preprocess(ImageOps.mirror(image)).unsqueeze(0)) if check_view else None)
        for fold in range(5):
            model,_,binding=load_selected_fold(fold);out=model.forward_diagnosis_only(prefix);diag=out["diagnosis"].cpu().numpy();mel=out["melanoma"].cpu().numpy()
            params=engine.calibrators["folds"][str(fold)]
            member_probabilities.append(Temperature(**params["temperature"]).probabilities(diag)[0])
            member_melanoma.append(float(Temperature(**params["binary_temperatures"][0]).probabilities(mel)[0]))
            raw.append({"diagnosis":diag[0].tolist(),"melanoma":float(mel[0])})
            if view_prefix is not None:
                alt=model.forward_diagnosis_only(view_prefix);view_raw.append({"diagnosis":alt["diagnosis"].cpu().numpy()[0].tolist(),"melanoma":float(alt["melanoma"].cpu().numpy()[0])})
            bindings.append(binding);del model
        sentinel=engine.adaptation._base_tail_features(engine.visual,prefix).cpu().numpy()[0]
    return {"diagnosis_probabilities":np.asarray(member_probabilities).mean(0).tolist(),"binary_probabilities":[float(np.mean(member_melanoma))],
        "raw_fold_outputs":raw,"raw_view_fold_outputs":view_raw,"sentinel_embedding":sentinel.tolist(),"checkpoint_bindings":bindings,
        "calibration_sha256":engine.calibration_sha256,"image_receipt":receipt,"clinical_receipt":None,"clinical_present":False,
        "elapsed_seconds":time.perf_counter()-started,"inference_policy":"five frozen Phase-2.5 global diagnosis-only members",
        "ensemble_calibration_status":"dataset-specific calibration must be applied by the evaluator",
        "coverage_status":"dataset-specific reliability policy required","view_policy":"horizontal reflection" if check_view else "not assessed",
        "route":"diagnosis_only"}
