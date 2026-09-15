"""Frozen raw-image inference averaging member-calibrated probabilities."""
import json
import os
import time
import numpy as np
import torch
from PIL import ImageOps
from scipy.special import softmax,expit
from dermpact.eval.dermpact_phase3_calibration import ROOT,digest
from dermpact.reliability.sentinels import sanitize_image
from dermpact.reliability.inference import load_selected_fold,calibrated_fold_probabilities,aggregate_fold_probabilities


class SelectedPerception:
    """One process-owned frozen prefix and sequential fold inference."""
    def __init__(self,threads=12):
        if not isinstance(threads,int) or threads<1:raise ValueError('positive thread count required')
        os.environ.update(HF_HOME=str(ROOT/'friend-cuda/model_cache'),HF_HUB_CACHE=str(ROOT/'friend-cuda/model_cache/hub'),
                          HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        torch.set_num_threads(threads)
        from dermpact.eval import dermpact_phase2_backbone_adaptation as adaptation
        from dermpact.eval import dermpact_phase2_5_training as s1
        self.adaptation=adaptation;self.s1=s1
        self.visual,self.preprocess,self.encoder_provenance=adaptation._load_hybrid_visual()
        self.visual.requires_grad_(False);s1._af._VISUAL_TEMPLATE=self.visual
        path=ROOT/'research/experiments/dermpact_phase3/calibration_v1/calibrators.json'
        if digest(path)!=type(path)(str(path)+'.sha256').read_text().strip():raise AssertionError('calibrators changed')
        self.calibrators=json.loads(path.read_text());self.calibration_sha256=digest(path)

    def _prefix(self,image,clinical):
        window=self.s1.window_cache
        global_tokens=self.adaptation._prefix_tokens(self.visual,self.preprocess(image).unsqueeze(0))
        padded,padding=window.aspect_preserve_pad(image)
        pixels=window.normalize_for_panDerm(window.tile_windows(padded))
        tokens=torch.cat([self.adaptation._prefix_tokens(self.visual,pixels[i:i+4]) for i in range(0,16,4)]).unsqueeze(0)
        clinical_tokens=(torch.zeros_like(global_tokens) if clinical is None else
                         self.adaptation._prefix_tokens(self.visual,self.preprocess(clinical).unsqueeze(0)))
        return (global_tokens,tokens,clinical_tokens,torch.from_numpy(window.compute_patch_valid_mask(padding)).unsqueeze(0),
                torch.from_numpy(window.window_positions()),torch.tensor([clinical is not None]))

    def predict(self,dermoscopy_bytes,clinical_bytes=None,check_view=True):
        started=time.perf_counter()
        image,_,receipt=sanitize_image(dermoscopy_bytes)
        clinical=None;clinical_receipt=None
        if clinical_bytes is not None:
            clinical,_,clinical_receipt=sanitize_image(clinical_bytes)
            if np.array_equal(np.asarray(image),np.asarray(clinical)):
                clinical=None;clinical_receipt={**clinical_receipt,'availability':'duplicate_dermoscopy_excluded'}
        outputs={};raws=[];views={};raw_views=[];model_bindings=[]
        with torch.inference_mode():
            prefixes=self._prefix(image,clinical)
            alternative=self._prefix(ImageOps.mirror(image),clinical) if check_view else None
            for fold in range(5):
                model,_,binding=load_selected_fold(fold)
                result=model(*prefixes)
                outputs[fold]=calibrated_fold_probabilities(result,fold,self.calibrators)
                raws.append({k:result[k].cpu().numpy()[0].tolist() for k in ['diagnosis','melanoma','criteria']})
                if alternative is not None:
                    changed=model(*alternative);views[fold]=calibrated_fold_probabilities(changed,fold,self.calibrators)
                    raw_views.append({k:changed[k].cpu().numpy()[0].tolist() for k in ['diagnosis','melanoma','criteria']})
                model_bindings.append(binding);del model
            sentinel_prefix=self.adaptation._prefix_tokens(self.visual,self.preprocess(image.resize((224,224))).unsqueeze(0))
            sentinel=self.adaptation._base_tail_features(self.visual,sentinel_prefix).cpu().numpy()
        p,b=aggregate_fold_probabilities(outputs)
        view_binary=aggregate_fold_probabilities(views)[1][0].tolist() if views else None
        return {'diagnosis_probabilities':p[0].tolist(),'binary_probabilities':b[0].tolist(),
                'alternative_binary_probabilities':view_binary,'sentinel_embedding':sentinel[0].tolist(),
                'raw_fold_outputs':raws,'raw_view_fold_outputs':raw_views,'checkpoint_bindings':model_bindings,
                'calibration_sha256':self.calibration_sha256,
                'image_receipt':receipt,'clinical_receipt':clinical_receipt,'clinical_present':clinical is not None,
                'elapsed_seconds':time.perf_counter()-started,
                'inference_policy':'five frozen models; arithmetic mean of independently calibrated probabilities',
                'ensemble_calibration_status':'not independently validated for paired Derm7pt route',
                'coverage_status':'member-wise conservative deferral; ensemble coverage not established',
                'view_policy':'horizontal reflection of dermoscopy, same clinical input'}
