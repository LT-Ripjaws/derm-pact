"""Load frozen, hash-verified Phase 3 policy artifacts for downstream use."""
import json
from pathlib import Path
import numpy as np
from dermpact.eval.dermpact_phase3_calibration import digest
from dermpact.reliability.policy import assess_input


class ReliabilityBundle:
    def __init__(self, manifest_path):
        path=Path(manifest_path).resolve()
        if digest(path)!=Path(str(path)+'.sha256').read_text().strip():raise ValueError('reliability bundle manifest changed')
        self.manifest=json.loads(path.read_text())
        root=path.parents[3]
        for relative,expected in self.manifest['source_bindings'].items():
            source=(root/relative).resolve()
            if not source.is_relative_to(root) or digest(source)!=expected:
                raise ValueError('reliability source drift')
        self.artifacts={}
        for name,binding in self.manifest['artifacts'].items():
            artifact=(root/binding['path']).resolve()
            if not artifact.is_relative_to(root) or digest(artifact)!=binding['sha256']:
                raise ValueError(f'reliability artifact integrity failure: {name}')
            self.artifacts[name]=artifact
        self.quality=json.loads(self.artifacts['quality_parameters'].read_text())
        self.calibration=json.loads(self.artifacts['derm_calibrators'].read_text())
        data=np.load(self.artifacts['ood_reference_bank'],allow_pickle=False)
        self.bank=data['features'].copy()
        data.close()
        self.threshold=float(self.manifest['ood']['threshold'])

    def assess(self,payload,*,encode_sanitized_image,diagnosis_probabilities,binary_probabilities,
               acceptance_threshold,alternative_binary=None,stability_fold=None,view_baseline_binary=None):
        from dermpact.reliability.sentinels import sanitize_image
        image,_,_=sanitize_image(payload)
        embedding=np.asarray(encode_sanitized_image(image),dtype=float)
        view_policy=self.manifest['view_stability']
        if 'per_fold_thresholds' in view_policy:
            if alternative_binary is not None and (stability_fold not in range(5) or view_baseline_binary is None):
                raise ValueError('fold-specific raw view baseline is required for the v2 stability policy')
            threshold=1. if stability_fold is None else view_policy['per_fold_thresholds'][str(stability_fold)]['threshold']
        else:
            threshold=view_policy['threshold']
        decision=assess_input(payload,quality_parameters=self.quality,sentinel_embedding=embedding,
                              sentinel_reference=self.bank,sentinel_threshold=self.threshold,
                              diagnosis_probabilities=diagnosis_probabilities,binary_probabilities=binary_probabilities,
                              acceptance_threshold=acceptance_threshold,alternative_binary=alternative_binary,
                              stability_threshold=threshold,view_baseline_binary=view_baseline_binary)
        return decision

    def assess_selected_prediction(self,payload,prediction):
        """Conservative ensemble gate: every member must pass its own fitted gates."""
        from dataclasses import replace
        from scipy.special import expit
        from dermpact.reliability.calibration import Temperature,risk_score
        from dermpact.reliability.inference import aggregate_fold_probabilities
        from dermpact.reliability.sentinels import sanitize_image
        _,_,receipt=sanitize_image(payload)
        if receipt['input_sha256']!=prediction['image_receipt']['input_sha256'] or receipt['sanitized_sha256']!=prediction['image_receipt']['sanitized_sha256']:
            raise ValueError('prediction/image identity mismatch')
        if prediction['calibration_sha256']!=digest(self.artifacts['derm_calibrators']):
            raise ValueError('prediction/calibration version mismatch')
        raw=prediction['raw_fold_outputs'];views=prediction.get('raw_view_fold_outputs',[])
        if len(raw)!=5 or (views and len(views)!=5):raise ValueError('all five members required')
        output={};risk_values=[];deferred=[];view_scores=[];view_flags=[]
        policies=self.manifest['view_stability']['per_fold_thresholds']
        for fold in range(5):
            params=self.calibration['folds'][str(fold)]
            p=Temperature(**params['temperature']).probabilities(np.array([raw[fold]['diagnosis']]))
            z=np.array([raw[fold]['melanoma'],*raw[fold]['criteria']])
            b=np.array([[Temperature(**params['binary_temperatures'][j]).probabilities([z[j]])[0] for j in range(8)]])
            output[fold]=(p,b);r=float(risk_score(p,b)[0]);risk_values.append(r);deferred.append(r>params['acceptance_threshold'])
            if views:
                alt=np.array([views[fold]['melanoma'],*views[fold]['criteria']])
                if alt.shape!=(8,) or not np.isfinite(alt).all():raise ValueError('invalid alternate-view logits')
                shift=float(np.max(np.abs(expit(z)-expit(alt))))
                view_scores.append(shift);view_flags.append(shift>policies[str(fold)]['threshold'])
        p,b=aggregate_fold_probabilities(output)
        if not np.allclose(p[0],prediction['diagnosis_probabilities'],atol=1e-10,rtol=0) or not np.allclose(b[0],prediction['binary_probabilities'],atol=1e-10,rtol=0):
            raise ValueError('aggregate probabilities do not reproduce')
        result=assess_input(payload,quality_parameters=self.quality,sentinel_embedding=prediction['sentinel_embedding'],
            sentinel_reference=self.bank,sentinel_threshold=self.threshold,diagnosis_probabilities=p[0],
            binary_probabilities=b[0],acceptance_threshold=1.)
        warnings=[w for w in result.warnings if w['code']!='incomplete_reliability']
        for flag,code,reason in [(any(deferred),'selective_deferral','At least one member exceeds its fitted uncertainty threshold'),
                                 (bool(views) and any(view_flags),'view_instability','At least one member exceeds its fitted reflection threshold'),
                                 (not views,'incomplete_reliability','Alternate-view inference was not assessed')]:
            if flag:warnings.append({'code':code,'reason':reason,'recovery_action':'clinician_review',
                                      'remaining_usable_output':'raw evidence and member-calibrated probability averages; ensemble calibration unvalidated'})
        return replace(result,selective_abstention='deferred' if any(deferred) else 'answered',
            view_stability_state=('unstable' if any(view_flags) else 'not_flagged') if views else 'not_assessed',
            warnings=tuple(warnings),disposition='review' if any(w['code'] not in ('unfamiliar_image','unfamiliarity_unassessed') for w in warnings) else 'display',
            scores={**result.scores,'member_uncertainty':risk_values,'member_view_shifts':view_scores})
