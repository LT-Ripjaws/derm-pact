"""Explicit runtime composition of separate reliability signals and recoveries."""
from dataclasses import asdict,dataclass
import numpy as np
from dermpact.reliability.calibration import risk_score
from dermpact.reliability.quality import describe,score_detector
from dermpact.reliability.sentinels import sanitize_image,cosine_ood,diagonal_distance


@dataclass(frozen=True)
class ReliabilityDecision:
    iqa_state: str
    ood_state: str
    view_stability_state: str
    selective_abstention: str
    disposition: str
    warnings: tuple
    scores: dict
    metadata_receipt: dict

    def to_dict(self):return asdict(self)


def assess_input(payload, *, quality_parameters, sentinel_embedding, sentinel_reference,
                 sentinel_threshold, diagnosis_probabilities, binary_probabilities,
                 acceptance_threshold, score_method='cosine_nearest', alternative_binary=None,
                 stability_threshold=.10, view_baseline_binary=None):
    """Missing assessments are explicit and require review; no invented PASS state."""
    image,_,receipt=sanitize_image(payload)
    warnings=[];scores={}
    if quality_parameters is None:
        iqa='not_assessed'
    else:
        scores['synthetic_corruption_probability']=float(score_detector(describe(image)[None,:],quality_parameters)[0])
        threshold=float(quality_parameters['threshold'])
        if not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('invalid quality threshold')
        iqa='review_required' if scores['synthetic_corruption_probability']>threshold else 'not_flagged'
    if sentinel_embedding is None or sentinel_reference is None:
        ood='not_assessed'
    else:
        if not np.isfinite(sentinel_threshold):raise ValueError('invalid OOD threshold')
        methods={'cosine_nearest':cosine_ood,'diagonal_distance':diagonal_distance}
        if score_method not in methods:raise ValueError('unknown OOD score')
        values=methods[score_method](sentinel_reference,np.asarray(sentinel_embedding).reshape(1,-1))
        scores['unfamiliarity']=float(values[0]);ood='flagged' if values[0]>sentinel_threshold else 'not_flagged'
    p=np.asarray(diagnosis_probabilities).reshape(1,-1);b=np.asarray(binary_probabilities).reshape(1,-1)
    scores['selective_risk']=float(risk_score(p,b)[0])
    if not np.isfinite(acceptance_threshold) or not 0<=acceptance_threshold<=1:raise ValueError('invalid acceptance threshold')
    selective='deferred' if scores['selective_risk']>acceptance_threshold else 'answered'
    if alternative_binary is None:
        stability='not_assessed'
    else:
        alt=np.asarray(alternative_binary,dtype=float)
        if alt.ndim!=2 or alt.shape[1]!=8 or not len(alt) or not np.isfinite(alt).all() or np.any(alt<0) or np.any(alt>1):
            raise ValueError('alternative view probabilities must be finite [V,8]')
        if not np.isfinite(stability_threshold) or not 0<=stability_threshold<=1:raise ValueError('invalid stability threshold')
        view_base=b if view_baseline_binary is None else np.asarray(view_baseline_binary,dtype=float).reshape(1,-1)
        if view_base.shape!=(1,8) or not np.isfinite(view_base).all() or np.any(view_base<0) or np.any(view_base>1):
            raise ValueError('invalid baseline view probabilities')
        scores['maximum_view_shift']=float(np.abs(alt-view_base).max())
        stability='unstable' if scores['maximum_view_shift']>stability_threshold else 'not_flagged'
    for condition,code,reason,action in [
        (iqa=='review_required','image_quality','Specified corruption detector flagged image','review_image_or_reacquire'),
        (ood=='flagged','unfamiliar_image','Frozen reference-bank score exceeded operating threshold; advisory only','display_unfamiliarity_warning'),
        (stability=='unstable','view_instability','Predictions vary across assessed views','review_image_or_reacquire'),
        (selective=='deferred','selective_deferral','Predictive uncertainty exceeds operating threshold','clinician_review'),
        (ood=='not_assessed','unfamiliarity_unassessed','Optional unfamiliarity assessment unavailable','display_unfamiliarity_warning'),
        ('not_assessed' in [iqa,stability],'incomplete_reliability','A required reliability check was not assessed','complete_assessment_or_clinician_review'),
    ]:
        if condition:warnings.append({'code':code,'reason':reason,'recovery_action':action,
                                      'remaining_usable_output':'raw model evidence, source provenance and clinician-supplied facts'})
    blocking=any(w['code'] not in ('unfamiliar_image','unfamiliarity_unassessed') for w in warnings)
    return ReliabilityDecision(iqa,ood,stability,selective,'review' if blocking else 'display',tuple(warnings),scores,receipt)
