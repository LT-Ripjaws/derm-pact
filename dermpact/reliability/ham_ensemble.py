"""Frozen HAM diagnosis-only ensemble policy; never apply to the paired Derm route."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax,expit,logit
from dermpact.eval.dermpact_phase3_calibration import digest
from dermpact.reliability.calibration import Temperature

CLASSES=['akiec','bcc','bkl','df','melanoma','nevus','vasc']
COLUMNS=[4,2,5,6,0,1,7]

def assess_ham_lesion(diagnosis_logits,melanoma_logits,parameters):
    """Inputs are all five diagnosis-only members on the same lesion images."""
    z=np.asarray(diagnosis_logits,dtype=float);m=np.asarray(melanoma_logits,dtype=float)
    if z.ndim!=3 or z.shape[0]!=5 or z.shape[1]<1 or z.shape[2]!=9 or m.shape!=z.shape[:2]:
        raise ValueError('require diagnosis [5,images,9] and melanoma [5,images] for one lesion')
    if not np.isfinite(z).all() or not np.isfinite(m).all():raise ValueError('nonfinite logits')
    if parameters['classes']!=CLASSES:raise ValueError('HAM label space mismatch')
    threshold=float(parameters['acceptance_threshold'])
    if not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('invalid threshold')
    p=softmax(z,axis=2)[:,:,COLUMNS];p/=p.sum(2,keepdims=True)
    p=p.mean(0).mean(0);b=float(expit(m).mean(0).mean(0))
    cp=Temperature(**parameters['diagnosis']).probabilities(np.log(np.clip(p,1e-12,1))[None,:])[0]
    cm=float(Temperature(**parameters['melanoma']).probabilities([logit(np.clip(b,1e-12,1-1e-12))])[0])
    q=np.clip(cm,1e-12,1-1e-12);ordered=np.sort(cp)
    risk=float(.5*(1-(ordered[-1]-ordered[-2]))+.5*(-q*np.log2(q)-(1-q)*np.log2(1-q)))
    return {'raw_diagnosis_probabilities':p.tolist(),'raw_melanoma_probability':b,
        'diagnosis_probabilities':cp.tolist(),'melanoma_probability':cm,'selective_risk':risk,
        'acceptance_threshold':threshold,'selective_abstention':'deferred' if risk>threshold else 'answered',
        'scope':'HAM seven-class global diagnosis-only lesion ensemble',
        'calibration_status':'ensemble temperature fitted; independent assessment showed mixed calibration effects',
        'coverage_status':'observed development coverage, not a per-case or future-cohort guarantee'}

def load_parameters(path):
    path=Path(path)
    if digest(path)!=Path(str(path)+'.sha256').read_text().strip():raise ValueError('HAM calibration integrity failure')
    return json.loads(path.read_text())
