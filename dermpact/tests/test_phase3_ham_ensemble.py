import numpy as np
import pytest
from scipy.special import softmax,expit
from dermpact.reliability.ham_ensemble import assess_ham_lesion,CLASSES,COLUMNS

PARAMS={'classes':CLASSES,'diagnosis':{'value':1.,'status':'identity'},
        'melanoma':{'value':1.,'status':'identity'},'acceptance_threshold':.9}

def test_exact_raw_aggregation_identity_and_no_target_input():
    rng=np.random.default_rng(93);z=rng.normal(size=(5,3,9));m=rng.normal(size=(5,3))
    result=assess_ham_lesion(z,m,PARAMS)
    p=softmax(z,axis=2)[:,:,COLUMNS];p/=p.sum(2,keepdims=True)
    assert np.allclose(result['diagnosis_probabilities'],p.mean(0).mean(0),atol=1e-12)
    assert np.isclose(result['melanoma_probability'],expit(m).mean())

def test_rejects_missing_member_and_wrong_label_space():
    z=np.zeros((5,1,9));m=np.zeros((5,1))
    with pytest.raises(ValueError):assess_ham_lesion(z[:4],m[:4],PARAMS)
    with pytest.raises(ValueError):assess_ham_lesion(z,m,{**PARAMS,'classes':list(reversed(CLASSES))})
    with pytest.raises(ValueError):assess_ham_lesion(z,np.full_like(m,np.nan),PARAMS)

def test_exact_threshold_tie_is_answered():
    z=np.zeros((5,1,9));m=np.zeros((5,1))
    risk=assess_ham_lesion(z,m,PARAMS)['selective_risk']
    assert assess_ham_lesion(z,m,{**PARAMS,'acceptance_threshold':risk})['selective_abstention']=='answered'
    assert assess_ham_lesion(z,m,{**PARAMS,'acceptance_threshold':risk-1e-6})['selective_abstention']=='deferred'
