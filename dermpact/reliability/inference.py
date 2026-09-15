"""Frozen Phase 2.5 fold reconstruction from the final selection manifest."""
import json
from pathlib import Path
import numpy as np
import torch
from dermpact.eval.dermpact_phase3_calibration import ROOT, digest


def load_selected_fold(fold):
    if fold not in range(5):raise ValueError('fold must be 0..4')
    selection_path=ROOT/'research/manifests/dermpact_phase2_final/selection_2026-09-06.json'
    if digest(selection_path)!=Path(str(selection_path)+'.sha256').read_text().strip():
        raise AssertionError('selection integrity failed')
    selection=json.loads(selection_path.read_text())
    from dermpact.eval import dermpact_phase2_5_training as stage1
    stage1._configure_training_metadata_root(ROOT/'friend-cuda')
    protocol,_=stage1.phase24b.load_protocol()
    _,cache=stage1.phase24b._live_inputs(protocol)
    paired=stage1.training.load_training_data(cache)
    model=stage1._build_phase25(fold,paired)
    model.unified_tail.unfreeze_tail()
    binding=selection['artifacts']['stage2']['fold_checkpoints'][fold]
    checkpoint=ROOT/binding['path']
    if digest(checkpoint)!=binding['sha256']:raise AssertionError('selected checkpoint changed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['fold']!=fold or state['completed_epochs']!=17 or state['swa_count']!=5:
        raise AssertionError('selected checkpoint incomplete')
    values=state['trainable_state']
    expected={name for name,p in model.named_parameters() if p.requires_grad}
    if set(values)!=expected:raise AssertionError('checkpoint keys do not match mutable selected state')
    incompat=model.load_state_dict(values,strict=False)
    if incompat.unexpected_keys:raise AssertionError('unexpected checkpoint keys')
    model.eval()
    model.requires_grad_(False)
    return model,paired,binding


def calibrated_fold_probabilities(output,fold,calibrators):
    """Future-case inference; assessment must instead use its crossfit parameters."""
    from dermpact.reliability.calibration import Temperature
    params=calibrators['folds'][str(fold)]
    def array(x):return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
    diagnosis=Temperature(**params['temperature']).probabilities(array(output['diagnosis']))
    scores=np.column_stack([array(output['melanoma']),array(output['criteria'])])
    binary=np.stack([Temperature(**params['binary_temperatures'][j]).probabilities(scores[:,j]) for j in range(8)],axis=1)
    return diagnosis,binary


def aggregate_fold_probabilities(outputs):
    """Equal probability ensemble, never weight averaging or per-case fold selection."""
    if set(outputs)!=set(range(5)):raise ValueError('all five fold outputs required')
    diag=np.stack([outputs[f][0] for f in range(5)])
    binary=np.stack([outputs[f][1] for f in range(5)])
    if not np.isfinite(diag).all() or not np.isfinite(binary).all():raise ValueError('nonfinite fold probabilities')
    if np.any(diag<0) or np.any(diag>1) or not np.allclose(diag.sum(-1),1):raise ValueError('invalid diagnosis probabilities')
    if np.any(binary<0) or np.any(binary>1):raise ValueError('invalid binary probabilities')
    return diag.mean(0),binary.mean(0)
