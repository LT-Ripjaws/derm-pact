"""Phase 3 development calibration/deferral on selected-model held-out scores."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import roc_auc_score

from dermpact.reliability.calibration import (
    Temperature, calibration_metrics, risk_score, risk_coverage,
    fit_acceptance_threshold, conformal_quantile,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'research/experiments/dermpact_phase3/calibration_v1'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    body = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding='utf-8')
    Path(str(path) + '.sha256').write_text(digest(path) + '\n')


def fixed_halves(groups):
    """Outcome-independent balanced hash split; groups stay intact."""
    groups = np.asarray(groups).astype(str)
    unique = sorted(set(groups), key=lambda g: hashlib.sha256(('phase3-v1:' + g).encode()).hexdigest())
    mapping = {g: i % 2 for i, g in enumerate(unique)}
    return np.array([mapping[g] for g in groups])


def crossfit(logits, binary_logits, labels, binary_labels, folds, groups):
    n = len(labels)
    calibrated = np.full_like(logits, np.nan, dtype=float)
    binary = np.full_like(binary_logits, np.nan, dtype=float)
    accepted = np.zeros(n, dtype=bool)
    risk = np.full(n, np.nan)
    sets = np.zeros_like(logits, dtype=bool)
    receipts, deployment = [], {}
    for fold in sorted(set(folds)):
        ids = np.flatnonzero(folds == fold)
        halves = fixed_halves(groups[ids])
        if len(ids) < 4:
            raise ValueError('insufficient held-out rows for calibration crossfit')
        for half in [0, 1]:
            fit = ids[halves != half]
            test = ids[halves == half]
            if set(groups[fit]) & set(groups[test]):
                raise AssertionError('calibration group overlap')
            t = Temperature.fit(logits[fit], labels[fit])
            ts = [Temperature.fit(binary_logits[fit, j], binary_labels[fit, j]) for j in range(8)]
            pfit, ptest = t.probabilities(logits[fit]), t.probabilities(logits[test])
            bfit = np.stack([ts[j].probabilities(binary_logits[fit, j]) for j in range(8)], axis=1)
            btest = np.stack([ts[j].probabilities(binary_logits[test, j]) for j in range(8)], axis=1)
            threshold = fit_acceptance_threshold(risk_score(pfit, bfit))
            calibrated[test], binary[test] = ptest, btest
            risk[test] = risk_score(ptest, btest)
            accepted[test] = risk[test] <= threshold
            original_fit = softmax(logits[fit], axis=1)
            original_test = softmax(logits[test], axis=1)
            qs = []
            for c in range(logits.shape[1]):
                scores = 1 - original_fit[labels[fit] == c, c]
                q = conformal_quantile(scores) if len(scores) else 1.
                qs.append(q)
                sets[test, c] = 1 - original_test[:, c] <= q
            receipts.append({'fold': int(fold), 'assessment_half': half, 'fit_count': len(fit),
                             'assessment_count': len(test), 'fit_group_ids': groups[fit].tolist(),
                             'assessment_group_ids': groups[test].tolist(),
                             'temperature': asdict(t), 'binary_temperatures': [asdict(v) for v in ts],
                             'acceptance_threshold': threshold, 'conformal_class_quantiles': qs})
        t = Temperature.fit(logits[ids], labels[ids])
        ts = [Temperature.fit(binary_logits[ids, j], binary_labels[ids, j]) for j in range(8)]
        pf = t.probabilities(logits[ids])
        bf = np.stack([ts[j].probabilities(binary_logits[ids, j]) for j in range(8)], axis=1)
        deployment[str(fold)] = {'temperature': asdict(t), 'binary_temperatures': [asdict(v) for v in ts],
                                 'acceptance_threshold': fit_acceptance_threshold(risk_score(pf, bf)),
                                 'n_fit': len(ids)}
    if not np.isfinite(calibrated).all() or not np.isfinite(binary).all():
        raise AssertionError('calibration coverage incomplete')
    return calibrated, binary, accepted, risk, sets, receipts, deployment


def run(output):
    selection_path = ROOT / 'research/manifests/dermpact_phase2_final/selection_2026-09-06.json'
    if digest(selection_path) != Path(str(selection_path) + '.sha256').read_text().strip():
        raise AssertionError('model selection integrity failure')
    selection = json.loads(selection_path.read_text())
    artifacts = selection['artifacts']
    paths = [ROOT / artifacts[s]['oof']['path'] for s in ['stage1', 'stage2']]
    for s, path in zip(['stage1', 'stage2'], paths):
        if digest(path) != artifacts[s]['oof']['sha256']:
            raise AssertionError('selected OOF hash mismatch')
    meta = ROOT / 'friend-cuda/research/manifests'
    inventory_path = meta / 'dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json'
    inventory = json.loads(inventory_path.read_text())
    rows = inventory['records']
    if len(rows) != 616 or any(r['phase1_role'] != 'development' for r in rows):
        raise AssertionError('development-only input required')
    class_names = ['melanoma','nevus','basal_cell_carcinoma','squamous_cell_carcinoma',
                   'actinic_keratosis_or_intraepithelial_carcinoma','seborrheic_keratosis_or_benign_keratosis',
                   'dermatofibroma','vascular_lesion','other']
    def label(raw):
        raw = raw.lower()
        if raw.startswith('melanoma'): return 0
        if 'nevus' in raw: return 1
        return {'basal cell carcinoma':2,'seborrheic keratosis':5,'dermatofibroma':6,
                'vascular lesion':7,'lentigo':8,'melanosis':8,'miscellaneous':8}[raw]
    y = np.array([label(r['diagnosis']['raw']) for r in rows])
    folds = np.array([r['development_fold'] for r in rows])
    groups = np.array([r['group_id'] for r in rows])
    a, b = [np.load(p, allow_pickle=False) for p in paths]
    z = b['derm_diagnosis_logits']
    bz = np.column_stack([b['derm_melanoma_logits'], b['derm_criteria_logits']])
    by = np.column_stack([a['mel_targets'], a['crit_targets']]).astype(int)
    protocol = {'schema_version':'dermpact-phase3-calibration-v1', 'selection_sha256':digest(selection_path),
                'source_sha256':digest(Path(__file__)), 'calibration_source_sha256':digest(ROOT/'dermpact/reliability/calibration.py'),
                'oof_bindings':{s:artifacts[s]['oof'] for s in ['stage1','stage2']},
                'inventory_sha256':digest(inventory_path),
                'method':'temperature NLL; two hash halves within each original held-out fold; no perception fitting',
                'risk':'equal mean of top-two diagnosis margin risk and maximum eight-output binary entropy',
                'target_coverage':.90,'conformal':'class conditional 90% using original fixed logits; sparse classes yield full inclusion',
                'scope':{'development_rows':616,'protected_or_final_rows_loaded':0},
                'inference_policy':'Calibrate each fixed fold model using its own holdout; average probabilities for future ensemble inference. OOF assessment evaluates held-out members, NOT the full ensemble.',
                'limitations':['Post-selection development diagnostic; final calibration/generalization not established.',
                               'Class conditional conformal validity requires exchangeability; no external coverage guarantee.']}
    protocol_path = output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise AssertionError('protocol changed; use a new version')
    write_json(protocol_path, protocol)
    p, bp, accept, risk, sets, receipts, deployment = crossfit(z,bz,y,by,folds,groups)
    original = softmax(z, axis=1)
    original_binary = expit(bz)
    def aucs(scores, mask):
        return [float(roc_auc_score(by[mask,j],scores[mask,j])) if len(np.unique(by[mask,j]))==2 else None for j in range(8)]
    errors = (p.argmax(1)!=y).astype(float)
    mean_binary_errors = ((bp>=.5)!=by).mean(1)
    all_rows = np.ones(len(y),dtype=bool)
    result = {'status':'DEVELOPMENT_CALIBRATION_AND_DEFERRAL_EVALUATED',
              'scope':protocol['scope'], 'original':calibration_metrics(y,original),
              'calibrated_crossfit':calibration_metrics(y,p),
              'binary':{str(j):{'original':calibration_metrics(by[:,j],original_binary[:,j]),
                                'calibrated_crossfit':calibration_metrics(by[:,j],bp[:,j])} for j in range(8)},
              'auroc_original':aucs(original_binary,all_rows),'auroc_calibrated':aucs(bp,all_rows),
              'deferral':{'n_accepted':int(accept.sum()),'coverage':float(accept.mean()),
                          'diagnosis_error_all':float(errors.mean()),
                          'diagnosis_error_accepted':float(errors[accept].mean()) if accept.any() else None,
                          'binary_error_all':float(mean_binary_errors.mean()),
                          'binary_error_accepted':float(mean_binary_errors[accept].mean()) if accept.any() else None,
                          'error_capture_fraction':float(errors[~accept].sum()/errors.sum()) if errors.sum() else None,
                          'auroc_accepted':aucs(bp,accept)},
              'risk_coverage_diagnosis':risk_coverage(errors,risk),
              'conformal_ablation':{'coverage':float(sets[np.arange(len(y)),y].mean()),
                                     'mean_set_size':float(sets.sum(1).mean()),
                                     'per_class':{class_names[c]:{'n':int((y==c).sum()),
                                        'coverage':float(sets[y==c,c].mean()) if np.any(y==c) else None} for c in range(9)}}}
    write_json(output/'evaluation.json', result)
    write_json(output/'calibrators.json', {'folds':deployment,'assessment_fits':receipts,
                                          'protocol_sha256':digest(protocol_path)})
    np.savez_compressed(output/'assessment_predictions.npz', ids=groups, folds=folds, labels=y,
                        diagnosis_logits=z,binary_logits=bz, binary_labels=by,
                        diagnosis_probabilities=p, binary_probabilities=bp, accepted=accept, risk=risk,
                        conformal_sets=sets)
    Path(str(output/'assessment_predictions.npz')+'.sha256').write_text(digest(output/'assessment_predictions.npz')+'\n')
    print(json.dumps({k:result[k] for k in ['status','original','calibrated_crossfit','deferral','conformal_ablation']},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    run(parser.parse_args().output_dir)
