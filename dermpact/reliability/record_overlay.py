"""Calibration overlay preserves Phase 5 raw evidence and verifies transformations."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from dermpact.reliability.calibration import Temperature, risk_score
from dermpact.reliability.sentinels import recovery


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class VerifiedCalibrationOverlay:
    case_id: str
    raw_record_sha256: str
    calibration_binding_sha256: str
    diagnosis_probabilities: tuple
    binary_probabilities: tuple
    risk: float
    deferred: bool

    def to_dict(self):
        return {'case_id':self.case_id,'raw_record_sha256':self.raw_record_sha256,
                'calibration_binding_sha256':self.calibration_binding_sha256,
                'diagnosis_probabilities':list(self.diagnosis_probabilities),
                'binary_probabilities':list(self.binary_probabilities),'risk':self.risk,'deferred':self.deferred}


def build_overlay(record, trusted_calibration):
    """Caller supplies verified parameters; never read parameters from an LLM record."""
    p=Temperature(**trusted_calibration['temperature']).probabilities(np.array([record.perception.diagnosis_logits]))
    z=np.array([[record.perception.melanoma_logit,*record.perception.criterion_logits]])
    temperatures=trusted_calibration['binary_temperatures']
    if len(temperatures)!=8:raise ValueError('eight output calibrators required')
    b=np.stack([Temperature(**temperatures[j]).probabilities(z[:,j]) for j in range(8)],axis=1)
    risk=float(risk_score(p,b)[0]);threshold=float(trusted_calibration['acceptance_threshold'])
    if not np.isfinite(threshold) or threshold<0 or threshold>1:raise ValueError('invalid deferral threshold')
    return VerifiedCalibrationOverlay(record.identity.case_id,record.record_sha256,fingerprint(trusted_calibration),
                                      tuple(p[0]),tuple(b[0]),risk,risk>threshold)


def verify_overlay(record, overlay, trusted_calibration):
    expected=build_overlay(record,trusted_calibration)
    if overlay.case_id!=expected.case_id or overlay.raw_record_sha256!=expected.raw_record_sha256:
        return False
    if overlay.calibration_binding_sha256!=expected.calibration_binding_sha256 or overlay.deferred!=expected.deferred:
        return False
    if np.shape(overlay.diagnosis_probabilities)!=np.shape(expected.diagnosis_probabilities) or np.shape(overlay.binary_probabilities)!=(8,):
        return False
    return (np.allclose(overlay.diagnosis_probabilities,expected.diagnosis_probabilities,atol=1e-12,rtol=0)
            and np.allclose(overlay.binary_probabilities,expected.binary_probabilities,atol=1e-12,rtol=0)
            and abs(overlay.risk-expected.risk)<1e-12)


def presentation(record,overlay):
    return {'raw_model_evidence':record.to_dict(), 'calibrated_assessment':overlay.to_dict(),
            'decision':recovery(iqa_failed=record.reliability.iqa_state=='fail',
                                ood_flagged=record.reliability.ood_state=='out_of_distribution',
                                deferred=overlay.deferred,unstable=record.reliability.view_stability_state=='unstable'),
            'display_contract':'Raw and calibrated probabilities are separate; calibrated overlay never replaces model logits or clinician amendments.'}
