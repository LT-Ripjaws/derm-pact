"""Single-image end-to-end entry point: one dermoscopy image -> the decision report.

Everything the evaluated system does, for ONE case, from a raw image path:

    frozen DermLIP encode (image + horizontal-flip mean for the criteria view)
    -> DOMAIN GATE (mean top-k cosine to the frozen TRAIN bank; below the valid-fit
       threshold means the input is outside the checklist's validated domain and the
       pipeline downgrades to the single-view report with an explicit disclosure)
    -> promoted heads (diagnosis: dermlip+metadata C=100 unbalanced; criteria:
       dermlip_tta+metadata lr_balanced) with the SK/MISC operating point
    -> DecisionContract (margins, conflict, hash-pinned guideline binding,
       self-verification) -- or SingleViewContract when the domain gate fires
    -> rendered clinician report (+ optionally a verified case brief / agent session).

This is glue, not new behaviour: every stage is the exact function the cohort
pipeline runs (fit_readers_promoted, apply_operating_point, _head_values_and_margins,
build_contract, render_markdown), so a prediction here is the same system the paper's
numbers describe. The identity smoke test (scripts/dev/smoke_predict_one.py) proves it
by feeding a dataset image through THIS path -- raw JPG, no feature caches -- and
requiring the resulting contract to match the cached-pipeline contract for that case.

THE DOMAIN GATE (R-A, closing the N3 loop). N3 measured that the criterion heads'
margins get MORE confident under the Derm7pt->BCN centre shift, so margin-based
abstention cannot gate out-of-domain inputs. The gate that can is feature-bank
similarity (medguard_v2/perception/domain_check.py, validated in
research/evidence/derm7pt_v1/domain_gate_validation.json: valid-vs-BCN image AUROC
0.9657 at k=1; threshold = p1 of Derm7pt VALID scores, BCN never fit on). When the
gate fires, the honest output is NOT the dual-view contract -- the checklist path was
never validated out there -- and not silence either: it is the single-view report
whose disclosure says the input is outside the validated domain AND that the
classifier reading itself (also Derm7pt-trained) should be weighted accordingly.
``force_dual_view=True`` is the recorded escape hatch (identity smoke, power users).

Heads are fit from the frozen TRAIN split on first use (deterministic: fixed seed,
fixed C, same sklearn) and memoised per process. Persisting fitted heads to disk was
deliberately not added: refitting takes seconds, cannot drift from the training
recipe, and avoids a second artefact whose staleness would need auditing. The encoder
is memoised separately so embedding-level callers (tests, cached-feature pipelines)
never load it.

Derm7pt clinical metadata is sex / location / elevation (14-d one-hot; there is no
age field in this dataset). Unknown or missing values one-hot to the all-zero block,
the same convention the batch encoder uses.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from medguard_v2.criteria.schema import CRITERION_NAMES
from medguard_v2.data.derm7pt import load_cases
from medguard_v2.evidence.contract import DecisionContract, build_contract
from medguard_v2.evidence.guideline_binding import GuidelineCatalog
from medguard_v2.evidence.render import render_markdown
from medguard_v2.evidence.single_view import SingleViewContract, build_single_view_contract
from medguard_v2.perception.clinical_metadata import encode_derm7pt
from medguard_v2.perception.domain_check import load_domain_gate
from medguard_v2.pipeline.run_cohort import (
    PROMOTED_OPERATING_POINT,
    _head_values_and_margins,
    _promoted_feature_matrices,
    apply_operating_point,
    fit_readers_promoted,
)

_STATE: Dict[str, object] = {}
_ENCODER_STATE: Dict[str, object] = {}

#: The mandatory disclosure a domain-gated report carries. The honest subtlety it must state:
#: the promoted DIAGNOSIS head was ALSO Derm7pt-trained, so out-of-domain its reading is
#: itself suspect -- shown single-view, weighted accordingly, never dressed up as a verified
#: dual-view decision.
DOMAIN_GATE_DISCLOSURE = (
    "input is outside the checklist's validated domain (bank similarity {score:.2f} < "
    "{threshold:.2f}); classifier reading shown single-view and should be weighted accordingly"
)


@dataclass(frozen=True)
class SinglePrediction:
    #: Dual-view :class:`DecisionContract` normally; :class:`SingleViewContract` when the
    #: domain gate downgraded the case (``domain["gated"]`` says which).
    contract: Union[DecisionContract, SingleViewContract]
    report_markdown: str
    diagnosis: str
    confidence: float
    #: Per-criterion readings. EMPTY for a domain-gated case: the criteria heads are not run
    #: outside their validated domain -- an absent checklist is disclosed, not imputed.
    criteria: Dict[str, str]
    #: The domain-gate record: score, threshold, below_threshold, gated, k, bank_hash (+
    #: forced_dual_view when the escape hatch overrode a below-threshold score).
    domain: Dict[str, object] = field(default_factory=dict)


def _fitted_state() -> Dict[str, object]:
    """Fit the promoted readers once per process from the frozen TRAIN split (no encoder)."""
    if _STATE:
        return _STATE
    cases = load_cases()
    train = [c for c in cases if c.split == "train"]
    diag_feats, crit_feats = _promoted_feature_matrices(cases)
    diagnosis, heads = fit_readers_promoted(train, diag_feats, crit_feats)
    _STATE.update({
        "diagnosis": diagnosis, "heads": heads, "catalog": GuidelineCatalog.load(),
    })
    return _STATE


def _encoder_state() -> Dict[str, object]:
    """Load the frozen DermLIP encoder once per process (image-path callers only)."""
    if _ENCODER_STATE:
        return _ENCODER_STATE
    from medguard_v2.perception.encode import load_encoder

    encode, provenance = load_encoder("dermlip")
    _ENCODER_STATE.update({"encode": encode, "provenance": provenance})
    return _ENCODER_STATE


def predict_from_embeddings(
    plain: np.ndarray,
    tta: Optional[np.ndarray] = None,
    *,
    sex: str = "unknown",
    location: str = "unknown",
    elevation: str = "unknown",
    case_id: str = "external-case",
    force_dual_view: bool = False,
    entry_point: str = "predict_from_embeddings",
    encoder_provenance: Optional[Dict[str, object]] = None,
) -> SinglePrediction:
    """The post-encode core: frozen 512-d dermlip embedding(s) -> decision report.

    ``plain`` is the plain-view embedding (what the domain gate and the diagnosis head read);
    ``tta`` is the plain+flip mean the criteria heads read, required only when the dual view
    actually runs -- a domain-gated case never needs it. Exists as its own entry point so
    cached-feature callers and the test suite exercise the identical decision logic without
    the encoder.
    """
    st = _fitted_state()
    gate = load_domain_gate()
    plain = np.asarray(plain, dtype=np.float64).reshape(-1)

    evaluation = gate.evaluate(plain)
    gated = bool(evaluation["below_threshold"] and not force_dual_view)
    domain: Dict[str, object] = {
        **evaluation,
        "gated": gated,
        "forced_dual_view": bool(force_dual_view and evaluation["below_threshold"]),
    }

    meta = np.asarray(encode_derm7pt(sex, location, elevation), dtype=np.float64)
    x_diag = np.concatenate([plain, meta]).reshape(1, -1)
    posterior = st["diagnosis"].predict_proba(x_diag)
    predicted, confidences = apply_operating_point(
        posterior, st["diagnosis"].classes_, PROMOTED_OPERATING_POINT
    )

    receipt: Dict[str, object] = {
        "entry_point": entry_point,
        "encoder": encoder_provenance
        or "embedding supplied by caller (frozen dermlip 512-d assumed)",
        "operating_point": dict(PROMOTED_OPERATING_POINT),
        "metadata_provided": {"sex": sex, "location": location, "elevation": elevation},
        "domain_gate": dict(domain),
    }

    if gated:
        reason = DOMAIN_GATE_DISCLOSURE.format(
            score=evaluation["score"], threshold=evaluation["threshold"]
        )
        contract = build_single_view_contract(
            case_id=case_id,
            classifier_diagnosis=str(predicted[0]),
            classifier_confidence=float(confidences[0]),
            catalog=st["catalog"],
            reason=reason,
            receipt=receipt,
        )
        return SinglePrediction(
            contract=contract,
            report_markdown=render_markdown(contract),
            diagnosis=contract.classifier_diagnosis,
            confidence=contract.classifier_confidence,
            criteria={},
            domain=domain,
        )

    if tta is None:
        raise ValueError(
            "dual-view prediction requires the TTA (plain+flip mean) embedding; only a "
            "domain-gated case may omit it"
        )
    tta = np.asarray(tta, dtype=np.float64).reshape(-1)
    x_crit = np.concatenate([tta, meta]).reshape(1, -1)
    criteria: Dict[str, str] = {}
    margins: Dict[str, float] = {}
    for name in CRITERION_NAMES:
        values, m = _head_values_and_margins(st["heads"][name], x_crit)
        criteria[name] = values[0]
        margins[name] = float(m[0])

    contract = build_contract(
        case_id=case_id,
        classifier_diagnosis=str(predicted[0]),
        classifier_confidence=float(confidences[0]),
        criteria=criteria,
        catalog=st["catalog"],
        criterion_margins=margins,
        receipt=receipt,
    )
    return SinglePrediction(
        contract=contract,
        report_markdown=render_markdown(contract),
        diagnosis=contract.classifier_diagnosis,
        confidence=contract.classifier_confidence,
        criteria=criteria,
        domain=domain,
    )


def predict_one(
    image_path: Path | str,
    *,
    sex: str = "unknown",
    location: str = "unknown",
    elevation: str = "unknown",
    case_id: str = "external-case",
    force_dual_view: bool = False,
) -> SinglePrediction:
    """One raw dermoscopy image (+ optional sex/location/elevation) -> decision report.

    The domain gate runs on the plain embedding BEFORE the heads: a below-threshold image is
    downgraded to the single-view report (and the flip-TTA encode is skipped -- the criteria
    heads are never consulted outside their validated domain). ``force_dual_view=True``
    bypasses the downgrade, recorded in ``domain["forced_dual_view"]``.
    """
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(f"image not found: {img}")
    enc = _encoder_state()

    # Frozen encode: plain view for diagnosis; plain+flip mean (TTA) for criteria --
    # exactly the promoted feature recipe.
    plain = np.asarray(enc["encode"]([img])[0], dtype=np.float64)

    # Flip-encode only when the dual view will actually run.
    gate = load_domain_gate()
    need_dual = force_dual_view or not gate.evaluate(plain)["below_threshold"]
    tta: Optional[np.ndarray] = None
    if need_dual:
        from PIL import Image

        with Image.open(img) as im:
            flipped = im.transpose(Image.FLIP_LEFT_RIGHT)
            with tempfile.TemporaryDirectory() as td:
                fp = Path(td) / "flip.png"
                flipped.save(fp)
                flip = np.asarray(enc["encode"]([fp])[0], dtype=np.float64)
        tta = (plain + flip) / 2.0

    return predict_from_embeddings(
        plain,
        tta,
        sex=sex,
        location=location,
        elevation=elevation,
        case_id=case_id,
        force_dual_view=force_dual_view,
        entry_point="predict_one",
        encoder_provenance=enc["provenance"],
    )
