"""Domain gate: is this input inside the checklist's validated domain at all?

The N3 shift analysis (research/evidence/derm7pt_v1/abstention_under_shift.json) closed the
door on the obvious mechanism: the criterion heads' posterior margins get on average *wider*
-- more confident -- on out-of-domain BCN inputs, so an in-domain-fit margin cutoff can never
announce distribution shift. A margin policy is a selective-prediction tool WITHIN a centre,
not an out-of-domain guard. What can gate is the feature geometry itself: how similar is the
input's frozen DermLIP embedding to the frozen TRAIN bank the whole checklist path was fit on?

That is this module. ``DomainGate`` scores a query as the **mean top-k cosine** to the
train-only 512-d DermLIP image bank and flags it when the score falls below a threshold fit on
Derm7pt VALID alone:

* **Image-only 512-d.** The promoted heads read a 526-d dermlip+metadata concatenation, but the
  14-d metadata one-hot block would poison a cosine: two images with identical metadata gain
  spurious similarity and unknown-metadata externals lose it -- the same verified reason the
  kNN ensemble (P2) and the case-based XAI bank retrieve over the image embedding only.
* **Train-only, hash-versioned bank.** The bank is a :class:`FrozenKNNMemory` over the frozen
  TRAIN split features -- one more case or one different vector is a different ``bank_hash``,
  so the gate's identity is quotable in every artifact it stamps.
* **Threshold fit on VALID only.** ``DOMAIN_GATE_VALID_PERCENTILE`` (p1) of the valid-split
  domain scores: ~99% of genuinely in-domain images pass by construction. BCN never touches
  the threshold -- it is the held-out shift population the gate is *validated against*
  (scripts/dev/validate_domain_gate.py), not fit on.
* **k registered from the validation grid.** ``DOMAIN_GATE_K`` is selected from {1, 3, 5} by
  the Derm7pt-valid-vs-BCN separation AUROC in domain_gate_validation.json. That selection
  reads BCN (disclosed there, k-sensitivity reported for all three); the threshold does not.

What a below-threshold score *means* is deliberately modest: the input is far from everything
the checklist path was validated on, so the dual-view contract's cross-check guarantee does not
apply -- the pipeline downgrades to the single-view report with an explicit disclosure
(medguard_v2/pipeline/predict_one.py). Whether bank similarity separates *centre shift*
(BCN is dermoscopy too) or only gross non-dermoscopic inputs is an empirical question the
validation artifact answers plainly, whichever way it lands.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from medguard_v2.data.derm7pt import load_cases
from medguard_v2.perception.encode import feature_path
from medguard_v2.perception.knn_memory import FrozenKNNMemory

__all__ = [
    "DOMAIN_GATE_K",
    "DOMAIN_GATE_VALID_PERCENTILE",
    "DomainGate",
    "build_train_bank",
    "domain_scores",
    "fit_valid_threshold",
    "load_domain_gate",
]

#: Registered neighbourhood size for the domain score (mean top-k cosine). Selected from
#: {1, 3, 5} by the Derm7pt-valid-vs-BCN separation AUROC recorded in
#: research/evidence/derm7pt_v1/domain_gate_validation.json (image-level AUROC 0.9657 /
#: 0.9602 / 0.9556 for k=1/3/5; k-sensitivity for all three is reported there, the choice is
#: not load-bearing). k=1 is the nearest-neighbour cosine. Not tunable here.
DOMAIN_GATE_K = 1

#: Percentile of the Derm7pt VALID domain-score distribution used as the pass threshold.
#: p1 => ~99% of genuinely in-domain images pass by construction. Fit on valid ONLY --
#: BCN is the held-out validation population and never touches this number.
DOMAIN_GATE_VALID_PERCENTILE = 1.0

_BANK_NAME = "derm7pt-train@dermlip512-domain"

_GATE_STATE: Dict[str, object] = {}


def _load_dermlip_features() -> Dict[str, np.ndarray]:
    """The frozen plain-view DermLIP cache, keyed by ``str(case_num)`` (all 1011 cases)."""
    data = np.load(feature_path("dermlip"))
    return {str(c): v for c, v in zip(data["case_num"], data["features"])}


def build_train_bank(
    features: Optional[Dict[str, np.ndarray]] = None,
    cases=None,
) -> FrozenKNNMemory:
    """The frozen TRAIN-only 512-d image bank the domain score is measured against.

    No labels are stored: the gate reads geometry, not votes. The bank hash still commits to
    every feature byte and case id, so the gate's identity is auditable.
    """
    if cases is None:
        cases = load_cases()
    if features is None:
        features = _load_dermlip_features()
    train = [c for c in cases if c.split == "train"]
    ids = [str(c.case_num) for c in train]
    matrix = np.stack([features[cid] for cid in ids])
    return FrozenKNNMemory(matrix, ids, {}, name=_BANK_NAME)


def domain_scores(bank: FrozenKNNMemory, queries: np.ndarray, k: int) -> np.ndarray:
    """Mean top-``k`` cosine similarity of each query row to the bank. Shape ``(M,)``.

    Exact cosine via the bank's own retrieval -- no ANN, no randomness. A wrong-dimension
    query (e.g. a 526-d promoted vector fed by mistake) raises ``ValueError`` in the bank.
    """
    q = np.ascontiguousarray(queries, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    _idx, sims = bank.retrieve(q, k)
    return sims.mean(axis=1)


def fit_valid_threshold(
    bank: FrozenKNNMemory,
    valid_features: np.ndarray,
    *,
    k: int,
    percentile: float = DOMAIN_GATE_VALID_PERCENTILE,
) -> Dict[str, object]:
    """Threshold = the ``percentile`` of the VALID domain scores. Returns value + provenance.

    The provenance dict is stored verbatim in the gate summary and the validation artifact so
    "fit on valid only, BCN never read" is a recorded property, not a code comment.
    """
    scores = domain_scores(bank, valid_features, k)
    threshold = float(np.percentile(scores, percentile))
    return {
        "threshold": threshold,
        "fit_population": "derm7pt-valid",
        "n_fit": int(scores.shape[0]),
        "percentile": float(percentile),
        "percentile_method": "linear (numpy default)",
        "bcn_used_for_threshold": False,
        "valid_score_min": float(scores.min()),
        "valid_score_mean": float(scores.mean()),
        "valid_score_max": float(scores.max()),
    }


class DomainGate:
    """Frozen domain gate: mean top-k cosine to the TRAIN bank vs a valid-fit threshold."""

    def __init__(
        self,
        bank: FrozenKNNMemory,
        *,
        k: int,
        threshold: float,
        threshold_provenance: Optional[Dict[str, object]] = None,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self._bank = bank
        self._k = int(k)
        self._threshold = float(threshold)
        self._threshold_provenance = dict(threshold_provenance or {})

    # -- identity ---------------------------------------------------------------

    @property
    def bank_hash(self) -> str:
        return self._bank.bank_hash

    @property
    def k(self) -> int:
        return self._k

    @property
    def threshold(self) -> float:
        return self._threshold

    def summary(self) -> Dict[str, object]:
        """Everything an artifact needs to reproduce this gate: bank hash, k, threshold."""
        return {
            "gate": "domain-similarity-v1",
            "score_definition": (
                f"mean top-{self._k} exact cosine to the frozen TRAIN dermlip image bank "
                "(512-d, image-only; metadata block excluded so cosine is not poisoned)"
            ),
            "k": self._k,
            "threshold": round(self._threshold, 6),
            "threshold_provenance": dict(self._threshold_provenance),
            "bank": self._bank.summary(),
        }

    # -- scoring ----------------------------------------------------------------

    def score(self, queries: np.ndarray) -> np.ndarray:
        return domain_scores(self._bank, queries, self._k)

    def evaluate(self, query: np.ndarray) -> Dict[str, object]:
        """One query vector -> the domain block predict_one stamps into its output."""
        score = float(self.score(np.asarray(query, dtype=np.float64).reshape(1, -1))[0])
        return {
            "score": round(score, 6),
            "threshold": round(self._threshold, 6),
            "below_threshold": bool(score < self._threshold),
            "k": self._k,
            "bank_hash": self._bank.bank_hash,
            "bank_name": self._bank.name,
        }


def load_domain_gate() -> DomainGate:
    """The canonical gate, memoised per process.

    Deterministic from the frozen caches alone: TRAIN bank from the dermlip feature cache,
    threshold refit from the VALID scores at the registered k/percentile on every load (same
    no-persisted-artifact rationale as predict_one's head fitting -- refitting is milliseconds
    and cannot go stale).
    """
    gate = _GATE_STATE.get("gate")
    if gate is not None:
        return gate  # type: ignore[return-value]
    cases = load_cases()
    features = _load_dermlip_features()
    bank = build_train_bank(features, cases)
    valid = [c for c in cases if c.split == "valid"]
    valid_matrix = np.stack([features[str(c.case_num)] for c in valid])
    fit = fit_valid_threshold(bank, valid_matrix, k=DOMAIN_GATE_K)
    gate = DomainGate(
        bank,
        k=DOMAIN_GATE_K,
        threshold=fit["threshold"],
        threshold_provenance={key: v for key, v in fit.items() if key != "threshold"},
    )
    _GATE_STATE["gate"] = gate
    return gate
