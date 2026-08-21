"""Encode clinical metadata as features for concatenation with DermLIP embeddings.

Dermatologists always consider patient demographics and lesion characteristics
before examining through the dermatoscope.  This module encodes the available
metadata (sex, location, elevation for Derm7pt; age, sex, localization for
HAM10000) as a fixed-width numeric vector that can be appended to the visual
embedding before the classifier.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


# -- Vocabularies (sorted for deterministic one-hot order) --------------------

DERM7PT_SEX = ("female", "male")
DERM7PT_LOCATION = (
    "abdomen", "acral", "back", "buttocks", "chest",
    "genital areas", "head neck", "lower limbs", "upper limbs",
)
DERM7PT_ELEVATION = ("flat", "nodular", "palpable")

HAM_SEX = ("female", "male", "unknown")
HAM_LOCALIZATION = (
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp",
    "trunk", "unknown", "upper extremity",
)

BCN_SEX = ("female", "male")
BCN_SITE = (
    "anterior torso", "head/neck", "lower extremity",
    "oral/genital", "palms/soles", "upper extremity",
)


def _one_hot(value: str, vocab: Sequence[str]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        vec[vocab.index(value)] = 1.0
    return vec


def encode_derm7pt(
    sex: str,
    location: str,
    elevation: str,
) -> np.ndarray:
    """Encode Derm7pt clinical metadata → fixed-width vector (14-d)."""
    parts = [
        _one_hot(sex, DERM7PT_SEX),           # 2
        _one_hot(location, DERM7PT_LOCATION),  # 9
        _one_hot(elevation, DERM7PT_ELEVATION), # 3
    ]
    return np.concatenate(parts)  # 14-d


def encode_ham(
    age: Optional[float],
    sex: str,
    localization: str,
) -> np.ndarray:
    """Encode HAM10000 clinical metadata → fixed-width vector (19-d)."""
    age_norm = np.array([age / 100.0 if age is not None and age >= 0 else 0.5],
                        dtype=np.float32)
    parts = [
        age_norm,                                    # 1
        _one_hot(sex, HAM_SEX),                      # 3
        _one_hot(localization, HAM_LOCALIZATION),     # 15
    ]
    return np.concatenate(parts)  # 19-d


def encode_bcn(
    age: Optional[float],
    sex: str,
    site: str,
) -> np.ndarray:
    """Encode BCN20000 clinical metadata → fixed-width vector (9-d)."""
    age_norm = np.array([age / 100.0 if age is not None and age >= 0 else 0.5],
                        dtype=np.float32)
    parts = [
        age_norm,                        # 1
        _one_hot(sex, BCN_SEX),          # 2
        _one_hot(site, BCN_SITE),        # 6
    ]
    return np.concatenate(parts)  # 9-d


def encode_batch_derm7pt(cases: Sequence[Any]) -> np.ndarray:
    """Encode a batch of Derm7ptCase objects → (N, 14) array."""
    return np.stack([
        encode_derm7pt(c.sex, c.location, c.elevation)
        for c in cases
    ])


def encode_batch_ham(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Encode a batch of HAM manifest records → (N, 19) array."""
    return np.stack([
        encode_ham(
            r.get("metadata", {}).get("age"),
            r.get("metadata", {}).get("sex", "unknown"),
            r.get("metadata", {}).get("localization", "unknown"),
        )
        for r in records
    ])


def augment_features(
    visual_features: np.ndarray,
    metadata_features: np.ndarray,
) -> np.ndarray:
    """Concatenate visual (512-d DermLIP) and metadata features → (512+M)-d."""
    if visual_features.shape[0] != metadata_features.shape[0]:
        raise ValueError(
            f"Batch size mismatch: {visual_features.shape[0]} vs {metadata_features.shape[0]}"
        )
    return np.hstack([visual_features, metadata_features])
