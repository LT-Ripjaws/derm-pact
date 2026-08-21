"""BCN20000: a frozen lesion-level split, and the in-domain classifier cohort.

U1 established that a classifier trained on HAM does not transfer to Derm7pt (-0.105 melanoma
AUROC). The same logic applies in reverse, so the Derm7pt-trained classifier should not be
expected to work on BCN either. This module sets up the design that follows from that:

    diagnosis classifier   -> trained IN-DOMAIN on BCN
    criterion heads        -> stay OUT-OF-DOMAIN, trained on Derm7pt (the only criterion labels)
    confirmatory cohort    -> held-out BCN lesions

which asks the question the paper actually needs: **does an out-of-domain criteria path still add
melanoma sensitivity on top of a strong in-domain classifier?**

Splitting is by ``cluster_id`` (lesion), never by image. BCN has 3,570 lesions over 12,356 images
with a minimum of two images per lesion, so an image-level split would leak the same lesion across
train and test and inflate everything. Every lesion carries exactly one label (verified: zero
lesions have conflicting image labels), so lesion-level stratification is well defined.

The split is frozen and hashed before any model is fitted. The test portion is not read by any
development code -- ``load_split`` refuses to hand it out unless explicitly asked.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "research" / "manifests" / "bcn20000_external_v1.json"
FEATURES = REPO / "research/experiments/derm_perception/v2_tournament/external_bcn_v1/features/dermlip/features.npy"
SPLIT_PATH = REPO / "research" / "manifests" / "bcn20000_lesion_split_v2.json"

SPLIT_SEED = 20260811
SPLIT_FRACTIONS = {"train": 0.6, "valid": 0.2, "test": 0.2}

#: BCN label -> the Derm7pt five-group taxonomy. ``scc`` has no Derm7pt counterpart and joins
#: MISC alongside akiec/df/vasc; this is a declared judgement call, not a silent merge.
BCN_TO_GROUP: Mapping[str, str] = {
    "melanoma": "MEL",
    "nevus": "NEV",
    "bcc": "BCC",
    "bkl": "SK",
    "akiec": "MISC",
    "scc": "MISC",
    "df": "MISC",
    "vasc": "MISC",
}


@dataclass(frozen=True)
class BcnRecord:
    case_id: str
    lesion_id: str
    label: str
    group: str
    split: str
    row: int          # index into the frozen feature matrix


def _load_records() -> List[Dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["records"]


def build_split() -> Dict:
    """Deterministic lesion-level split, stratified by lesion group."""
    records = _load_records()

    lesion_group: Dict[str, str] = {}
    for record in records:
        group = BCN_TO_GROUP[record["label"]]
        existing = lesion_group.setdefault(record["cluster_id"], group)
        if existing != group:
            raise RuntimeError(f"lesion {record['cluster_id']} has conflicting labels")

    by_group: Dict[str, List[str]] = defaultdict(list)
    for lesion, group in lesion_group.items():
        by_group[group].append(lesion)

    rng = np.random.default_rng(SPLIT_SEED)
    assignment: Dict[str, str] = {}
    for group, lesions in sorted(by_group.items()):
        ordered = sorted(lesions)               # deterministic before shuffling
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = int(round(SPLIT_FRACTIONS["train"] * n))
        n_valid = int(round(SPLIT_FRACTIONS["valid"] * n))
        for i, lesion in enumerate(ordered):
            if i < n_train:
                assignment[lesion] = "train"
            elif i < n_train + n_valid:
                assignment[lesion] = "valid"
            else:
                assignment[lesion] = "test"

    body = {
        "protocol": "bcn20000-lesion-split-v2",
        "seed": SPLIT_SEED,
        "fractions": SPLIT_FRACTIONS,
        "unit": "lesion (cluster_id)",
        "n_lesions": len(assignment),
        "n_images": len(records),
        "label_mapping": dict(BCN_TO_GROUP),
        "lesion_split": dict(sorted(assignment.items())),
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    counts: Dict[str, Counter] = {s: Counter() for s in SPLIT_FRACTIONS}
    image_counts: Dict[str, int] = {s: 0 for s in SPLIT_FRACTIONS}
    for record in records:
        split = assignment[record["cluster_id"]]
        image_counts[split] += 1
    for lesion, split in assignment.items():
        counts[split][lesion_group[lesion]] += 1

    return {
        **body,
        "body_sha256": digest,
        "lesion_counts_by_split": {s: dict(sorted(c.items())) for s, c in counts.items()},
        "image_counts_by_split": image_counts,
    }


def freeze_split(path: Path = SPLIT_PATH) -> Dict:
    split = build_split()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return split


def load_split(path: Path = SPLIT_PATH) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not frozen yet; call freeze_split() first")
    split = json.loads(path.read_text(encoding="utf-8"))
    body = {k: split[k] for k in (
        "protocol", "seed", "fractions", "unit", "n_lesions", "n_images",
        "label_mapping", "lesion_split")}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != split["body_sha256"]:
        raise RuntimeError("BCN split hash mismatch: the frozen split has been edited")
    return split


def load_records(*, splits: Sequence[str] = ("train", "valid"), allow_test: bool = False
                 ) -> Tuple[List[BcnRecord], np.ndarray]:
    """Records and features for the requested splits.

    ``test`` must be requested explicitly AND unlocked with ``allow_test``. Development code that
    asks for it by accident gets an exception rather than a contaminated result.
    """
    if "test" in splits and not allow_test:
        raise PermissionError(
            "The BCN test split is the confirmatory cohort. Pass allow_test=True only from the "
            "registered confirmatory runner."
        )

    split = load_split()
    assignment = split["lesion_split"]
    records = _load_records()
    features = np.load(FEATURES)
    if features.shape[0] != len(records):
        raise RuntimeError(
            f"feature/record mismatch: {features.shape[0]} vectors vs {len(records)} records"
        )

    selected: List[BcnRecord] = []
    rows: List[int] = []
    for i, record in enumerate(records):
        lesion = record["cluster_id"]
        where = assignment[lesion]
        if where not in splits:
            continue
        selected.append(
            BcnRecord(
                case_id=record["case_id"],
                lesion_id=lesion,
                label=record["label"],
                group=BCN_TO_GROUP[record["label"]],
                split=where,
                row=i,
            )
        )
        rows.append(i)
    return selected, features[rows]


def lesion_level(records: Sequence[BcnRecord], features: np.ndarray
                 ) -> Tuple[List[BcnRecord], np.ndarray]:
    """Collapse images to lesions by mean-pooling their features, then re-normalising.

    A lesion is the clinical unit and carries one label; scoring images independently would count
    the same lesion several times and weight multi-image lesions more heavily.
    """
    order: List[str] = []
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, record in enumerate(records):
        if record.lesion_id not in buckets:
            order.append(record.lesion_id)
        buckets[record.lesion_id].append(i)

    pooled = []
    representatives = []
    for lesion in order:
        idx = buckets[lesion]
        vector = features[idx].mean(axis=0)
        norm = np.linalg.norm(vector)
        pooled.append(vector / norm if norm else vector)
        representatives.append(records[idx[0]])
    return representatives, np.stack(pooled)
