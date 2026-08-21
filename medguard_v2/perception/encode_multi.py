"""Additional frozen feature sets: the second view, and a second encoder.

:mod:`medguard_v2.perception.encode` produces exactly one feature set --
DermLIP over the **dermoscopic** image.  Every v2 model (the diagnosis probe and
all seven criterion heads) is a linear read-out of that single representation,
so they are re-slicings of one vector and can only be so complementary.  Two
things are sitting unused and cost nothing:

* Derm7pt ships a **paired clinical photograph** per case (``clinic_path``).
* **BiomedCLIP** is already in the local Hugging Face cache.

This module adds the three missing corners of the 2x2, in the same cache format
and under the same hashing discipline as ``encode.py``::

    features/dermlip/           <- encode.py owns this one (DermLIP  x derm)
    features/dermlip_clinic/    <- here          (DermLIP    x clinic)
    features/biomedclip_derm/   <- here          (BiomedCLIP x derm)
    features/biomedclip_clinic/ <- here          (BiomedCLIP x clinic)

``encode.py`` is deliberately not modified: it is the input to the frozen
perception baseline and to gate G5, and its cache is already written.

Differences from ``encode.py``, all deliberate
----------------------------------------------
* **The manifest body is byte-reproducible.**  ``encode.py`` writes
  ``elapsed_seconds``, which changes every run.  Timing is printed to stdout
  here instead, so two runs over the same images with the same checkpoint
  produce identical ``feature_manifest.json`` *and* identical ``features.npz``
  (verified: ``numpy.savez_compressed`` stamps every zip member with the
  ``ZipInfo`` default date of 1980-01-01, not the wall clock).
* **The cache-currency check includes the checkpoint fingerprint.**
  ``encode.py``'s docstring promises this but its ``_cache_is_current`` only
  compares protocol / view / image digest / case count.  Swapping the
  checkpoint under an existing cache there is silent; here it re-encodes.
* **The duplicate-view caveat is carried in the artifact.**  Nine cases have
  ``clinic_path == derm_path`` -- the dataset's own ``notes`` column records
  that the clinical image was missing and was replaced by the dermoscopic one
  (``notes/DERM7PT_DATA_CAVEATS.md``; all nine are MEL or MISC).  Their
  "clinical" vector is therefore a *copy of their dermoscopic vector*, not a
  second view.  Every manifest records the count and the case_nums, and the
  clinical manifests additionally set ``duplicated_from_view: "derm"``, so a
  consumer stacking the two views cannot count one image as two pieces of
  evidence without first having read that it is one.

The image tower only
--------------------
``open_clip`` builds the whole CLIP model; only ``encode_image`` is ever called,
so the text tower is constructed and never run.  There is no supported way to
ask ``create_model_and_transforms`` for the vision half alone, so this is
recorded (``provenance.tower == "image"``) rather than hacked around.

Licensing: Derm7pt is CC BY-NC-ND.  As in ``encode.py``, no image bytes are
copied anywhere -- only feature vectors, content digests and metadata.

CLI::

    python -m medguard_v2.perception.encode_multi              # encode (cached)
    python -m medguard_v2.perception.encode_multi --report     # + separability check
    python -m medguard_v2.perception.encode_multi --force      # ignore the cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from medguard_v2.data.derm7pt import (
    CRITERION_VOCABULARY,
    Derm7ptCase,
    load_cases,
)
from medguard_v2.perception import encode as _encode
from medguard_v2.perception.encode import (
    ENCODERS,
    FEATURE_ROOT,
    FeatureBundle,
    PerceptionEncodeError,
    checkpoint_fingerprint,
    encoder_snapshot,
    feature_path,
    manifest_path,
)

__all__ = [
    "CACHE_PROTOCOL",
    "FEATURE_SETS",
    "IMAGE_VIEWS",
    "OWNED_ELSEWHERE",
    "PRODUCED_HERE",
    "REPORT_TARGETS",
    "FeatureSetSpec",
    "duplicate_view_record",
    "encode_all",
    "encode_feature_set",
    "feature_path",
    "image_path",
    "image_set_digest",
    "load_feature_set",
    "manifest_path",
    "separability_report",
]


REPO_ROOT = Path(__file__).resolve().parents[2]

#: Distinct from ``encode.CACHE_PROTOCOL``: the manifest carries extra fields
#: (``feature_set_key``, the duplicate-view record) and drops ``elapsed_seconds``.
CACHE_PROTOCOL = "derm7pt-frozen-features-multiview-v1"

#: ``view name -> Derm7ptCase attribute holding the resolved on-disk path``.
IMAGE_VIEWS: Mapping[str, str] = {"derm": "derm_path", "clinic": "clinic_path"}


@dataclass(frozen=True)
class FeatureSetSpec:
    """One (encoder, view) corner of the 2x2, and where its cache lives."""

    key: str
    encoder_key: str
    image_view: str
    #: True when this module writes the cache.  ``features/dermlip`` is written
    #: by ``encode.py`` and is only ever *read* here.
    produced_here: bool
    note: str

    @property
    def directory(self) -> Path:
        # Resolved through ``encode.feature_path`` rather than the imported
        # ``FEATURE_ROOT`` constant, so it follows a temporarily rebound root.
        return feature_path(self.key).parent


FEATURE_SETS: Mapping[str, FeatureSetSpec] = {
    "dermlip": FeatureSetSpec(
        key="dermlip",
        encoder_key="dermlip",
        image_view="derm",
        produced_here=False,
        note="the existing v2 feature set; owned by medguard_v2.perception.encode",
    ),
    "dermlip_clinic": FeatureSetSpec(
        key="dermlip_clinic",
        encoder_key="dermlip",
        image_view="clinic",
        produced_here=True,
        note="same encoder, the unused paired clinical photograph",
    ),
    "biomedclip_derm": FeatureSetSpec(
        key="biomedclip_derm",
        encoder_key="biomedclip",
        image_view="derm",
        produced_here=True,
        note="second encoder, same view: a different representation of the same pixels",
    ),
    "biomedclip_clinic": FeatureSetSpec(
        key="biomedclip_clinic",
        encoder_key="biomedclip",
        image_view="clinic",
        produced_here=True,
        note="second encoder, second view: the corner furthest from the existing set",
    ),
}

#: Written by this module.
PRODUCED_HERE = tuple(key for key, spec in FEATURE_SETS.items() if spec.produced_here)
#: Read but never written by this module.
OWNED_ELSEWHERE = tuple(key for key, spec in FEATURE_SETS.items() if not spec.produced_here)


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------

def image_path(case: Derm7ptCase, view: str) -> Path:
    try:
        attribute = IMAGE_VIEWS[view]
    except KeyError:
        raise PerceptionEncodeError(
            f"unknown image view {view!r}; choose from {sorted(IMAGE_VIEWS)}"
        ) from None
    return getattr(case, attribute)


def image_set_digest(cases: Sequence[Derm7ptCase], view: str) -> str:
    """One digest over the *content* of every image in ``view``.

    The line format is byte-identical to :func:`encode.image_set_digest`, so the
    ``derm`` digest computed here equals the one already recorded in
    ``features/dermlip/feature_manifest.json``.  :func:`encode_all` asserts that
    equality: it is a free cross-check that this module is reading the same
    pixels as the module it is extending.

    Only the digest is stored, never image bytes (CC BY-NC-ND).
    """
    lines: list[str] = []
    for case in sorted(cases, key=lambda item: item.case_num):
        path = image_path(case, view)
        # ``_sha256_file`` is imported rather than re-implemented on purpose:
        # this digest is only meaningful if it is computed exactly the way
        # ``encode.py`` computes it, and ``encode.py`` must not be edited to
        # export it.
        lines.append(f"{case.case_num}:{path.name}:{_encode._sha256_file(path)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def duplicate_view_record(cases: Sequence[Derm7ptCase], view: str) -> dict:
    """The nine ``clinic == derm`` cases, recorded so they cannot be missed.

    Carried in *every* manifest (uniform schema), but the interpretation is
    view-dependent and stated inline: in a ``clinic`` feature set these rows are
    a copy of the dermoscopic vector and are **not** a second view; in a
    ``derm`` feature set they are ordinary rows and the field is informational.
    """
    affected = sorted(
        int(case.case_num) for case in cases if not case.has_distinct_views
    )
    duplicated = view == "clinic"
    return {
        "n_cases_without_distinct_views": len(affected),
        "case_nums_without_distinct_views": affected,
        "duplicated_from_view": "derm" if duplicated else None,
        "note": (
            "these case_nums have clinic_path == derm_path (the dataset's notes column "
            "records the clinical image as missing and replaced with the dermoscopic "
            "one), so their vectors in THIS feature set are identical to their "
            "dermoscopic vectors and are not independent evidence"
            if duplicated
            else "these case_nums have clinic_path == derm_path; in a derm feature set "
            "that is immaterial -- the vector here is the genuine dermoscopic one. "
            "The flag is carried so the count is the same in every manifest."
        ),
        "source": "notes/DERM7PT_DATA_CAVEATS.md",
    }


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def load_feature_set(key: str) -> FeatureBundle | None:
    """Read one cached feature set, or ``None`` if it is absent.

    Delegates the npz/manifest digest check to
    :func:`encode.load_cached_features` (so a corrupt cache fails the same way
    in both modules) and then adds the checks specific to this schema.
    """
    spec = _spec(key)
    bundle = _encode.load_cached_features(key)
    if bundle is None:
        return None
    manifest = bundle.manifest
    if spec.produced_here:
        if manifest.get("protocol") != CACHE_PROTOCOL:
            raise PerceptionEncodeError(
                f"{manifest_path(key)} has protocol {manifest.get('protocol')!r}, "
                f"expected {CACHE_PROTOCOL!r}"
            )
        if manifest.get("feature_set_key") != key:
            raise PerceptionEncodeError(
                f"{manifest_path(key)} claims feature_set_key "
                f"{manifest.get('feature_set_key')!r}, but lives in {key!r}"
            )
    if manifest.get("image_view") != spec.image_view:
        raise PerceptionEncodeError(
            f"{manifest_path(key)} records image_view {manifest.get('image_view')!r}, "
            f"but {key!r} is defined over {spec.image_view!r}"
        )
    return bundle


def _spec(key: str) -> FeatureSetSpec:
    try:
        return FEATURE_SETS[key]
    except KeyError:
        raise PerceptionEncodeError(
            f"unknown feature set {key!r}; choose from {sorted(FEATURE_SETS)}"
        ) from None


def _cache_is_current(
    bundle: FeatureBundle,
    cases: Sequence[Derm7ptCase],
    spec: FeatureSetSpec,
    digest: str,
    checkpoint_sha256: str | None,
) -> bool:
    manifest = bundle.manifest
    expected = {int(case.case_num) for case in cases}
    provenance = manifest.get("provenance") or {}
    cached_checkpoint = (provenance.get("checkpoint") or {}).get("combined_sha256")
    if checkpoint_sha256 is not None and cached_checkpoint != checkpoint_sha256:
        return False
    return (
        manifest.get("protocol") == CACHE_PROTOCOL
        and manifest.get("feature_set_key") == spec.key
        and manifest.get("encoder_key") == spec.encoder_key
        and manifest.get("image_view") == spec.image_view
        and manifest.get("image_set_sha256") == digest
        and manifest.get("n_cases") == len(expected)
        and expected.issubset({int(value) for value in bundle.case_nums})
    )


def _write_bundle(key: str, case_nums: np.ndarray, features: np.ndarray, manifest: dict) -> None:
    """Atomically write ``features.npz`` and ``feature_manifest.json``.

    Both writes go through a temp file and ``os.replace`` so an interrupted run
    can never leave a manifest describing a half-written array.
    """
    target = feature_path(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".npz.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, case_num=case_nums, features=features)
    os.replace(temp, target)
    meta = manifest_path(key)
    temp_meta = meta.with_suffix(".json.tmp")
    temp_meta.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_meta, meta)


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------

def _check_images_readable(cases: Sequence[Derm7ptCase], view: str) -> None:
    """Fail before spending a minute of forward passes, naming every bad case."""
    missing = [
        f"case_num {case.case_num} -> {image_path(case, view)}"
        for case in cases
        if not image_path(case, view).is_file()
    ]
    if missing:
        raise PerceptionEncodeError(
            f"{len(missing)} {view} image(s) do not exist on disk: " + "; ".join(missing[:10])
        )


def _encode_chunk(encode_fn, chunk: Sequence[Derm7ptCase], view: str) -> np.ndarray:
    """Encode one batch, re-raising a decode failure with the offending case_num."""
    paths = [image_path(case, view) for case in chunk]
    try:
        return encode_fn(paths)
    except Exception as exc:  # noqa: BLE001 - PIL/torch raise a family of errors
        # Only on the failure path: narrow the batch to the exact case, because
        # "one of these 32 images is unreadable" is not an actionable message.
        for case in chunk:
            try:
                encode_fn([image_path(case, view)])
            except Exception as inner:  # noqa: BLE001
                raise PerceptionEncodeError(
                    f"could not encode the {view} image of case_num {case.case_num} "
                    f"({image_path(case, view)}): {type(inner).__name__}: {inner}"
                ) from inner
        raise PerceptionEncodeError(
            f"batch of {len(chunk)} {view} images failed to encode but every image "
            f"encodes on its own: {type(exc).__name__}: {exc}"
        ) from exc


def encode_feature_set(
    key: str,
    cases: Sequence[Derm7ptCase] | None = None,
    *,
    allow_download: bool = False,
    batch_size: int | None = None,
    force: bool = False,
    verbose: bool = False,
    feature_root: Path | None = None,
) -> FeatureBundle:
    """Return features for one (encoder, view) pair, encoding only if needed.

    ``feature_root`` overrides where the cache is read from and written to; it
    exists so tests can prove byte-reproducibility in a temp directory without
    touching the repository's evidence tree.
    """
    spec = _spec(key)
    if not spec.produced_here:
        raise PerceptionEncodeError(
            f"feature set {key!r} is produced by medguard_v2.perception.encode, not here; "
            "read it with load_feature_set()"
        )
    cases = list(cases if cases is not None else load_cases())
    if not cases:
        raise PerceptionEncodeError("no cases to encode")
    case_nums = [int(case.case_num) for case in cases]
    if len(set(case_nums)) != len(case_nums):
        raise PerceptionEncodeError("duplicate case_num in the requested cases")

    with _feature_root(feature_root):
        # Resolving the snapshot is offline unless ``allow_download``; it also
        # gives us the fingerprint that the cache-currency check needs, so a
        # swapped checkpoint invalidates the cache instead of being ignored.
        snapshot = encoder_snapshot(spec.encoder_key, allow_download=allow_download)
        fingerprint = checkpoint_fingerprint(snapshot)
        digest = image_set_digest(cases, spec.image_view)
        if not force:
            existing = load_feature_set(key)
            if existing is not None and _cache_is_current(
                existing, cases, spec, digest, str(fingerprint["combined_sha256"])
            ):
                return existing

        _check_images_readable(cases, spec.image_view)
        encode_fn, provenance = _encode.load_encoder(
            spec.encoder_key, allow_download=allow_download
        )
        provenance = dict(provenance)
        provenance["checkpoint"] = fingerprint
        provenance["tower"] = "image"
        provenance["tower_note"] = (
            "open_clip builds the full CLIP model; only encode_image is called, so the "
            "text tower is constructed and never run"
        )
        provenance["allow_download"] = bool(allow_download)
        provenance["checkpoint_source"] = (
            "local Hugging Face cache (offline)"
            if not allow_download
            else "local Hugging Face cache or hub (download permitted)"
        )

        ordered = sorted(cases, key=lambda case: case.case_num)
        effective_batch = int(batch_size or ENCODERS[spec.encoder_key].batch_size)
        if effective_batch <= 0:
            raise PerceptionEncodeError("batch_size must be positive")

        matrix: np.ndarray | None = None
        started = time.perf_counter()
        for start in range(0, len(ordered), effective_batch):
            chunk = ordered[start : start + effective_batch]
            block = _encode_chunk(encode_fn, chunk, spec.image_view)
            if block.ndim != 2 or block.shape[0] != len(chunk):
                raise PerceptionEncodeError(f"encoder returned invalid shape {block.shape}")
            if not np.isfinite(block).all():
                failing = [
                    chunk[i].case_num for i in np.where(~np.isfinite(block).all(axis=1))[0]
                ]
                raise PerceptionEncodeError(f"non-finite features for case_num {failing}")
            norms = np.linalg.norm(block, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise PerceptionEncodeError(
                    f"features are not L2-normalised "
                    f"(min {norms.min():.4f}, max {norms.max():.4f})"
                )
            if matrix is None:
                matrix = np.zeros((len(ordered), block.shape[1]), dtype=np.float32)
            elif block.shape[1] != matrix.shape[1]:
                raise PerceptionEncodeError("feature dimension changed mid-run")
            matrix[start : start + len(chunk)] = block
            if verbose:
                done = start + len(chunk)
                elapsed = time.perf_counter() - started
                print(f"[{key}] {done}/{len(ordered)} images; {elapsed:.1f}s", flush=True)

        assert matrix is not None  # guaranteed: ``ordered`` is non-empty
        elapsed = time.perf_counter() - started
        keys = np.asarray([case.case_num for case in ordered], dtype=np.int64)
        manifest = {
            "protocol": CACHE_PROTOCOL,
            "feature_set_key": key,
            "encoder_key": spec.encoder_key,
            "image_view": spec.image_view,
            "n_cases": len(ordered),
            "feature_dim": int(matrix.shape[1]),
            "l2_normalised": True,
            "batch_size": effective_batch,
            "image_set_sha256": digest,
            # Same private helper ``encode.load_cached_features`` validates
            # against, so a cache written here passes that check unchanged.
            "features_sha256": _encode._features_digest(keys, matrix),
            "duplicate_view": duplicate_view_record(ordered, spec.image_view),
            "note": spec.note,
            # Deliberately no elapsed_seconds and no absolute paths: the manifest
            # body must be byte-identical across runs.  Timing goes to stdout.
            "reproducibility": (
                "manifest body and features.npz are byte-identical across runs on the "
                "same checkpoint and images; nothing here is wall-clock or path dependent"
            ),
            "provenance": provenance,
        }
        _write_bundle(key, keys, matrix, manifest)
        if verbose:
            print(f"[{key}] wrote {feature_path(key)} in {elapsed:.1f}s", flush=True)
        return FeatureBundle(key, keys, matrix, manifest)


class _feature_root:
    """Temporarily point the shared ``FEATURE_ROOT`` somewhere else.

    ``encode.feature_path`` / ``encode.manifest_path`` read a module-level
    constant, and ``encode.py`` cannot be edited to take a root argument, so a
    test that wants an isolated cache directory has to rebind it.  Doing that
    through a context manager keeps the rebinding scoped and restores the real
    root even if encoding raises.
    """

    def __init__(self, root: Path | None) -> None:
        self._root = Path(root) if root is not None else None
        self._previous: Path | None = None

    def __enter__(self) -> None:
        if self._root is None:
            return
        self._previous = _encode.FEATURE_ROOT
        _encode.FEATURE_ROOT = self._root

    def __exit__(self, *exc_info: object) -> None:
        if self._root is None:
            return
        _encode.FEATURE_ROOT = self._previous


def encode_all(
    cases: Sequence[Derm7ptCase] | None = None,
    *,
    allow_download: bool = False,
    batch_size: int | None = None,
    force: bool = False,
    verbose: bool = False,
    feature_root: Path | None = None,
) -> dict[str, FeatureBundle]:
    """Encode every feature set this module owns, plus load the one it does not.

    Returns all four bundles keyed by feature-set name, so a caller building a
    stacked model gets the complete 2x2 from one call.
    """
    cases = list(cases if cases is not None else load_cases())
    bundles: dict[str, FeatureBundle] = {}
    for key in PRODUCED_HERE:
        bundles[key] = encode_feature_set(
            key,
            cases,
            allow_download=allow_download,
            batch_size=batch_size,
            force=force,
            verbose=verbose,
            feature_root=feature_root,
        )
    with _feature_root(feature_root):
        for key in OWNED_ELSEWHERE:
            existing = load_feature_set(key)
            if existing is None:
                raise PerceptionEncodeError(
                    f"feature set {key!r} is owned by medguard_v2.perception.encode and is "
                    f"not cached at {feature_path(key)}; run that module first"
                )
            bundles[key] = existing

    # Free cross-check: the derm digest computed here must equal the one
    # ``encode.py`` recorded, or the two modules are not reading the same pixels.
    derm_digests = {
        key: bundle.manifest.get("image_set_sha256")
        for key, bundle in bundles.items()
        if FEATURE_SETS[key].image_view == "derm"
    }
    if len(set(derm_digests.values())) != 1:
        raise PerceptionEncodeError(
            "derm feature sets disagree about the image set they encoded: " f"{derm_digests}"
        )
    return bundles


# --------------------------------------------------------------------------
# separability report
# --------------------------------------------------------------------------

#: What ``--report`` probes.  A sanity signal that a feature set carries usable
#: information -- not a study.  ``diagnosis_group`` is checked against the
#: registered ``derm7pt`` taxonomy; ``pigment_network`` is a criterion, whose
#: vocabulary lives in the loader (no taxonomy is registered for criteria, and
#: inventing one here would be a second definition of the order).
REPORT_TARGETS = ("diagnosis_group", "pigment_network")

FIT_SPLIT = "train"
EVAL_SPLIT = "valid"


def _target_values(case: Derm7ptCase, target: str) -> str:
    if target == "diagnosis_group":
        return case.diagnosis_group
    return case.criteria[target]


def _target_labels(target: str, cases: Sequence[Derm7ptCase]) -> tuple[str, ...]:
    from medguard_v2.eval.taxonomy import assert_labels_match

    if target == "diagnosis_group":
        return assert_labels_match([case.diagnosis_group for case in cases], "derm7pt")
    labels = CRITERION_VOCABULARY[target]
    observed = {case.criteria[target] for case in cases}
    unknown = sorted(observed - set(labels))
    if unknown:
        raise PerceptionEncodeError(
            f"observed {target} values outside CRITERION_VOCABULARY: {unknown}"
        )
    return tuple(labels)


def separability_report(
    bundles: Mapping[str, FeatureBundle], cases: Sequence[Derm7ptCase] | None = None
) -> dict:
    """Fit a linear probe per (feature set, target) on train; score on valid.

    Deliberately minimal.  It reuses :func:`probe.fit_probe` -- the same C-grid,
    the same out-of-fold log-loss selection rule, the same seed, fit on
    ``train`` only -- so the numbers are directly comparable to the existing
    perception baseline rather than being a second, differently-tuned probe.

    Also reports, for the diagnosis target, the pairwise fraction of valid cases
    on which two feature sets predict different labels.  That is the number that
    says how much complementarity a stacked model could exploit: two feature
    sets with equal accuracy and zero disagreement are the same model twice.
    """
    from medguard_v2.eval.metrics import classification_summary
    from medguard_v2.perception.probe import fit_probe, posterior

    cases = list(cases if cases is not None else load_cases())
    ordered = sorted(cases, key=lambda case: case.case_num)
    train_cases = [case for case in ordered if case.split == FIT_SPLIT]
    eval_cases = [case for case in ordered if case.split == EVAL_SPLIT]
    if not train_cases or not eval_cases:
        raise PerceptionEncodeError("the report needs a non-empty train and valid split")

    keys = [key for key in FEATURE_SETS if key in bundles]
    report: dict = {
        "fit_split": FIT_SPLIT,
        "eval_split": EVAL_SPLIT,
        "n_fit": len(train_cases),
        "n_eval": len(eval_cases),
        "purpose": (
            "sanity signal that each feature set carries usable information, and how much "
            "the four disagree; NOT a stacking study"
        ),
        "feature_sets": {
            key: {
                "encoder_key": FEATURE_SETS[key].encoder_key,
                "image_view": FEATURE_SETS[key].image_view,
                "feature_dim": int(bundles[key].dim),
                "produced_by": (
                    "encode_multi" if FEATURE_SETS[key].produced_here else "encode"
                ),
            }
            for key in keys
        },
        "targets": {},
    }

    predictions: dict[str, dict[str, list[str]]] = {}
    for target in REPORT_TARGETS:
        labels = _target_labels(target, ordered)
        y_train = [_target_values(case, target) for case in train_cases]
        y_eval = [_target_values(case, target) for case in eval_cases]
        per_set: dict[str, dict] = {}
        predictions[target] = {}
        for key in keys:
            bundle = bundles[key]
            x_train = bundle.matrix_for([case.case_num for case in train_cases])
            x_eval = bundle.matrix_for([case.case_num for case in eval_cases])
            fit = fit_probe(
                x_train,
                y_train,
                labels,
                target=target,
                fit_case_nums=[case.case_num for case in train_cases],
            )
            probabilities = posterior(fit, x_eval)
            predicted = [labels[index] for index in probabilities.argmax(axis=1)]
            predictions[target][key] = predicted
            summary = classification_summary(y_eval, predicted, list(labels))
            per_set[key] = {
                "selected_C": fit.selected_c,
                "accuracy": round(float(summary["accuracy"]), 4),
                "balanced_accuracy": round(float(summary["balanced_accuracy"]), 4),
                "macro_f1": round(float(summary["macro_f1"]), 4),
            }
        majority = max(labels, key=lambda label: y_eval.count(label))
        report["targets"][target] = {
            "labels": list(labels),
            "n_classes": len(labels),
            "majority_class": majority,
            "majority_class_accuracy": round(y_eval.count(majority) / len(y_eval), 4),
            "chance_balanced_accuracy": round(1.0 / len(labels), 4),
            "by_feature_set": per_set,
        }

    diagnosis = predictions["diagnosis_group"]
    matrix = [
        [
            round(
                sum(1 for a, b in zip(diagnosis[left], diagnosis[right]) if a != b)
                / len(eval_cases),
                4,
            )
            for right in keys
        ]
        for left in keys
    ]
    report["diagnosis_disagreement"] = {
        "split": EVAL_SPLIT,
        "n": len(eval_cases),
        "order": list(keys),
        "matrix": matrix,
        "definition": "fraction of valid cases on which the two feature sets predict "
        "different diagnosis_group labels",
        "all_four_agree": round(
            sum(
                1
                for row in range(len(eval_cases))
                if len({diagnosis[key][row] for key in keys}) == 1
            )
            / len(eval_cases),
            4,
        ),
    }
    return report


def _print_report(report: dict) -> None:
    print(
        f"\nseparability: fit on {report['fit_split']} (n={report['n_fit']}), "
        f"scored on {report['eval_split']} (n={report['n_eval']})"
    )
    keys = list(report["feature_sets"])
    width = max(len(key) for key in keys) + 2
    for target, block in report["targets"].items():
        print(
            f"\n  {target}  ({block['n_classes']} classes; majority "
            f"{block['majority_class']} = {block['majority_class_accuracy']:.4f}, "
            f"chance bal_acc {block['chance_balanced_accuracy']:.4f})"
        )
        print(f"  {'feature set':<{width}}{'dim':>6}{'acc':>10}{'bal_acc':>10}{'macroF1':>10}")
        for key in keys:
            row = block["by_feature_set"][key]
            dim = report["feature_sets"][key]["feature_dim"]
            print(
                f"  {key:<{width}}{dim:>6}{row['accuracy']:>10.4f}"
                f"{row['balanced_accuracy']:>10.4f}{row['macro_f1']:>10.4f}"
            )

    disagreement = report["diagnosis_disagreement"]
    print(
        f"\n  pairwise diagnosis disagreement on {disagreement['split']} "
        f"(n={disagreement['n']}); all four agree on {disagreement['all_four_agree']:.4f}"
    )
    print(f"  {'':<{width}}" + "".join(f"{key[:12]:>14}" for key in keys))
    for index, key in enumerate(keys):
        row = "".join(f"{value:>14.4f}" for value in disagreement["matrix"][index])
        print(f"  {key:<{width}}{row}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--feature-set",
        choices=sorted(PRODUCED_HERE),
        default=None,
        help="encode only one feature set (default: all three)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="ignore any existing cache")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit fetching a checkpoint that is not in the local cache",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="after encoding, fit a linear probe per feature set and report separability",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="also write the separability report to this path",
    )
    args = parser.parse_args()

    cases = load_cases()
    started = time.perf_counter()
    if args.feature_set is not None:
        bundles = {
            args.feature_set: encode_feature_set(
                args.feature_set,
                cases,
                allow_download=args.allow_download,
                batch_size=args.batch_size,
                force=args.force,
                verbose=True,
            )
        }
    else:
        bundles = encode_all(
            cases,
            allow_download=args.allow_download,
            batch_size=args.batch_size,
            force=args.force,
            verbose=True,
        )
    print(f"\nencoded/loaded {len(bundles)} feature set(s) in {time.perf_counter() - started:.1f}s")
    for key in FEATURE_SETS:
        if key not in bundles:
            continue
        bundle = bundles[key]
        provenance = bundle.manifest.get("provenance", {})
        checkpoint = provenance.get("checkpoint", {})
        duplicates = bundle.manifest.get("duplicate_view", {})
        print(
            f"  {key:<20} view={bundle.manifest.get('image_view'):<7} "
            f"n={bundle.manifest.get('n_cases')} dim={bundle.dim} "
            f"model={provenance.get('resolved_model_id')} "
            f"ckpt={str(checkpoint.get('combined_sha256'))[:12]} "
            f"dup_views={duplicates.get('n_cases_without_distinct_views')}"
        )

    if args.report or args.report_json is not None:
        if len(bundles) != len(FEATURE_SETS):
            bundles = encode_all(
                cases, allow_download=args.allow_download, batch_size=args.batch_size
            )
        report = separability_report(bundles, cases)
        _print_report(report)
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"\nwrote {args.report_json}")


if __name__ == "__main__":  # pragma: no cover
    _main()
