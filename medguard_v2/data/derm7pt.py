"""Frozen loader for the Derm7pt release_v0 dataset.

The dataset itself (Kawahara et al., "Seven-Point Checklist and Skin Lesion
Classification using Multi-Task Multi-Modal Neural Nets") is distributed under
CC BY-NC-ND: it must not be redistributed.  Nothing in this module copies image
bytes anywhere; it only reads metadata and resolves paths on disk.

Two on-disk realities this loader has to survive:

* The ``clinic``/``derm`` columns in ``meta.csv`` do not always match the true
  case of the file on disk (verified: ``FCl/Fcl068.jpg`` in the CSV is
  ``FCL/Fcl068.jpg`` on disk).  On Windows/macOS the mismatch is invisible; on a
  case-sensitive filesystem it is a hard failure.  We therefore resolve every
  path through a case-insensitive index of the image tree and store the *true*
  on-disk path, so downstream artefacts are portable.
* Nine cases point ``clinic`` and ``derm`` at the same file (the ``notes``
  column records that the clinical image was missing and was replaced with the
  dermoscopic one).  That is real, documented, and preserved here rather than
  silently patched -- but it is exposed via :func:`cases_without_distinct_views`
  so a two-view model cannot pretend it has two views.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "Derm7ptCase",
    "Derm7ptDataError",
    "DERM7PT_CLASS_ORDER",
    "DIAGNOSIS_GROUPS",
    "DIAGNOSIS_GROUP_ORDER",
    "SEVEN_CRITERIA",
    "CRITERION_VOCABULARY",
    "SPLITS",
    "EXPECTED_SPLIT_SIZES",
    "DEFAULT_ROOT",
    "analysis_cohort",
    "class_counts",
    "split_counts",
    "cases_without_distinct_views",
    "load_cases",
]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "release_v0"

N_CASES = 1011

SPLITS = ("train", "valid", "test")
EXPECTED_SPLIT_SIZES = {"train": 413, "valid": 203, "test": 395}

#: Canonical order of the seven-point checklist criteria (column order in meta.csv).
SEVEN_CRITERIA = (
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
)

#: Observed value vocabulary for each criterion (verified against all 1011 rows).
CRITERION_VOCABULARY: Mapping[str, tuple[str, ...]] = {
    "pigment_network": ("absent", "atypical", "typical"),
    "streaks": ("absent", "irregular", "regular"),
    "pigmentation": (
        "absent",
        "diffuse irregular",
        "diffuse regular",
        "localized irregular",
        "localized regular",
    ),
    "regression_structures": ("absent", "blue areas", "combinations", "white areas"),
    "dots_and_globules": ("absent", "irregular", "regular"),
    "blue_whitish_veil": ("absent", "present"),
    "vascular_structures": (
        "absent",
        "arborizing",
        "comma",
        "dotted",
        "hairpin",
        "linear irregular",
        "within regression",
        "wreath",
    ),
}

#: Canonical column order for the 5-class grouping used by the dataset authors
#: (Kawahara et al.).  This is the taxonomy Derm7pt work must be reported in.
#:
#: It lives *here* and not in ``medguard_v2/config.py`` deliberately.  ``config.py``
#: exists for one reason: ``eval/metrics.py`` was ported verbatim from v1 and carries
#: ``from ..config import MALIGNANT_LABELS``.  Everything in that module is the
#: **HAM10000** 7-class vocabulary copied byte-for-byte from v1, and
#: ``tests/test_metrics_parity.py`` pins it against v1.  Adding a second,
#: incompatible taxonomy there would (a) park the two taxonomies one
#: attribute-access apart in the same namespace -- precisely the confusion that
#: lets a caller hand ``DIAGNOSIS_LABELS`` to Derm7pt data -- and (b) stop
#: ``config.py`` from being "the v1 vocabulary, verbatim", which is the only
#: property that makes the parity test meaningful.  The 5-class grouping is a
#: property of *this dataset*: it is the range of ``DIAGNOSIS_GROUPS`` below, so
#: it belongs with the loader that produces it.  ``medguard_v2/eval/taxonomy.py``
#: imports it from here to resolve the name ``"derm7pt"``, so there is exactly
#: one definition of the order in the package.
#:
#: Beware the near-collision with ``metrics.DERM_CLASS_ORDER`` -- that is
#: HAM10000, seven classes, lowercase, a *different* taxonomy.  Keeping the two
#: apart is what ``eval/taxonomy.py`` is for.
DERM7PT_CLASS_ORDER = ("BCC", "NEV", "MEL", "SK", "MISC")

#: Backwards-compatible alias (same object).  The unqualified name does not say
#: which taxonomy it belongs to, so new code should prefer
#: :data:`DERM7PT_CLASS_ORDER`.
DIAGNOSIS_GROUP_ORDER = DERM7PT_CLASS_ORDER

DIAGNOSIS_GROUPS: Mapping[str, str] = {
    # basal cell carcinoma
    "basal cell carcinoma": "BCC",
    # nevus (all variants)
    "blue nevus": "NEV",
    "clark nevus": "NEV",
    "combined nevus": "NEV",
    "congenital nevus": "NEV",
    "dermal nevus": "NEV",
    "recurrent nevus": "NEV",
    "reed or spitz nevus": "NEV",
    # melanoma (all variants, including in-situ and metastasis)
    "melanoma": "MEL",
    "melanoma (in situ)": "MEL",
    "melanoma (less than 0.76 mm)": "MEL",
    "melanoma (0.76 to 1.5 mm)": "MEL",
    "melanoma (more than 1.5 mm)": "MEL",
    "melanoma metastasis": "MEL",
    # seborrheic keratosis
    "seborrheic keratosis": "SK",
    # everything else
    "dermatofibroma": "MISC",
    "lentigo": "MISC",
    "melanosis": "MISC",
    "miscellaneous": "MISC",
    "vascular lesion": "MISC",
}

MANAGEMENT_VOCABULARY = ("clinical follow up", "excision", "no further examination")
DIFFICULTY_VOCABULARY = ("low", "medium", "high")
ELEVATION_VOCABULARY = ("flat", "nodular", "palpable")
SEX_VOCABULARY = ("female", "male")


class Derm7ptDataError(RuntimeError):
    """Raised when the on-disk dataset does not match the frozen expectations."""


@dataclass(frozen=True)
class Derm7ptCase:
    """One Derm7pt case: metadata plus the two resolved image paths."""

    case_index: int
    case_num: int
    diagnosis: str
    diagnosis_group: str
    management: str
    seven_point_score: int
    criteria: Mapping[str, str]
    difficulty: str
    elevation: str
    location: str
    sex: str
    clinic_path: Path
    derm_path: Path
    split: str

    @property
    def has_distinct_views(self) -> bool:
        """False when the clinical image is a copy of the dermoscopic one."""
        return self.clinic_path != self.derm_path


def _index_images(images_root: Path) -> dict[str, Path]:
    """Map lowercased ``DIR/FILE`` to the true on-disk path.

    Raises if two files differ only by case, which would make resolution
    ambiguous (verified absent in release_v0, but a silent wrong pick would be
    worse than a loud failure).
    """
    if not images_root.is_dir():
        raise Derm7ptDataError(f"image root does not exist: {images_root}")
    index: dict[str, Path] = {}
    for directory in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            key = f"{directory.name}/{path.name}".lower()
            if key in index:
                raise Derm7ptDataError(
                    "ambiguous case-insensitive image path "
                    f"{key!r}: {index[key]} and {path}"
                )
            index[key] = path
    return index


def _resolve_image(rel: str, images_root: Path, index: Mapping[str, Path]) -> Path:
    normalised = rel.strip().replace("\\", "/")
    if not normalised:
        raise Derm7ptDataError("empty image path in meta.csv")
    direct = images_root / normalised
    key = normalised.lower()
    resolved = index.get(key)
    if resolved is None:
        raise Derm7ptDataError(
            f"image path does not resolve on disk: {rel!r} (looked for {direct})"
        )
    return resolved.resolve()


def _read_splits(meta_dir: Path) -> dict[int, str]:
    """Return case_index -> split, verifying an exact partition of 0..1010."""
    per_split: dict[str, list[int]] = {}
    for split in SPLITS:
        path = meta_dir / f"{split}_indexes.csv"
        if not path.is_file():
            raise Derm7ptDataError(f"missing split index file: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "indexes" not in rows[0]:
            raise Derm7ptDataError(f"{path} does not have an 'indexes' column")
        per_split[split] = [int(row["indexes"]) for row in rows]

    problems: list[str] = []
    assignment: dict[int, str] = {}
    for split in SPLITS:
        values = per_split[split]
        duplicates = sorted({v for v in values if values.count(v) > 1})
        if duplicates:
            problems.append(f"{split} repeats indexes {duplicates}")
        for value in values:
            owner = assignment.get(value)
            if owner is not None and owner != split:
                problems.append(f"index {value} appears in both {owner} and {split}")
            assignment[value] = split

    expected = set(range(N_CASES))
    seen = set(assignment)
    uncovered = sorted(expected - seen)
    out_of_range = sorted(seen - expected)
    if uncovered:
        problems.append(f"indexes never assigned to a split: {uncovered}")
    if out_of_range:
        problems.append(f"indexes outside 0..{N_CASES - 1}: {out_of_range}")
    for split, size in EXPECTED_SPLIT_SIZES.items():
        actual = len(per_split[split])
        if actual != size:
            problems.append(f"{split} has {actual} rows, expected {size}")

    if problems:
        raise Derm7ptDataError(
            "split indexes do not partition 0..%d exactly: %s"
            % (N_CASES - 1, "; ".join(problems))
        )
    return assignment


def load_cases(root: Path = DEFAULT_ROOT) -> list[Derm7ptCase]:
    """Load all Derm7pt cases with their official split labels.

    Raises :class:`Derm7ptDataError` if any image path fails to resolve, if a
    diagnosis string is not in :data:`DIAGNOSIS_GROUPS`, or if the split index
    files do not partition ``0..1010`` exactly.
    """
    root = Path(root)
    meta_path = root / "meta" / "meta.csv"
    if not meta_path.is_file():
        raise Derm7ptDataError(f"missing meta.csv: {meta_path}")

    splits = _read_splits(root / "meta")
    images_root = root / "images"
    index = _index_images(images_root)

    with meta_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != N_CASES:
        raise Derm7ptDataError(f"expected {N_CASES} rows in meta.csv, found {len(rows)}")

    cases: list[Derm7ptCase] = []
    unresolved: list[str] = []
    unmapped: list[str] = []
    for case_index, row in enumerate(rows):
        diagnosis = row["diagnosis"].strip()
        group = DIAGNOSIS_GROUPS.get(diagnosis)
        if group is None:
            unmapped.append(f"row {case_index}: {diagnosis!r}")
            continue
        try:
            clinic = _resolve_image(row["clinic"], images_root, index)
            derm = _resolve_image(row["derm"], images_root, index)
        except Derm7ptDataError as exc:
            unresolved.append(f"row {case_index} (case_num {row['case_num']}): {exc}")
            continue
        cases.append(
            Derm7ptCase(
                case_index=case_index,
                case_num=int(row["case_num"]),
                diagnosis=diagnosis,
                diagnosis_group=group,
                management=row["management"].strip(),
                seven_point_score=int(row["seven_point_score"]),
                criteria={name: row[name].strip() for name in SEVEN_CRITERIA},
                difficulty=row["level_of_diagnostic_difficulty"].strip(),
                elevation=row["elevation"].strip(),
                location=row["location"].strip(),
                sex=row["sex"].strip(),
                clinic_path=clinic,
                derm_path=derm,
                split=splits[case_index],
            )
        )

    if unmapped:
        raise Derm7ptDataError(
            "diagnosis strings with no group in DIAGNOSIS_GROUPS: " + "; ".join(unmapped)
        )
    if unresolved:
        raise Derm7ptDataError(
            "image paths that do not resolve on disk (%d): %s"
            % (len(unresolved), "; ".join(unresolved[:10]))
        )
    return cases


def split_counts(cases: list[Derm7ptCase]) -> dict[str, int]:
    counts = {split: 0 for split in SPLITS}
    for case in cases:
        counts[case.split] += 1
    return counts


def class_counts(cases: list[Derm7ptCase]) -> dict[str, int]:
    counts = {group: 0 for group in DIAGNOSIS_GROUP_ORDER}
    for case in cases:
        counts[case.diagnosis_group] += 1
    return counts


def cases_without_distinct_views(cases: list[Derm7ptCase]) -> list[Derm7ptCase]:
    """Cases whose clinical image is literally the dermoscopic image."""
    return [case for case in cases if not case.has_distinct_views]


def _observed_class_counts(cases: Sequence[Derm7ptCase]) -> dict[str, int]:
    """Per-class counts in canonical order, omitting classes that do not occur."""
    counts = class_counts(list(cases))
    return {group: n for group, n in counts.items() if n}


def analysis_cohort(
    cases: Sequence[Derm7ptCase], *, require_distinct_views: bool = False
) -> tuple[list[Derm7ptCase], dict[str, object]]:
    """Return the cases to analyse **plus** an explicit record of what was excluded.

    With ``require_distinct_views=True`` this drops the nine cases whose
    "clinical" image is literally the dermoscopic file (documented in
    ``notes/DERM7PT_DATA_CAVEATS.md`` and exposed by
    :func:`cases_without_distinct_views`).  Ablation A6 -- "drop the clinical
    image" -- is a **no-op** on those cases, so leaving them in dilutes the
    measured cross-modality effect with cases that never had two views.

    The exclusion is *returned*, not logged and forgotten, because it is not
    distribution-neutral: every one of the nine is MEL or MISC, none is
    NEV/BCC/SK.  Any analysis that calls this can therefore print its own
    denominator and the per-class cost of the filter instead of quoting an
    ``n`` that silently disagrees with the dataset's headline 1011.

    The two count tables have deliberately different shapes:
    ``kept_class_counts`` is the full canonical table (zeros included) because
    it is a denominator; ``dropped_class_counts`` lists only the classes that
    actually lost cases, because its job is to make the skew visible.
    """
    all_cases = list(cases)
    if require_distinct_views:
        kept = [case for case in all_cases if case.has_distinct_views]
        dropped = [case for case in all_cases if not case.has_distinct_views]
    else:
        kept, dropped = all_cases, []

    report: dict[str, object] = {
        "filter": "distinct_clinical_view" if require_distinct_views else "none",
        "require_distinct_views": require_distinct_views,
        "n_input": len(all_cases),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "dropped_case_nums": sorted(case.case_num for case in dropped),
        "dropped_class_counts": _observed_class_counts(dropped),
        "kept_class_counts": class_counts(kept),
        # When the filter is off, say how many no-distinct-view cases were kept,
        # so a caller cannot mistake "nothing was dropped" for "nothing to drop".
        "retained_without_distinct_views": sum(
            1 for case in kept if not case.has_distinct_views
        ),
        "reason": (
            "clinic image is a copy of the derm image, so an ablation that drops "
            "the clinical view is a no-op on these cases"
            if require_distinct_views
            else None
        ),
    }
    return kept, report


def _main() -> None:
    cases = load_cases()
    print(f"cases: {len(cases)}")
    print("splits:", split_counts(cases))
    print("classes:")
    for group in DIAGNOSIS_GROUP_ORDER:
        print(f"  {group:<5} {class_counts(cases)[group]}")
    shared = cases_without_distinct_views(cases)
    print(f"cases with clinic == derm (no distinct clinical view): {len(shared)}")
    for case in shared:
        print(f"  case_num {case.case_num} ({case.diagnosis}) -> {case.clinic_path.name}")


if __name__ == "__main__":
    _main()
