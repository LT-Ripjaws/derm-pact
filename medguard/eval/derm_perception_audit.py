"""Credit-free audit of the frozen dermoscopy perception head on testsub200.

This is a retirement gate for historical development claims, not the final rebuilt
HAM10000 study.  It deduplicates repeated VQA questions by image and bootstraps by
lesion identity.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .. import config
from ..pipeline.image_retriever import ExemplarAtlas
from .metrics import aurc, ece, selective_accuracy_at_coverage
from .statistics import binary_auc, clustered_bootstrap_ci


DEFAULT_TEST = Path(r"E:\Studies\Research\IEEE THAI CONF\data\isic\vqa_testsub200.jsonl")
OUTPUT_DIR = Path("research/experiments/derm_perception/v1_testsub200")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm_path(value: str) -> str:
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 6 and value[6] == "/":
        return value[5].upper() + ":" + value[6:].replace("/", "\\")
    return value


def _read_unique_images(path: Path) -> List[Dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    unique: Dict[str, Dict[str, object]] = {}
    for row in rows:
        image_path = str(row.get("image_path") or row.get("image") or "")
        if not image_path:
            continue
        existing = unique.get(image_path)
        if existing is not None:
            # Repeated VQA questions for one image must never disagree on the
            # perception target or lesion cluster.
            if (existing.get("class"), existing.get("lesion_id")) != (
                    row.get("class"), row.get("lesion_id")):
                raise ValueError(f"conflicting repeated image record: {image_path}")
            continue
        unique[image_path] = row
    return list(unique.values())


def _macro_recall(records: Sequence[Mapping[str, object]]) -> float:
    recalls = []
    for label in config.DIAGNOSIS_LABELS:
        subset = [row for row in records if row["label"] == label]
        if subset:
            recalls.append(sum(row["pred"] == label for row in subset) / len(subset))
    return sum(recalls) / len(recalls) if recalls else 0.0


def _malignant_recall(records: Sequence[Mapping[str, object]]) -> float:
    positive = [row for row in records if row["label"] in config.MALIGNANT_LABELS]
    return (sum(row["pred"] in config.MALIGNANT_LABELS for row in positive) / len(positive)
            if positive else 0.0)


def _false_reassurance(records: Sequence[Mapping[str, object]]) -> float:
    return 1.0 - _malignant_recall(records)


def _malignant_auc(records: Sequence[Mapping[str, object]]) -> float:
    return binary_auc(
        [row["label"] in config.MALIGNANT_LABELS for row in records],
        [float(row["malignant_score"]) for row in records],
    )


def _accuracy(records: Sequence[Mapping[str, object]]) -> float:
    return sum(row["correct"] for row in records) / len(records) if records else 0.0


def _aurc(records: Sequence[Mapping[str, object]]) -> float:
    return aurc([float(row["confidence"]) for row in records],
                [bool(row["correct"]) for row in records])


def _report(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    clusters = [str(row["cluster_id"]) for row in records]
    labels = list(config.DIAGNOSIS_LABELS)
    confusion = {
        truth: {pred: sum(row["label"] == truth and row["pred"] == pred for row in records)
                for pred in labels}
        for truth in labels
    }
    per_class = {
        label: {
            "n": sum(row["label"] == label for row in records),
            "recall": (sum(row["label"] == label and row["pred"] == label for row in records)
                       / sum(row["label"] == label for row in records))
            if any(row["label"] == label for row in records) else None,
        }
        for label in labels
    }
    malignant_truth = [row["label"] in config.MALIGNANT_LABELS for row in records]
    malignant_prob = [float(row["malignant_score"]) for row in records]
    binary_pred = [value >= 0.5 for value in malignant_prob]
    n_pos = sum(malignant_truth)
    n_neg = len(records) - n_pos
    binary_recall = (sum(p and y for p, y in zip(binary_pred, malignant_truth)) / n_pos
                     if n_pos else 0.0)
    specificity = (sum((not p) and (not y) for p, y in zip(binary_pred, malignant_truth)) / n_neg
                   if n_neg else 0.0)
    return {
        "n_images": len(records),
        "n_lesion_clusters": len(set(clusters)),
        "class_counts": dict(Counter(str(row["label"]) for row in records)),
        "diagnosis_accuracy": _accuracy(records),
        "diagnosis_accuracy_ci": asdict(clustered_bootstrap_ci(records, clusters, _accuracy)),
        "macro_recall": _macro_recall(records),
        "per_class": per_class,
        "confusion": confusion,
        "malignant_auc": _malignant_auc(records),
        "malignant_auc_ci": asdict(clustered_bootstrap_ci(records, clusters, _malignant_auc)),
        "committed_malignant_recall": _malignant_recall(records),
        "committed_malignant_recall_ci": asdict(
            clustered_bootstrap_ci(records, clusters, _malignant_recall)),
        "false_reassurance_rate": _false_reassurance(records),
        "false_reassurance_rate_ci": asdict(
            clustered_bootstrap_ci(records, clusters, _false_reassurance)),
        "binary_recall_at_0_5": binary_recall,
        "binary_specificity_at_0_5": specificity,
        "binary_brier": sum((p - float(y)) ** 2 for p, y in zip(malignant_prob, malignant_truth)) / len(records),
        "diagnosis_ece": ece([float(row["confidence"]) for row in records],
                             [bool(row["correct"]) for row in records]),
        "diagnosis_aurc": _aurc(records),
        "diagnosis_aurc_ci": asdict(clustered_bootstrap_ci(records, clusters, _aurc)),
        "selective_accuracy": {
            str(coverage): selective_accuracy_at_coverage(
                [float(row["confidence"]) for row in records],
                [bool(row["correct"]) for row in records], coverage,
            ) for coverage in (0.5, 0.65, 0.8, 0.9)
        },
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_unique_images(DEFAULT_TEST)
    artifacts = {
        "test_jsonl": DEFAULT_TEST,
        "atlas_index": Path(config.EXEMPLAR_INDEX_PATH),
        "atlas_meta": Path(config.EXEMPLAR_META_PATH),
        "diagnosis_probe": Path(config.EXEMPLAR_DIAG_CLF),
        "malignancy_probe": Path(config.EXEMPLAR_MALIGNANT_CLF),
        "implementation": Path(__file__),
    }
    missing = [str(path) for path in artifacts.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"audit artifacts missing: {missing}")
    manifest = {
        "protocol": "derm-perception-testsub200-v1",
        "created": "2026-07-13",
        "scope": "credit-free retirement gate; not final rebuilt HAM10000 result",
        "deduplication": "first source record per image_path; bootstrap by lesion_id",
        "n_source_rows": sum(1 for line in DEFAULT_TEST.read_text(encoding="utf-8").splitlines() if line.strip()),
        "n_unique_images": len(rows),
        "encoder_model": config.EXEMPLAR_ENCODER_MODEL,
        "python": platform.python_version(),
        "artifact_sha256": {name: _sha(path) for name, path in artifacts.items()},
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (OUTPUT_DIR / "manifest.json").write_text(manifest_text, encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()

    atlas = ExemplarAtlas.load()
    records: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id") or row.get("id") or index)
        try:
            probe = atlas.probe(_norm_path(str(row["image_path"])))
            label = str(row["class"]).lower()
            records.append({
                "case_id": case_id,
                "cluster_id": str(row.get("lesion_id") or case_id),
                "image_id": str(row.get("case_id") or case_id),
                "label": label,
                "pred": probe.exemplar_pred,
                "confidence": probe.exemplar_confidence,
                "malignant_score": probe.exemplar_malignant_score,
                "correct": probe.exemplar_pred == label,
            })
        except Exception as exc:  # noqa: BLE001 -- failures remain explicit in result
            failures.append({"case_id": case_id, "error_type": type(exc).__name__,
                             "message": str(exc)[:200]})
        if index % 10 == 0 or index == len(rows):
            print(f"[derm-perception] {index}/{len(rows)} images; failures={len(failures)}", flush=True)

    result = {
        "protocol": "derm-perception-testsub200-v1",
        "manifest_sha256": manifest_sha,
        "n_failures": len(failures),
        "failures": failures,
        "metrics": _report(records) if records else {},
        "records": records,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "manifest_sha256": manifest_sha, "n_failures": len(failures),
        "metrics": result["metrics"],
        "result_path": str((OUTPUT_DIR / "result.json").resolve()),
    }, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
