"""Fail-closed verification for the frozen dermatology evidence bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_OUTPUTS = {
    "audit.json",
    "bm25_index.json",
    "citation_registry.jsonl",
    "corpus.jsonl",
    "exemplar.faiss",
    "exemplar_meta.jsonl",
    "source_policy.json",
    "text_dense.faiss",
}

# The C2 receipt predates the decision to keep C1 code exploratory until the
# pre-H2H gate and therefore hashes these runtime integration files alongside
# true corpus-build inputs.  Preserve the receipt bytes, verify that these files
# still exist, and report drift explicitly.  All data, index, source-policy,
# retriever, and freezer hashes remain hard failures.
#
# Membership is decided by one question only: can this file's bytes change the
# frozen bundle's bytes?  freeze_derm_evidence_v1 imports data.manifest,
# evidence.source_policy and pipeline.hybrid_retriever -- none of which reach
# any file below, so none of them can.  Do not add a file here to silence a
# failure; if the freezer can reach it, the drift is real and the bundle is
# stale.  The durable fix is a receipt that pins build inputs only; see
# research/EXPERIMENT_SPEC_V2.json.
EXPLORATORY_RUNTIME_INPUTS = {
    "medguard/evidence/bundle.py",
    "medguard/evidence/contract.py",
    "medguard/evidence/verification.py",
    "medguard/pipeline/generator.py",
    "medguard/pipeline/llm.py",
    "medguard/pipeline/ops.py",
}


@dataclass(frozen=True)
class FrozenEvidenceRuntime:
    """Receipt-verified retrieval and policy objects for an experiment arm."""

    text_retriever: Any
    image_retriever: Any
    source_policy: Any
    receipt: Dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_locator(locator: str) -> Path:
    candidate = Path(locator)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _verify_sidecar(path: Path, observed: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"evidence artifact has no SHA-256 sidecar: {path}")
    expected = sidecar.read_text(encoding="ascii").strip()
    if expected != observed:
        raise ValueError(f"evidence artifact sidecar mismatch: {path}")


def verify_evidence_bundle(base_dir: str | Path | None = None) -> Dict[str, Any]:
    """Verify the receipt, every frozen output, and every receipt-bound input.

    The function returns the parsed receipt only after the entire bundle and the
    data/build inputs match their raw-byte SHA-256 values. Exploratory C1
    runtime drift is returned under ``_verification`` and is re-locked before
    paid H2H; it does not mutate or masquerade as the frozen C2 receipt.
    """
    if base_dir is None:
        from .. import config

        base = Path(config.EVIDENCE_BUNDLE_DIR).resolve()
    else:
        base = Path(base_dir).resolve()
    receipt_path = base / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"frozen evidence receipt missing: {receipt_path}")
    receipt_sha = _sha256(receipt_path)
    _verify_sidecar(receipt_path, receipt_sha)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "medguard-derm-evidence-freeze-receipt-v1":
        raise ValueError("unsupported evidence receipt schema")
    if (receipt.get("sealed_test_access") is not False
            or int(receipt.get("test_records_loaded", -1)) != 0
            or int(receipt.get("generation_calls", -1)) != 0):
        raise ValueError("evidence receipt violates the offline development boundary")

    outputs = receipt.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("evidence receipt has no output inventory")
    seen = set()
    for row in outputs:
        path = _resolve_locator(str(row.get("path") or ""))
        if path.parent != base or path.name not in EXPECTED_OUTPUTS:
            raise ValueError(f"unexpected evidence output locator: {path}")
        if path.name in seen:
            raise ValueError(f"duplicate evidence output locator: {path.name}")
        seen.add(path.name)
        if not path.is_file():
            raise FileNotFoundError(f"frozen evidence output missing: {path}")
        observed = _sha256(path)
        if observed != row.get("sha256") or path.stat().st_size != int(row.get("bytes", -1)):
            raise ValueError(f"frozen evidence output differs from receipt: {path}")
        _verify_sidecar(path, observed)
    if seen != EXPECTED_OUTPUTS:
        raise ValueError(f"evidence output inventory mismatch: {sorted(EXPECTED_OUTPUTS - seen)}")

    bound_inputs = list(receipt.get("inputs") or []) + [receipt.get("generator") or {}]
    runtime_drift = []
    for row in bound_inputs:
        locator = str(row.get("path") or "").replace("\\", "/")
        path = _resolve_locator(locator)
        if not path.is_file():
            raise ValueError(f"receipt-bound evidence input changed: {path}")
        observed = _sha256(path)
        if locator in EXPLORATORY_RUNTIME_INPUTS:
            if observed != row.get("sha256"):
                runtime_drift.append({
                    "path": locator,
                    "receipt_sha256": row.get("sha256"),
                    "current_sha256": observed,
                })
            continue
        if observed != row.get("sha256"):
            raise ValueError(f"receipt-bound evidence input changed: {path}")

    audit = json.loads((base / "audit.json").read_text(encoding="utf-8"))
    if (audit.get("valid") is not True
            or int(audit.get("corpus_documents", -1)) != int(receipt.get("corpus_documents", -2))
            or int(audit.get("test_records_loaded", -1)) != 0):
        raise ValueError("evidence audit and receipt disagree")
    receipt["_verification"] = {
        "exploratory_runtime_drift": runtime_drift,
        "runtime_lock_required_before_paid_h2h": True,
    }
    return receipt


def load_frozen_evidence_runtime(
    base_dir: str | Path | None = None,
    *,
    text_embedder: Any = None,
    image_encoder: Any = None,
    reranker: Any = None,
    candidate_k: int = 20,
) -> FrozenEvidenceRuntime:
    """Construct the dense+sparse text retriever and train-only image atlas.

    Heavy dependencies remain lazy.  Passing fake encoders/embedders keeps this
    factory testable without changing the frozen artifact paths.
    """
    from .. import config
    from ..pipeline.hybrid_retriever import FrozenBM25Retriever, HybridRetriever
    from ..pipeline.image_retriever import ExemplarAtlas, make_encoder
    from ..pipeline.retriever import FaissTextRetriever
    from .source_policy import CitationPolicyRegistry

    base = Path(base_dir or config.EVIDENCE_BUNDLE_DIR).resolve()
    receipt = verify_evidence_bundle(base)
    dense = FaissTextRetriever.load(
        index_path=str(base / "text_dense.faiss"),
        meta_path=str(base / "corpus.jsonl"),
        embedder=text_embedder,
    )
    sparse = FrozenBM25Retriever.load(base / "corpus.jsonl", base / "bm25_index.json")
    text_retriever = HybridRetriever(
        dense,
        sparse,
        reranker=reranker,
        candidate_k=int(candidate_k),
    )
    image_retriever = ExemplarAtlas.load(
        index_path=base / "exemplar.faiss",
        meta_path=base / "exemplar_meta.jsonl",
        encoder=image_encoder or make_encoder(),
        load_probes=False,
        strict_train_only=True,
    )
    source_policy = CitationPolicyRegistry.from_files(
        base / "source_policy.json", base / "citation_registry.jsonl")
    return FrozenEvidenceRuntime(
        text_retriever=text_retriever,
        image_retriever=image_retriever,
        source_policy=source_policy,
        receipt=receipt,
    )
