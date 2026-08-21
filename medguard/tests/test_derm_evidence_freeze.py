from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile

from medguard.data.freeze_derm_evidence_v1 import (
    GUIDELINE_PMIDS,
    ROOT,
    build_bm25_artifact,
    build_citation_registry,
    build_source_policy,
    normalize_text_corpus,
)
from medguard.evidence.source_policy import CitationPolicyRegistry
from medguard.evidence.bundle import EXPLORATORY_RUNTIME_INPUTS, verify_evidence_bundle
from medguard.pipeline.hybrid_retriever import FrozenBM25Retriever


def _guidelines():
    return [
        {
            "authority_class": "clinical_practice_guideline",
            "citation_id": f"PMID:{pmid}",
            "eligible_claim_types": ["management"],
            "pmid": pmid,
            "publication_date": "2020-01",
            "title": f"Guideline {pmid}",
            "abstract": f"Evidence based management recommendation {pmid}",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
        for pmid in GUIDELINE_PMIDS
    ]


def _mini_corpus():
    literature = [{
        "pmid": "1",
        "source": "1",
        "title": "Melanoma visual patterns",
        "abstract": "Dermoscopy supports differential diagnosis.",
    }]
    return normalize_text_corpus(literature, _guidelines())


def test_source_policy_separates_management_authority_from_literature_and_tampering():
    corpus = _mini_corpus()
    citations = {row["document_id"]: row for row in build_citation_registry(corpus)}
    registry = CitationPolicyRegistry(build_source_policy(), citations)
    literature = next(row for row in corpus if row["pmid"] == "1")
    guideline = next(row for row in corpus if row["pmid"] == GUIDELINE_PMIDS[0])
    assert registry.decide(literature, "diagnosis").admitted
    decision = registry.decide(literature, "management")
    assert not decision.admitted and decision.reason == "claim_type_not_eligible"
    assert registry.decide(guideline, "management").admitted
    tampered = dict(guideline, text=guideline["text"] + " Unsafe replacement.")
    decision = registry.decide(tampered, "management")
    assert not decision.admitted and decision.reason == "content_hash_mismatch"


def test_frozen_bm25_roundtrip_is_hash_bound_and_matches_expected_retrieval():
    corpus = _mini_corpus()
    corpus_raw = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in corpus
    )
    artifact = build_bm25_artifact(corpus, hashlib.sha256(corpus_raw).hexdigest())
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        corpus_path = base / "corpus.jsonl"
        index_path = base / "bm25.json"
        corpus_path.write_bytes(corpus_raw)
        index_path.write_text(json.dumps(artifact), encoding="utf-8")
        retriever = FrozenBM25Retriever.load(corpus_path, index_path)
        rows, scores = retriever.search("melanoma dermoscopy", 1)
        assert rows[0]["pmid"] == "1"
        assert scores[0] > 0.0
        corpus_path.write_bytes(corpus_raw + b"\n")
        try:
            FrozenBM25Retriever.load(corpus_path, index_path)
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("tampered corpus passed frozen BM25 validation")


def test_repository_evidence_freeze_is_receipted_and_train_only():
    base = ROOT / "research" / "evidence" / "dermatology_corpus_v1"
    receipt_path = base / "receipt.json"
    assert receipt_path.exists(), "run medguard.data.freeze_derm_evidence_v1"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["test_records_loaded"] == 0
    assert receipt["generation_calls"] == 0
    assert receipt["corpus_documents"] == 1243
    assert receipt["curated_guideline_documents"] == 3
    outputs = {row["path"]: row for row in receipt["outputs"]}
    for locator, row in outputs.items():
        path = ROOT / locator
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    input_paths = {row["path"] for row in receipt["inputs"]}
    assert {
        "medguard/config.py",
        "medguard/evidence/bundle.py",
        "medguard/evidence/source_policy.py",
        "medguard/pipeline/hybrid_retriever.py",
        "medguard/pipeline/image_retriever.py",
        "medguard/pipeline/ops.py",
        "medguard/pipeline/retriever.py",
    }.issubset(input_paths)
    meta = [json.loads(line) for line in (base / "exemplar_meta.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(meta) == 7010
    assert len({row["cluster_id"] for row in meta}) == 5228
    assert {row["split"] for row in meta} == {"train"}
    assert len({row["id"] for row in meta}) == len(meta)


def test_repository_evidence_bundle_verifies_before_runtime_loading():
    receipt = verify_evidence_bundle(ROOT / "research" / "evidence" / "dermatology_corpus_v1")
    assert receipt["corpus_documents"] == 1243
    assert receipt["exemplar"]["records"] == 7010


def _freezer_reachable_modules():
    """Every medguard module the evidence freezer can reach, transitively."""
    seen = set()
    queue = ["medguard.data.freeze_derm_evidence_v1"]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        source = ROOT / (name.replace(".", "/") + ".py")
        if not source.is_file():
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                target = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("medguard."):
                        queue.append(alias.name)
                continue
            elif isinstance(node, ast.ImportFrom) and node.level:
                package = name.rsplit(".", node.level)[0]
                target = f"{package}.{node.module}" if node.module else package
            else:
                continue
            if not target.startswith("medguard."):
                continue
            queue.append(target)
            for alias in getattr(node, "names", []):
                queue.append(f"{target}.{alias.name}")
    return seen


def test_exploratory_runtime_inputs_are_unreachable_from_the_freezer():
    """A file may be exempt from receipt drift only if it cannot change the bundle.

    EXPLORATORY_RUNTIME_INPUTS exists because the C2 receipt over-pins runtime
    integration code alongside true corpus-build inputs.  The exemption is only
    honest while the freezer cannot reach those files -- otherwise their drift
    would mean the frozen bundle is genuinely stale and we would be hiding it.
    This guards against the tempting fix of appending a file here to turn a
    failing verification green.
    """
    reachable = _freezer_reachable_modules()
    for locator in EXPLORATORY_RUNTIME_INPUTS:
        module = locator[: -len(".py")].replace("/", ".")
        assert module not in reachable, (
            f"{locator} is reachable from freeze_derm_evidence_v1 and therefore "
            f"can change the frozen bundle. Its receipt drift is real: re-freeze "
            f"the bundle instead of exempting the file."
        )
