"""Offline C1+C2 development preflight over the 160-lesion H2H manifest.

This is an engineering validation, not a paper result.  It consumes frozen,
truth-redacted perception commitments, performs the receipt-bound C2 hybrid
retrieval, and scores host-built C1 contracts without generation or test data.
Retrieval relevance and provenance never establish semantic entailment.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from medguard.evidence import (
    EvidenceContract,
    UncertaintyType,
    UnresolvedUncertainty,
    VerificationState,
    contract_sha256,
    evidence_from_chunk,
    load_development_perception_bindings,
    load_frozen_evidence_runtime,
    score_contract,
    verify_contract,
)
from medguard.evidence.claim_builder import (
    DIAGNOSIS_CLAIM_ID,
    NO_EXEMPLAR_SUPPORT_DESCRIPTION,
    NO_EXEMPLAR_SUPPORT_TERMINATION,
    NO_EXEMPLAR_SUPPORT_UNCERTAINTY_ID,
    build_image_bound_claims,
)
from medguard.evidence.guideline_claims import (
    GUIDELINE_CHANNEL,
    build_guideline_claim_assets,
    guideline_evidence_item,
    select_guideline_evidence,
)
from medguard.pipeline.guideline_retriever import FrozenGuidelineRetriever
from medguard.pipeline.hybrid_retriever import reciprocal_rank_fusion_records


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research/manifests/medguard_h2h_development_v1.json"
FEATURE_MANIFEST = ROOT / (
    "research/experiments/derm_perception/v2_tournament/features/dermlip/"
    "feature_manifest.json"
)
FEATURES = FEATURE_MANIFEST.with_name("features.npy")
DEFAULT_OUTPUT = ROOT / (
    "research/experiments/medguard_h2h_v1/development_preflight.json"
)
SCHEMA_VERSION = "medguard-development-preflight-v1"
QUESTION = (
    "What diagnosis is most consistent with the dermoscopic images, and what "
    "evidence supports that assessment?"
)
QUERY_TERMS = {
    "akiec": "actinic keratosis intraepidermal carcinoma",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesion",
    "df": "dermatofibroma",
    "melanoma": "cutaneous melanoma",
    "nevus": "melanocytic nevus",
    "vasc": "vascular skin lesion",
}
TEXT_CANDIDATE_K = 20
IMAGE_CANDIDATE_K = 20
FINAL_K = 5
GUIDELINE_CORPUS = ROOT / "medguard/data/corpus/derm_guideline_recommendations_v1.jsonl"
# Authoritative-guideline citations added per covered lesion. A separate tier from
# the FINAL_K fused ceiling; condition-scoped, so benign commitments reach zero.
GUIDELINE_K = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _path_key(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).resolve()))


class FrozenDermLIPFeatureEncoder:
    """Lookup encoder over the receipt-bound train+calibration feature matrix."""

    def __init__(
        self,
        feature_manifest: Path = FEATURE_MANIFEST,
        feature_file: Path = FEATURES,
    ) -> None:
        import numpy as np

        metadata = json.loads(feature_manifest.read_text(encoding="utf-8"))
        if metadata.get("test_records_loaded") != 0:
            raise ValueError("feature artifact violates the development boundary")
        if _sha256(feature_file) != metadata.get("feature_file_sha256"):
            raise ValueError("DermLIP feature matrix hash mismatch")
        parent = Path(str(metadata["development_manifest"]))
        if _sha256(parent) != metadata.get("development_manifest_sha256"):
            raise ValueError("DermLIP feature parent-manifest hash mismatch")
        raw_parent = json.loads(parent.read_text(encoding="utf-8"))
        parent_rows = raw_parent.get("records", [])
        matrix = np.load(feature_file, mmap_mode="r")
        if (
            int(metadata.get("n_records", -1)) != len(parent_rows)
            or matrix.shape != (len(parent_rows), int(metadata.get("feature_dim", -1)))
        ):
            raise ValueError("DermLIP feature rows do not match their manifest")
        paths: Dict[str, int] = {}
        for index, row in enumerate(parent_rows):
            if row.get("split") not in {"train", "calibration"}:
                raise ValueError("feature parent contains a non-development split")
            key = _path_key(str(row["image_path"]))
            if key in paths:
                raise ValueError(f"duplicate feature image path: {key}")
            paths[key] = index
        self._matrix = matrix
        self._paths = paths
        self.model_id = str(metadata["model_id"])
        self.feature_sha256 = str(metadata["feature_file_sha256"])

    def attach_sealed_features(self, manifest_path: Path) -> int:
        """Extend the lookup with frozen SEALED query features.

        Opt-in and explicit: a development run never calls this, so it can never
        resolve a sealed image. Required because the exemplar channel encodes the
        patient's own images through this lookup, and sealed images are absent from the
        development feature cache -- discovered by the sealed pipeline smoke test.

        The exemplar ATLAS is untouched and remains train-only; only the query side is
        extended.
        """
        import numpy as np

        manifest_path = Path(manifest_path)
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "medguard-sealed-feature-manifest-v1":
            raise ValueError("unsupported sealed feature manifest schema")
        if metadata.get("truth_fields_present") is not False:
            raise ValueError("sealed feature manifest claims truth fields are present")
        feature_file = manifest_path.with_name("features.npy")
        if _sha256(feature_file) != metadata.get("feature_file_sha256"):
            raise ValueError("sealed DermLIP feature matrix hash mismatch")
        matrix = np.load(feature_file, mmap_mode="r")
        paths = [str(p) for p in metadata["image_paths"]]
        if matrix.shape != (len(paths), int(metadata["feature_dim"])):
            raise ValueError("sealed feature rows do not match their manifest")
        offset = self._matrix.shape[0]
        self._matrix = np.concatenate(
            [np.asarray(self._matrix, dtype="float32"), np.asarray(matrix, dtype="float32")])
        added = 0
        for index, path in enumerate(paths):
            key = _path_key(path)
            if key in self._paths:
                raise ValueError(f"sealed feature path collides with development: {key}")
            self._paths[key] = offset + index
            added += 1
        self.sealed_feature_sha256 = str(metadata["feature_file_sha256"])
        return added

    def encode_image(self, image_path: str):
        import numpy as np

        index = self._paths.get(_path_key(image_path))
        if index is None:
            raise ValueError(f"image is absent from the frozen feature artifact: {image_path}")
        vector = np.asarray(self._matrix[index], dtype="float32")
        if vector.ndim != 1 or vector.size != 512 or not np.isfinite(vector).all():
            raise ValueError("invalid frozen DermLIP query feature")
        return vector


def _load_cases() -> tuple[Dict[str, list[Dict[str, str]]], str]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if (
        metadata.get("all_views_retained") is not True
        or int(metadata.get("test_records_loaded", -1)) != 0
    ):
        raise ValueError("development manifest violates the all-view/test boundary")
    grouped: Dict[str, list[Dict[str, str]]] = {}
    for raw in payload.get("records", []):
        # Deliberately project only non-truth input fields.  The manifest's label
        # and clinical metadata are never accessed by this runner.
        row = {
            "case_id": str(raw["case_id"]),
            "cluster_id": str(raw["cluster_id"]),
            "image_path": str(raw["image_path"]),
            "image_sha256": str(raw["image_sha256"]),
            "modality": str(raw["modality"]),
            "split": str(raw["split"]),
        }
        if row["split"] != "calibration":
            raise ValueError("development preflight encountered a non-calibration view")
        grouped.setdefault(row["cluster_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["case_id"])
    if len(grouped) != 160 or sum(map(len, grouped.values())) != 244:
        raise ValueError("development manifest must contain exactly 160 lesions / 244 views")
    return grouped, _sha256(MANIFEST)


def _exemplar_record_sha256(record: Mapping[str, Any]) -> str:
    """Bind a frozen exemplar label to identity and image content, not rank."""
    return _canonical_sha256({
        "schema": "medguard-exemplar-relation-record-v1",
        "document_id": record.get("document_id"),
        "citation_id": record.get("citation_id"),
        "source_id": record.get("source_id"),
        "source_version": record.get("source_version"),
        "image_sha256": record.get("image_sha256"),
        "canonical_label": record.get("label"),
        "split": record.get("split"),
    })


def _build_contract(
    *, case_id: str, perception: Any, records: Sequence[Mapping[str, Any]],
    guideline_items: Sequence[Any] = (),
    guideline_catalog: Any = None, guideline_registry: Any = None,
) -> tuple[EvidenceContract, list[Dict[str, str]]]:
    evidence = []
    record_by_evidence: Dict[str, Mapping[str, Any]] = {}
    for index, raw_record in enumerate(records, start=1):
        record = dict(raw_record)
        channel = str(record.get("retrieval_channel") or "")
        if channel == "image_exemplar":
            record["record_sha256"] = _exemplar_record_sha256(record)
            path = ("initial", "all_view_image_rrf", "cross_modal_rrf")
        else:
            path = ("initial", "dense_bm25_rrf", "cross_modal_rrf")
        item = evidence_from_chunk(record, evidence_id=f"e{index}", retrieval_path=path)
        evidence.append(item)
        record_by_evidence[item.evidence_id] = record
    # Authoritative-guideline tier: appended after the fused ceiling, not fused into it.
    evidence.extend(guideline_items)

    claims, relations, supporting_exemplars = build_image_bound_claims(
        perception=perception, evidence=evidence, record_by_evidence=record_by_evidence,
        guideline_catalog=guideline_catalog, guideline_registry=guideline_registry,
    )
    unresolved = []
    if not supporting_exemplars:
        unresolved.append(UnresolvedUncertainty(
            uncertainty_id=NO_EXEMPLAR_SUPPORT_UNCERTAINTY_ID,
            uncertainty_type=UncertaintyType.EVIDENCE_SUPPORT,
            description=NO_EXEMPLAR_SUPPORT_DESCRIPTION,
            related_claim_ids=(DIAGNOSIS_CLAIM_ID,),
        ))

    satisfied = not unresolved
    contract = EvidenceContract(
        case_id=case_id,
        modality="dermoscopy",
        clinical_question=QUESTION,
        perception=perception,
        claims=claims,
        evidence=evidence,
        unresolved_uncertainties=unresolved,
        verification_state=(
            VerificationState.SATISFIED if satisfied else VerificationState.FAILED_CLOSED
        ),
        proposed_action="PASS" if satisfied else "ABSTAIN",
        termination_reason=("" if satisfied else NO_EXEMPLAR_SUPPORT_TERMINATION),
    ).validate()
    return contract, relations


def run_preflight(output_path: str | Path = DEFAULT_OUTPUT) -> Dict[str, Any]:
    """Run the exact development preflight and write one canonical JSON report."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cases, manifest_sha256 = _load_cases()
    bindings = load_development_perception_bindings()
    expected_binding_ids = {f"lesion:{cluster_id}" for cluster_id in cases}
    if set(bindings) != expected_binding_ids:
        raise ValueError("truth-redacted perception cases do not match the development manifest")

    feature_encoder = FrozenDermLIPFeatureEncoder()
    runtime = load_frozen_evidence_runtime(image_encoder=feature_encoder)
    runtime_drift = runtime.receipt.get("_verification", {}).get(
        "exploratory_runtime_drift", [],
    )
    guideline_retriever = FrozenGuidelineRetriever.load(GUIDELINE_CORPUS)
    guideline_corpus_sha256 = _sha256(GUIDELINE_CORPUS)
    guideline_catalog, guideline_registry = build_guideline_claim_assets(
        guideline_retriever.records)
    text_cache: Dict[str, list[Dict[str, Any]]] = {}
    per_case = []
    relation_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    guideline_claim_type_counts: Counter[str] = Counter()
    lesions_with_guideline_claims = 0

    for ordinal, cluster_id in enumerate(sorted(cases), start=1):
        rows = cases[cluster_id]
        binding_id = f"lesion:{cluster_id}"
        perception = bindings[binding_id]
        diagnosis = perception.commitment.concept_code
        query = (
            f"dermoscopy {QUERY_TERMS[diagnosis]} diagnosis differential clinical evidence"
        )
        if query not in text_cache:
            retrieved, _scores = runtime.text_retriever.search(query, TEXT_CANDIDATE_K)
            admitted = []
            for record in retrieved:
                decision = runtime.source_policy.decide(record, "diagnosis")
                if decision.admitted:
                    admitted.append(dict(record))
            text_cache[query] = admitted
        text_records = [dict(record) for record in text_cache[query]]
        image_records = runtime.image_retriever.search_records_by_images(
            [row["image_path"] for row in rows], IMAGE_CANDIDATE_K,
        )
        if any(int(record.get("query_view_count", -1)) != len(rows) for record in image_records):
            raise ValueError(f"all-view image retrieval failed for {cluster_id}")
        fused = reciprocal_rank_fusion_records([text_records, image_records])[:FINAL_K]
        if len(fused) != FINAL_K:
            raise ValueError(f"cross-modal retrieval returned fewer than {FINAL_K}: {cluster_id}")
        # Authoritative-guideline tier, condition-scoped by the committed diagnosis
        # and claim-type-stratified so citations reflect the assessment question
        # rather than document_id alphabetical order.
        guideline_hits = select_guideline_evidence(guideline_retriever, diagnosis, GUIDELINE_K)
        guideline_items = [
            guideline_evidence_item(hit.to_evidence_record(), evidence_id=f"eg{offset}")
            for offset, hit in enumerate(guideline_hits, start=1)
        ]
        contract, relations = _build_contract(
            case_id=binding_id, perception=perception, records=fused,
            guideline_items=guideline_items,
            guideline_catalog=guideline_catalog, guideline_registry=guideline_registry,
        )
        # Guideline claims cite the authoritative-guideline channel; identify them
        # by that retrieval channel on their single cited evidence item.
        guideline_evidence_ids = {
            item.evidence_id for item in contract.evidence
            if item.retrieval_channel == GUIDELINE_CHANNEL
        }
        case_guideline_claims = [
            claim for claim in contract.claims
            if tuple(claim.evidence_ids) and set(claim.evidence_ids) <= guideline_evidence_ids
        ]
        if case_guideline_claims:
            lesions_with_guideline_claims += 1
        for claim in case_guideline_claims:
            guideline_claim_type_counts[claim.claim_type.value] += 1
        score = score_contract(contract)
        verification = verify_contract(contract)
        for relation in relations:
            relation_counts[relation["relation"]] += 1
        for failure in verification.failures:
            failure_counts[failure.value] += 1
        final_channels = Counter(str(record.get("retrieval_channel") or "") for record in fused)
        channel_counts.update(final_channels)
        view_ids = [row["case_id"] for row in rows]
        view_hashes = [row["image_sha256"] for row in rows]
        per_case.append({
            "case_id": binding_id,
            "cluster_id": cluster_id,
            "ordered_view_case_ids": view_ids,
            "view_count": len(rows),
            "view_lineage_sha256": _canonical_sha256(view_hashes),
            "committed_diagnosis": diagnosis,
            "commitment_confidence": perception.commitment.confidence,
            "retrieval": {
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "text_candidates_admitted": len(text_records),
                "image_candidates": len(image_records),
                "final_evidence": len(fused),
                "final_channels": dict(sorted(final_channels.items())),
                "all_views_preserved": True,
                "evidence": [{
                    "evidence_id": item.evidence_id,
                    "document_id": item.document_id,
                    "source_id": item.source_id,
                    "source_version": item.source_version,
                    "authority": item.authority,
                    "retrieval_channel": item.retrieval_channel,
                    "content_sha256": item.content_sha256,
                    "record_sha256": item.record_sha256,
                } for item in contract.evidence],
            },
            "relations": relations,
            "contract_sha256": contract_sha256(contract),
            "score": score.to_dict(),
            "verification": {
                "ok": verification.ok,
                "failures": [failure.value for failure in verification.failures],
            },
            "final_action": contract.proposed_action,
            "generation_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        })
        if ordinal % 40 == 0:
            print(f"preflight {ordinal}/160")

    score_rows = [row["score"] for row in per_case]
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "medguard-derm-evidence-contract-h2h-v1",
        "status": "exploratory_development_preflight_not_paper_result",
        "boundaries": {
            "development_only": True,
            "sealed_test_access": False,
            "test_records_loaded": 0,
            "truth_fields_consumed": False,
            "generation_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "paid_calls": 0,
        },
        "inputs": {
            "development_manifest_sha256": manifest_sha256,
            "lesions": len(cases),
            "images": sum(map(len, cases.values())),
            "perception_artifact_sha256": next(iter(bindings.values())).artifact_sha256,
            "evidence_receipt_sha256": _sha256(
                ROOT / "research/evidence/dermatology_corpus_v1/receipt.json"
            ),
            "query_feature_sha256": feature_encoder.feature_sha256,
            "query_feature_model": feature_encoder.model_id,
        },
        "retrieval": {
            "text": "all-mpnet dense + frozen BM25 RRF",
            "image": "DermLIP train-only atlas, RRF across every ordered view",
            "cross_modal": "RRF with a shared five-item ceiling",
            "text_candidate_k": TEXT_CANDIDATE_K,
            "image_candidate_k": IMAGE_CANDIDATE_K,
            "final_k": FINAL_K,
            "guideline": {
                "channel": GUIDELINE_CHANNEL,
                "corpus_sha256": guideline_corpus_sha256,
                "records": len(guideline_retriever.records),
                "guideline_k": GUIDELINE_K,
                "note": (
                    "Separate authoritative tier appended after the fused ceiling; "
                    "condition-scoped by the committed diagnosis, so benign commitments "
                    "reach zero. Claims are source-statement integrity, not diagnosis proof."
                ),
            },
            "distinct_text_queries": len(text_cache),
            "runtime_loaded": True,
            "bundle_verified": True,
            "exploratory_runtime_drift_count": len(runtime_drift),
            "runtime_lock_required_before_paid_h2h": True,
        },
        "aggregate": {
            "cases": len(per_case),
            "views": sum(row["view_count"] for row in per_case),
            "valid_contracts": sum(bool(score["valid"]) for score in score_rows),
            "contract_complete": sum(bool(score["contract_complete"]) for score in score_rows),
            "structured_pass_eligible": sum(
                bool(score["structured_pass_eligible"]) for score in score_rows
            ),
            "abstained": sum(row["final_action"] == "ABSTAIN" for row in per_case),
            "mean_structural_completion": sum(
                float(score["structural_completion"]) for score in score_rows
            ) / len(score_rows),
            "mean_image_support_rate": sum(
                float(score["image_support_rate"])
                for score in score_rows if score["image_support_rate"] is not None
            ) / sum(score["image_support_rate"] is not None for score in score_rows),
            "mean_independent_relation_rate": sum(
                float(score["independent_evidence_relation_rate"])
                for score in score_rows
                if score["independent_evidence_relation_rate"] is not None
            ) / sum(
                score["independent_evidence_relation_rate"] is not None
                for score in score_rows
            ),
            "relation_counts": dict(sorted(relation_counts.items())),
            "verification_failure_case_counts": dict(sorted(failure_counts.items())),
            "final_evidence_channel_counts": dict(sorted(channel_counts.items())),
            "lesions_with_guideline_claims": lesions_with_guideline_claims,
            "guideline_claims_total": sum(guideline_claim_type_counts.values()),
            "guideline_claim_type_counts": dict(sorted(guideline_claim_type_counts.items())),
        },
        "interpretation": {
            "verified": (
                "The offline C1 contract and C2 retrieval paths execute for every frozen "
                "development lesion while preserving all views and recording zero generation."
            ),
            "not_claimed": (
                "Retrieval relevance, provenance, authority, and citations do not prove "
                "semantic support. Text relations remain insufficient without an exact "
                "curated or human-reviewed relation basis."
            ),
            "next_gate": (
                "Add independently curated claim-evidence relations and the matched arms "
                "before any paid development H2H; lock runtime hashes only at that gate."
            ),
        },
        "per_case": per_case,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8"))
    return report


def main() -> int:
    report = run_preflight()
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    print(f"wrote {DEFAULT_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
