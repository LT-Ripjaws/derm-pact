"""Retrieval operations used by the deterministic remediation loop."""
from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from ..evidence import EvidenceContract, PerceptionBinding
from .hybrid_retriever import reciprocal_rank_fusion_records


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[str(item)] = scores.get(str(item), 0.0) + 1.0 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]


class EvidencePipelineOps:
    """Concrete tools for initial retrieval and bounded contract remediation.

    `independent_retriever` must be a separately configured source/index. Falling
    back to the original retriever would falsely claim source independence.
    """

    def __init__(
        self, *, retriever: Any, generator: Any, formulator: Any = None,
        reranker: Any = None, image_retriever: Any = None,
        independent_retriever: Any = None, top_k: int = 5, expand_k: int = 12,
        case_inputs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.formulator = formulator
        self.reranker = reranker
        self.image_retriever = image_retriever
        self.independent_retriever = independent_retriever
        self.top_k = int(top_k)
        self.expand_k = int(expand_k)
        self.case_inputs = {str(key): dict(value) for key, value in (case_inputs or {}).items()}
        if self.top_k <= 0 or self.expand_k < self.top_k:
            raise ValueError("require 0 < top_k <= expand_k")

    def _search(self, retriever: Any, query: str, k: int, path: Sequence[str]):
        chunks, scores = retriever.search(query, k)
        values = []
        for chunk, score in zip(chunks, scores):
            value = dict(chunk)
            value.setdefault("score", float(score))
            value["retrieval_path"] = list(path)
            values.append(value)
        if self.reranker is not None and values:
            values = list(self.reranker.rerank(query, values, k))
        return values[:k]

    def _inputs(self, contract: EvidenceContract) -> MutableMapping[str, Any]:
        return dict(self.case_inputs.get(contract.case_id, {}))

    def _retrieve(self, query: str, k: int, path: Sequence[str], *, image_input: Any = None):
        """Retrieve one fixed-size bundle and preserve each modality's lineage."""
        text = self._search(self.retriever, query, k, path)
        if self.image_retriever is None:
            return text
        if not image_input:
            raise ValueError("multimodal retrieval is configured but the case has no image path")
        is_multi_view = isinstance(image_input, (list, tuple))
        method = "search_records_by_images" if is_multi_view else "search_records_by_image"
        search_records = getattr(self.image_retriever, method, None)
        if not callable(search_records):
            raise TypeError(f"image retriever must expose lineage-preserving {method}")
        image = []
        query_images = [str(value) for value in image_input] if is_multi_view else str(image_input)
        for record in search_records(query_images, k):
            row = dict(record)
            image_path_parts = list(path[:-1]) if path and path[-1] == "dense" else list(path)
            row["retrieval_path"] = [*image_path_parts, "image_exemplar"]
            image.append(row)
        fused = reciprocal_rank_fusion_records([text, image])
        return fused[:k]

    def _regenerate(self, contract: EvidenceContract, chunks, *, question: Optional[str] = None):
        inputs = self._inputs(contract)
        return self.generator.generate_contract(
            case_id=contract.case_id,
            modality=contract.modality,
            question=question or contract.clinical_question,
            perception=contract.perception,
            chunks=chunks,
            image_path=inputs.get("image_paths") or inputs.get("image_path"),
        )

    def initial_contract(self, case: Mapping[str, Any]) -> EvidenceContract:
        case_id = str(case.get("case_id", ""))
        self.case_inputs[case_id] = dict(case)
        question = str(case.get("question", ""))
        perception = case.get("perception")
        if not isinstance(perception, PerceptionBinding):
            raise ValueError("case requires a typed host-owned perception binding")
        perception.validate()
        image_input = case.get("image_paths") or case.get("image_path")
        chunks = self._retrieve(
            question, self.top_k, ("initial", "dense"), image_input=image_input)
        return self.generator.generate_contract(
            case_id=case_id, modality=str(case.get("modality", "dermoscopy")),
            question=question, perception=perception, chunks=chunks,
            image_path=image_input,
        )

    def expand(self, contract: EvidenceContract) -> EvidenceContract:
        chunks = self._retrieve(
            contract.clinical_question, self.expand_k, ("remediation", "expand"),
            image_input=(self._inputs(contract).get("image_paths")
                         or self._inputs(contract).get("image_path")))
        return self._regenerate(contract, chunks)

    def reformulate(self, contract: EvidenceContract) -> EvidenceContract:
        query = contract.clinical_question
        if self.formulator is not None:
            result = self.formulator.formulate(
                image_path=self._inputs(contract).get("image_path"),
                question=contract.clinical_question,
                hint="retrieve evidence for unresolved or insufficient claims",
            )
            query = str(result.get("selected_query") or query)
        chunks = self._retrieve(
            query, self.top_k, ("remediation", "reformulate"),
            image_input=(self._inputs(contract).get("image_paths")
                         or self._inputs(contract).get("image_path")))
        return self._regenerate(contract, chunks, question=query)

    def independent_source(self, contract: EvidenceContract) -> EvidenceContract:
        if self.independent_retriever is None:
            return contract
        chunks = self._search(
            self.independent_retriever, contract.clinical_question, self.top_k,
            ("remediation", "independent_source"),
        )
        return self._regenerate(contract, chunks)


# Back-compatible import name for build scripts; semantics are the new evidence operations.
MedGuardPipelineOps = EvidencePipelineOps
