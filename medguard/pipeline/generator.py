"""Evidence-contract generator with host-owned perception/provenance binding.

The model proposes claims and relation labels.  It cannot set or replace the
perception record, evidence provenance, relation-assessor origin, output
obligations, or hashes.  Generator relation labels remain declarations until
checked by a host scorer.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ..evidence import (
    ClinicalClaim,
    ClaimType,
    EntailmentState,
    EvidenceContract,
    FindingState,
    ImageSupport,
    PerceptionBinding,
    RelationOrigin,
    UncertaintyType,
    UnresolvedUncertainty,
    VerificationState,
    evidence_from_chunk,
)
from ..errors import StructuredOutputError
from .llm import LLMClient


SYSTEM = (
    "You are a careful medical-imaging evidence analyst. Use only the patient image "
    "and numbered evidence. Describe uncertainty and contradictions explicitly."
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.I | re.S)
    if fenced:
        raw = fenced.group(1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"generator returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise StructuredOutputError("generator output must be one JSON object")
    return value


def _evidence_items(chunks: Sequence[Dict[str, Any]], max_chunks: int = 6):
    items = []
    for index, chunk in enumerate(chunks[:max_chunks], start=1):
        path = chunk.get("retrieval_path") or ("retrieval", str(chunk.get("retriever", "dense")))
        if isinstance(path, str):
            path = (path,)
        try:
            items.append(evidence_from_chunk(chunk, evidence_id=str(index), retrieval_path=path))
        except ValueError as exc:
            raise StructuredOutputError(f"retrieved evidence {index} rejected: {exc}") from exc
    return items


def _evidence_block(items) -> str:
    return "\n".join(
        f"[{item.evidence_id}] ({item.source_id}; {item.authority}; {item.source_version}) {item.text}"
        for item in items
    ) or "(no evidence retrieved)"


def _normalize_concept(raw: Any) -> str:
    """Mechanical case/whitespace canonicalization only; never invents content."""
    text = str(raw or "").strip().lower()
    return re.sub(r"\s+", "_", text)


def bind_evidence_contract(
    value: Dict[str, Any], *, case_id: str, modality: str, question: str,
    perception: PerceptionBinding, chunks: Sequence[Dict[str, Any]],
) -> EvidenceContract:
    """Bind model claims to immutable host perception and retrieved evidence."""
    try:
        if not isinstance(perception, PerceptionBinding):
            raise ValueError("host must provide a typed perception binding")
        perception.validate()
        expected_output = {"claims", "unresolved_uncertainties"}
        if set(value) != expected_output:
            raise ValueError(
                f"generator fields mismatch; missing={sorted(expected_output - set(value))}, "
                f"unknown={sorted(set(value) - expected_output)}")
        raw_claims = value.get("claims", [])
        if not isinstance(raw_claims, list):
            raise ValueError("claims must be a list")
        # The generator may cite an evidence/finding id absent from the supplied set.
        # That single droppable id must NOT reject the whole contract (that rejection
        # was 117 of 187 development gen-errors). The valid id sets are host-owned:
        # evidence ids are "1".."N" over the (<=6) supplied chunks, finding ids come
        # from the frozen perception binding.
        n_supplied_evidence = len(list(chunks)[:6])
        valid_evidence_ids = {str(index) for index in range(1, n_supplied_evidence + 1)}
        valid_finding_ids = {finding.finding_id for finding in perception.findings}
        image_bound_types = {
            ClaimType.DIAGNOSIS, ClaimType.DIFFERENTIAL, ClaimType.VISUAL_FINDING,
        }
        claims: List[ClinicalClaim] = []
        for raw in raw_claims:
            if not isinstance(raw, dict):
                raise ValueError("each claim must be an object")
            expected_claim = {
                "claim_id", "claim_type", "text", "image_support", "target_concept",
                "target_state", "image_finding_ids", "evidence_ids", "entailment",
                "contradiction",
            }
            if set(raw) != expected_claim:
                raise ValueError(
                    f"claim fields mismatch; missing={sorted(expected_claim - set(raw))}, "
                    f"unknown={sorted(set(raw) - expected_claim)}")
            evidence_ids = raw.get("evidence_ids", [])
            image_finding_ids = raw.get("image_finding_ids", [])
            if not isinstance(evidence_ids, list) or not isinstance(image_finding_ids, list):
                raise ValueError("claim evidence_ids and image_finding_ids must be lists")
            if "claim_type" not in raw:
                raise ValueError("each claim must declare claim_type")
            # Drop only the ids the host never supplied; keep declaration order, dedupe.
            kept_evidence = [
                eid for eid in dict.fromkeys(str(item) for item in evidence_ids)
                if eid in valid_evidence_ids
            ]
            kept_findings = [
                fid for fid in dict.fromkeys(str(item) for item in image_finding_ids)
                if fid in valid_finding_ids
            ]
            try:
                claim_type = ClaimType(str(raw["claim_type"]))
                # Keep the claim only if it still stands on its own: an image-bound
                # claim needs a host finding; any other claim needs >=1 valid evidence
                # id. Which of target_concept / image_finding_ids / image_support a
                # claim type may carry is fixed by the schema, not the model's choice,
                # so it is host-coerced here rather than trusted -- the model routinely
                # gets the applicability wrong (an exemplar claim citing a patient
                # finding, an assessment claim declaring an image_support) even when
                # every id and enum literal it used is individually valid.
                # target_concept gets a mechanical case/whitespace normalization only
                # (never invented content); if it still isn't this type's canonical
                # concept, claim.validate() below drops the claim.
                concept = _normalize_concept(raw.get("target_concept"))
                declared_support = str(raw.get("image_support") or "uncertain")
                if claim_type in image_bound_types:
                    if not kept_findings:
                        continue
                    target_concept = concept
                    image_support_value = (
                        "uncertain" if declared_support == "not_applicable"
                        else declared_support
                    )
                elif claim_type is ClaimType.IMAGE_EXEMPLAR_SUPPORT:
                    if not kept_evidence:
                        continue
                    target_concept = concept
                    kept_findings = []  # exemplar support never cites a patient finding
                    image_support_value = "not_applicable"
                else:
                    if not kept_evidence:
                        continue
                    target_concept = ""  # source-bound types are never image-bound
                    kept_findings = []
                    image_support_value = "not_applicable"
                entailment = EntailmentState(str(raw.get("entailment") or "not_assessed"))
                if entailment is EntailmentState.SUPPORTED and not kept_evidence:
                    # its only support was an unknown id -> it is no longer "supported".
                    entailment = EntailmentState.NOT_ASSESSED
                claim = ClinicalClaim(
                    claim_id=str(raw.get("claim_id", "")),
                    claim_type=claim_type,
                    text=str(raw.get("text", "")),
                    image_support=ImageSupport(image_support_value),
                    target_concept=target_concept,
                    target_state=FindingState(str(raw.get("target_state") or "uncertain")),
                    image_finding_ids=tuple(kept_findings),
                    evidence_ids=tuple(kept_evidence),
                    entailment=entailment,
                    contradiction=bool(raw.get("contradiction", False)),
                    evidence_relation_origin=(
                        RelationOrigin.NOT_ASSESSED
                        if entailment is EntailmentState.NOT_ASSESSED
                        else RelationOrigin.GENERATOR_DECLARED
                    ),
                    evidence_relation_assessor_id=(
                        "" if entailment is EntailmentState.NOT_ASSESSED
                        else "generator_declared"),
                )
                claim.validate()
            except ValueError:
                # An unparseable claim_type/enum literal, or a target_concept that
                # still isn't this type's canonical concept after normalization: the
                # host cannot safely guess the intended content, so drop the one claim
                # rather than reject the whole contract over it (same rule as a
                # dropped id).
                continue
            claims.append(claim)
        uncertainties = value.get("unresolved_uncertainties", [])
        if not isinstance(uncertainties, list):
            raise ValueError("unresolved_uncertainties must be a list")
        kept_claim_ids = {claim.claim_id for claim in claims}
        bound_uncertainties = []
        for raw in uncertainties:
            if not isinstance(raw, dict):
                raise ValueError("each uncertainty must be an object")
            expected_uncertainty = {
                "uncertainty_id", "uncertainty_type", "description", "related_claim_ids",
            }
            if set(raw) != expected_uncertainty:
                raise ValueError(
                    "uncertainty fields mismatch; "
                    f"missing={sorted(expected_uncertainty - set(raw))}, "
                    f"unknown={sorted(set(raw) - expected_uncertainty)}")
            related = raw.get("related_claim_ids", [])
            if not isinstance(related, list):
                raise ValueError("uncertainty related_claim_ids must be a list")
            # A related claim id may point at a claim the id-degradation just dropped;
            # keep only live references, and drop an uncertainty left with none (it now
            # binds nothing) rather than reject the contract over a dangling reference.
            kept_related = [
                cid for cid in dict.fromkeys(str(item) for item in related)
                if cid in kept_claim_ids
            ]
            if not kept_related:
                continue
            bound_uncertainties.append(UnresolvedUncertainty(
                uncertainty_id=str(raw.get("uncertainty_id", "")),
                uncertainty_type=UncertaintyType(str(raw.get("uncertainty_type", "other"))),
                description=str(raw.get("description", "")),
                related_claim_ids=tuple(kept_related),
            ))
        return EvidenceContract(
            case_id=str(case_id), modality=str(modality), clinical_question=str(question),
            perception=perception, claims=claims, evidence=_evidence_items(chunks),
            unresolved_uncertainties=bound_uncertainties,
            verification_state=VerificationState.PENDING,
        ).validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise StructuredOutputError(f"evidence contract rejected: {exc}") from exc


class VLMGenerator:
    """Generate a typed evidence contract over one fixed host-owned evidence state."""

    def __init__(self, client: Optional[LLMClient] = None, *, max_tokens: int = 768,
                 max_schema_repairs: int = 1, seed: int = 0):
        if max_schema_repairs not in (0, 1):
            raise ValueError("max_schema_repairs must be 0 or 1")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.client = client or LLMClient()
        self.max_tokens = int(max_tokens)
        self.max_schema_repairs = int(max_schema_repairs)
        self.seed = seed
        self.last_metadata: Dict[str, Any] = {}

    def _capture_usage(self, start_index: int, repairs: int, seed: int) -> None:
        events = list(getattr(self.client, "call_log", []))[start_index:]
        successful = [event for event in events if event.get("succeeded")]
        self.last_metadata = {
            "remote_calls": len(events),
            "failed_calls": sum(not event.get("succeeded", False) for event in events),
            "prompt_tokens": sum(int(event.get("prompt_tokens") or 0) for event in events),
            "completion_tokens": sum(int(event.get("completion_tokens") or 0) for event in events),
            "total_tokens": sum(int(event.get("total_tokens") or 0) for event in events),
            "latency_seconds": sum(float(event.get("latency_seconds") or 0.0) for event in events),
            "usage_available": bool(successful) and all(
                event.get("usage_available") for event in successful),
            "schema_repairs": repairs,
            "seed": seed,
        }

    def generate_contract(
        self, *, case_id: str, modality: str, question: str,
        perception: PerceptionBinding, chunks: Sequence[Dict[str, Any]],
        image_path: Optional[Any] = None, prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> EvidenceContract:
        perception.validate()
        items = _evidence_items(chunks)
        request_seed = self.seed if seed is None else seed
        if (isinstance(request_seed, bool) or not isinstance(request_seed, int)
                or request_seed < 0):
            raise ValueError("seed must be a non-negative integer")
        if prompt is None:
            host_findings = [{
                "finding_id": item.finding_id,
                "finding_type": item.finding_type.value,
                "concept_code": item.concept_code,
                "state": item.state.value,
            } for item in perception.findings]
            prompt = f"""Case: {case_id}
Modality: {modality}
Question: {question}
Host image findings: {json.dumps(host_findings, ensure_ascii=False)}
Host committed diagnosis: {perception.commitment.concept_code}

Evidence:
{_evidence_block(items)}

Return one JSON object only:
{{
  "claims": [{{
    "claim_id": "c1",
    "claim_type": "assessment|diagnosis|differential|visual_finding|management|referral|treatment|image_exemplar_support",
    "text": "one atomic clinical claim",
    "image_support": "supports|contradicts|uncertain|not_applicable",
    "target_concept": "closed diagnosis/finding code, or empty for non-image claims",
    "target_state": "present|absent|uncertain",
    "image_finding_ids": ["host finding id, or empty for non-image claims"],
    "evidence_ids": ["1"],
    "entailment": "supported|contradicted|insufficient|not_assessed",
    "contradiction": false
  }}],
  "unresolved_uncertainties": [{{
    "uncertainty_id": "u1",
    "uncertainty_type": "image_support|evidence_support|diagnosis|source_conflict|other",
    "description": "one unresolved uncertainty",
    "related_claim_ids": ["c1"]
  }}]
}}
Use only listed evidence and finding IDs. Do not invent provenance. Relation labels
are declarations and will be checked independently by the host scorer."""
        else:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a non-empty string when supplied")
        start = len(getattr(self.client, "call_log", []))
        repairs = 0
        try:
            raw = self.client.vision(
                prompt, image_path, system=SYSTEM, max_tokens=self.max_tokens,
                seed=request_seed,
            )
            try:
                return bind_evidence_contract(
                    _extract_json_object(raw), case_id=case_id, modality=modality,
                    question=question, perception=perception, chunks=chunks,
                )
            except StructuredOutputError as exc:
                if not self.max_schema_repairs:
                    raise
                repairs = 1
                valid_evidence_ids = [str(i) for i in range(1, len(list(chunks)[:6]) + 1)]
                valid_finding_ids = [f.finding_id for f in perception.findings]
                repair_prompt = (
                    f"{prompt}\n\nThe host rejected your previous JSON: {str(exc)[:400]}. "
                    f"The ONLY valid evidence_ids are {valid_evidence_ids}; the ONLY valid "
                    f"image_finding_ids are {valid_finding_ids}. Use only those; drop any "
                    "other id. Return a corrected JSON object only."
                )
                fixed = self.client.vision(
                    repair_prompt, image_path, system=SYSTEM, max_tokens=self.max_tokens,
                    seed=request_seed,
                )
                return bind_evidence_contract(
                    _extract_json_object(fixed), case_id=case_id, modality=modality,
                    question=question, perception=perception, chunks=chunks,
                )
        finally:
            self._capture_usage(start, repairs, request_seed)

    def generate(self, **kwargs: Any) -> str:
        return render_evidence_contract(self.generate_contract(**kwargs))


def render_evidence_contract(contract: EvidenceContract) -> str:
    contract.validate()
    if contract.proposed_action not in {"PASS", "ABSTAIN"}:
        raise StructuredOutputError("cannot render a non-terminal evidence contract")
    if contract.proposed_action == "PASS":
        from ..evidence import verify_contract

        report = verify_contract(contract)
        if not report.ok:
            raise StructuredOutputError(
                "cannot render an unverified PASS contract: " + "; ".join(report.issues))
    def rendered_claim(claim: ClinicalClaim) -> str:
        if claim.claim_type in {
            ClaimType.DIAGNOSIS, ClaimType.DIFFERENTIAL, ClaimType.VISUAL_FINDING,
        }:
            label = claim.claim_type.value.replace("_", " ").title()
            return f"{label}: {claim.target_concept} ({claim.target_state.value})."
        return claim.text

    claims = " ".join(rendered_claim(claim) for claim in contract.claims) or "No supported claim."
    uncertainty = "; ".join(
        item.description for item in contract.unresolved_uncertainties)
    suffix = f"\nUncertainty: {uncertainty}" if uncertainty else ""
    return f"Decision: {contract.committed_diagnosis}\n{claims}{suffix}".strip()
