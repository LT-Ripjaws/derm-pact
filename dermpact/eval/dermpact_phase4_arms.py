"""Phase 4.6: the five comparison arms for guideline applicability."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import httpx

from dermpact.guidelines.engine import ApplicabilityEngine, EntityMismatchError
from dermpact.guidelines.facts import CaseFacts, TV
from dermpact.guidelines.rules import ApplicabilityState as S, RuleCatalogue

STATES = tuple(s.value for s in S)


@dataclass
class ArmOutput:
    arm: str
    run_id: str
    refused: bool
    per_rule: Dict[str, Dict[str, Any]]      # rule_id -> {state, action, missing, source_ids}
    released_actions: List[Tuple[str, str]]  # (rule_id, action_code) the arm would DISPLAY
    question: Optional[str]
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    raw: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {**self.__dict__, "released_actions": [list(x) for x in self.released_actions]}




class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> Tuple[str, int, int]:
        """Return (text, prompt_tokens, completion_tokens)."""


class FakeLLMClient:
    """Deterministic plumbing stub. Its outputs carry no scientific meaning."""

    name = "fake-llm-plumbing-stub"

    def __init__(self, policy: str = "always_applicable"):
        self.policy = policy
        self.calls = 0

    def complete(self, system: str, user: str) -> Tuple[str, int, int]:
        self.calls += 1
        if "get_fact" in system and '"final"' in system:
            # traversal protocol: ask one fact, then finalise
            if "FACT VALUES SO FAR" in user and user.count("=") >= 1:
                return json.dumps({"final": {"state": "applicable", "action": "histopathological_confirmation", "missing": None}}), 50, 20
            return json.dumps({"call": "get_fact", "name": "lesion.melanoma_suspected"}), 50, 10
        return json.dumps({r: {"state": "applicable", "action": "histopathological_confirmation", "missing": None}
                           for r in re.findall(r"eado-mel-p1-2024:R\d+", user)}), 80, 30


class LiveGatewayClient:
    """OpenAI-compatible gateway client using dermpact.config (key never exposed)."""

    def __init__(self, model: Optional[str] = None, *, temperature: float = 0.0, max_tokens: int = 500):
        import os
        from dermpact import config as cfg  # code-only config; reads env, never prints
        cfg.load_env()
        if os.environ.get("DERMPACT_ALLOW_LIVE_LLM") != "1":
            raise RuntimeError("live calls disabled; set DERMPACT_ALLOW_LIVE_LLM=1 explicitly")
        if not cfg.has_api_key():
            raise RuntimeError("no gateway API key in environment; use --llm-mode fake or set DERMPACT_API_KEY")
        self._cfg = cfg
        self._base_url = cfg.GATEWAY_BASE_URL.rstrip("/")
        self.model = model or cfg.GENERATOR_MODEL
        self.name = f"gateway:{self.model}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = cfg.API_MAX_RETRIES

    def complete(self, system: str, user: str) -> Tuple[str, int, int]:
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                self._cfg.load_env()
                headers = {"Authorization": f"Bearer {self._cfg.GATEWAY_API_KEY}", "Content-Type": "application/json"}
                body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(f"{self._base_url}/chat/completions", json=body, headers=headers)
                response.raise_for_status()
                payload = response.json()
                text = str(payload["choices"][0]["message"].get("content") or "")
                usage = payload.get("usage") or {}
                return text, int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
            except Exception as exc:  # pragma: no cover - network
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"gateway call failed after {self.retries} attempts: {type(last).__name__}")




_TOKEN = re.compile(r"[a-z0-9]+")


class LexicalChunkIndex:
    """Hashed bag-of-words cosine index over guideline page text (naive RAG baseline)."""

    def __init__(self, pages: Sequence[Mapping[str, Any]], *, chunk_chars: int = 700, overlap: int = 120, dim: int = 4096):
        self.dim = dim
        self.chunks: List[Dict[str, Any]] = []
        for page in pages:
            text = page["text"]
            start = 0
            while start < len(text):
                end = min(len(text), start + chunk_chars)
                self.chunks.append({"chunk_id": f"p{page['pdf_page']}:{start}", "pdf_page": page["pdf_page"], "text": text[start:end]})
                if end == len(text):
                    break
                start = end - overlap
        self.vectors = np.stack([self._vec(c["text"]) for c in self.chunks]) if self.chunks else np.zeros((0, dim))

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        q = self._vec(query)
        sims = self.vectors @ q
        order = sorted(range(len(self.chunks)), key=lambda i: (-float(sims[i]), i))[:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in order]




def _facts_for_prompt(facts: CaseFacts) -> str:
    lines = []
    for name, value in sorted(facts.values.items()):
        v = value.value if isinstance(value, TV) else value
        if isinstance(v, tuple):
            v = list(v)
        lines.append(f"- {name} = {json.dumps(v)}")
    return "\n".join(lines) if lines else "- (no facts)"


def _catalogue_summary(catalogue: RuleCatalogue, rule_ids: Sequence[str]) -> str:
    out = []
    for rid in rule_ids:
        r = catalogue.get(rid)
        pre = r.prerequisites.describe() if r.prerequisites else "(none)"
        act = r.action.code if r.action else "(none)"
        out.append(f"{rid} [{r.status.value}] title={r.title!r} prerequisites={pre} action={act} "
                   f"required_facts={list(r.required_facts)} valid_until={r.supersession.valid_until} jurisdiction={r.jurisdiction}")
    return "\n".join(out)


def _extract_json(text: str) -> Optional[Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalise_llm_rule_output(raw: Any, rule_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    per: Dict[str, Dict[str, Any]] = {}
    for rid in rule_ids:
        entry = raw.get(rid) if isinstance(raw, Mapping) else None
        state = None
        action = None
        missing = None
        if isinstance(entry, Mapping):
            s = str(entry.get("state", "")).strip().lower().replace(" ", "_")
            aliases = {"unknown": "applicability_unknown", "not_applicable": "not_applicable", "na": "not_applicable",
                       "no_coverage": "no_catalogue_coverage", "none": "no_catalogue_coverage"}
            s = aliases.get(s, s)
            state = s if s in STATES else None
            action = entry.get("action") or None
            missing = entry.get("missing") or None
        per[rid] = {"state": state or "malformed", "action": action if state == "applicable" else None,
                    "missing": missing if state == "applicability_unknown" else None, "source_ids": []}
    return per


SYSTEM_DIRECT = (
    "You are a clinical decision-support component. Given locked case facts (three-valued: true/false/unknown) "
    "and a list of EADO melanoma Part 1 (2024) recommendation identifiers, decide for EACH rule one of: "
    "applicable, not_applicable, applicability_unknown, no_catalogue_coverage. If applicable, name the recommended "
    "next-step action code; if unknown, name the single decisive missing fact. Reply with ONLY a JSON object keyed "
    "by rule id: {\"<rule_id>\": {\"state\": ..., \"action\": ..., \"missing\": ...}}."
)
SYSTEM_TRAVERSAL = (
    "You are walking a decision tree for ONE guideline rule. You may request facts one at a time by replying "
    "{\"call\": \"get_fact\", \"name\": \"<fact name>\"}. When you can decide, reply "
    "{\"final\": {\"state\": ..., \"action\": ..., \"missing\": ...}} with state in "
    "[applicable, not_applicable, applicability_unknown, no_catalogue_coverage]. Reply with ONLY JSON."
)




class Arms:
    def __init__(self, catalogue: RuleCatalogue, *, source_pages: Sequence[Mapping[str, Any]],
                 llm: Optional[LLMClient] = None, rag_k: int = 3):
        self.catalogue = catalogue
        self.engine = ApplicabilityEngine(catalogue)
        self.index = LexicalChunkIndex(source_pages)
        self.llm = llm
        self.rag_k = rag_k

    # -- arm 1 ------------------------------------------------------------
    def deterministic_rules(self, run: Mapping[str, Any]) -> ArmOutput:
        facts = CaseFacts.from_dict(run["facts"])
        t0 = time.perf_counter()
        try:
            case = self.engine.evaluate_case(facts, expected_case_id=run["expected_case_id"], rule_ids=run["query_rule_ids"])
        except EntityMismatchError:
            return ArmOutput("deterministic_rules", run["run_id"], True, {}, [], None, latency_s=time.perf_counter() - t0)
        per = {r.rule_id: {"state": r.state.value, "action": r.action_code if r.state is S.APPLICABLE else None,
                           "missing": r.decisive_missing_fact, "source_ids": [r.source["document_id"]]} for r in case.results}
        return ArmOutput("deterministic_rules", run["run_id"], False, per, list(case.supported_next_steps),
                         case.targeted_question[2] if case.targeted_question else None, latency_s=time.perf_counter() - t0)

    # -- arm 2 ------------------------------------------------------------
    def rules_plus_retrieval(self, run: Mapping[str, Any]) -> ArmOutput:
        out = self.deterministic_rules(run)
        out.arm = "rules_plus_retrieval"
        if out.refused:
            return out
        for rid, entry in out.per_rule.items():
            rule = self.catalogue.get(rid)
            if entry["state"] in ("applicable", "applicability_unknown"):
                # retrieval restricted to the rule's own page: it ranks explanatory passages only
                hits = self.index.retrieve(rule.title + " " + (rule.action.text if rule.action else ""), k=self.rag_k)
                entry["source_ids"] = [f"{rule.source.document_id}#p{rule.source.pdf_page}"] + [h["chunk_id"] for h in hits]
        return out

    # -- arm 3 ------------------------------------------------------------
    def direct_llm(self, run: Mapping[str, Any]) -> ArmOutput:
        return self._llm_arm("direct_llm", run, with_context=False)

    # -- arm 4 ------------------------------------------------------------
    def naive_rag(self, run: Mapping[str, Any]) -> ArmOutput:
        return self._llm_arm("naive_rag", run, with_context=True)

    def _llm_arm(self, name: str, run: Mapping[str, Any], *, with_context: bool) -> ArmOutput:
        if self.llm is None:
            raise RuntimeError("LLM arms require an LLM client")
        facts = CaseFacts.from_dict(run["facts"])
        rule_ids = list(run["query_rule_ids"])
        user = (f"CASE ID: {facts.case_id} (request bound to case {run['expected_case_id']})\n"
                f"JURISDICTION: {facts.jurisdiction}\nEVALUATION DATE: {facts.evaluation_date.isoformat()}\n"
                f"FACTS:\n{_facts_for_prompt(facts)}\n\nRULE IDS: {rule_ids}\n")
        context_ids: List[str] = []
        if with_context:
            hits = self.index.retrieve("melanoma diagnosis dermatoscopy histopathology confirmation equivocal reflectance confocal high-risk", k=self.rag_k)
            context_ids = [h["chunk_id"] for h in hits]
            user += "\nRETRIEVED GUIDELINE PASSAGES:\n" + "\n---\n".join(h["text"] for h in hits) + "\n"
        else:
            user += "\n(No guideline text is provided; answer from the rule ids and facts.)\n"
        t0 = time.perf_counter()
        text, pt, ct = self.llm.complete(SYSTEM_DIRECT, user)
        raw = _extract_json(text)
        per = _normalise_llm_rule_output(raw or {}, rule_ids)
        for entry in per.values():
            entry["source_ids"] = context_ids
        released = [(rid, e["action"]) for rid, e in per.items() if e["state"] == "applicable" and e["action"]]
        refused = facts.case_id != run["expected_case_id"] and raw is None
        return ArmOutput(name, run["run_id"], refused, per, released, None, n_llm_calls=1, prompt_tokens=pt,
                         completion_tokens=ct, latency_s=time.perf_counter() - t0, raw=text[:2000])

    # -- arm 5 ------------------------------------------------------------
    def llm_traversal(self, run: Mapping[str, Any], *, max_steps: int = 6) -> ArmOutput:
        if self.llm is None:
            raise RuntimeError("LLM arms require an LLM client")
        facts = CaseFacts.from_dict(run["facts"])
        per: Dict[str, Dict[str, Any]] = {}
        calls = pt_total = ct_total = 0
        t0 = time.perf_counter()
        for rid in run["query_rule_ids"]:
            rule = self.catalogue.get(rid)
            known: Dict[str, Any] = {}
            final: Optional[Mapping[str, Any]] = None
            for _ in range(max_steps):
                user = (f"RULE: {_catalogue_summary(self.catalogue, [rid])}\nCASE ID: {facts.case_id}\n"
                        f"JURISDICTION: {facts.jurisdiction}; EVALUATION DATE: {facts.evaluation_date.isoformat()}\n"
                        f"FACT VALUES SO FAR:\n" + ("\n".join(f"- {k} = {json.dumps(v)}" for k, v in known.items()) or "- (none)"))
                text, pt, ct = self.llm.complete(SYSTEM_TRAVERSAL, user)
                calls += 1
                pt_total += pt
                ct_total += ct
                msg = _extract_json(text) or {}
                if isinstance(msg, Mapping) and msg.get("call") == "get_fact":
                    name = str(msg.get("name", ""))
                    if name in facts.values or name in known:
                        v = facts.values.get(name)
                        known[name] = v.value if isinstance(v, TV) else (list(v) if isinstance(v, tuple) else v)
                    else:
                        known[name] = "unknown"
                    continue
                if isinstance(msg, Mapping) and isinstance(msg.get("final"), Mapping):
                    final = msg["final"]
                    break
                break
            per.update(_normalise_llm_rule_output({rid: final or {}}, [rid]))
            per[rid]["source_ids"] = [rule.source.document_id]
        released = [(rid, e["action"]) for rid, e in per.items() if e["state"] == "applicable" and e["action"]]
        return ArmOutput("llm_traversal", run["run_id"], False, per, released, None, n_llm_calls=calls,
                         prompt_tokens=pt_total, completion_tokens=ct_total, latency_s=time.perf_counter() - t0)

    ARM_NAMES = ("deterministic_rules", "rules_plus_retrieval", "direct_llm", "naive_rag", "llm_traversal")

    def run_arm(self, name: str, run: Mapping[str, Any]) -> ArmOutput:
        return getattr(self, name)(run)


__all__ = ["ArmOutput", "Arms", "FakeLLMClient", "LLMClient", "LexicalChunkIndex", "LiveGatewayClient"]
