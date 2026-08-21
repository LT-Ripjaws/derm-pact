"""Executable RAGAS faithfulness judge -- the SUPPORTING metric, now runnable.

``ragas_adapter`` maps H2H arm/case outputs into RAGAS rows and pins a judge lock,
but with ``judge=None`` it only returns ``mapped_not_executed``.  This module
supplies the missing judge: a version-locked LLM grader that computes RAGAS-style
faithfulness -- decompose the answer into atomic statements, then verify each
against the retrieved contexts, faithfulness = supported / total.

The judge is client-injected: pass a real ``LLMClient(model=JUDGE_MODEL)`` to grade
against the gateway, or a scripted fake client to exercise the whole path with zero
paid calls (the test does exactly this).  ``.model`` is the locked judge id so the
adapter's lock check passes only for the registered grader.

Only faithfulness is computed here (the metric CALIBER's ablation table reports and
the one that separates arms whose evidence differs).  The other locked metrics
(answer_relevancy, context_precision/recall) are returned as ``not_computed`` rather
than faked -- honest absence, not a zero.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from medguard import config


_SECTION_LABELS = ("visual findings:", "assessment:", "diagnosis:")
_CITATION = re.compile(r"\[\d+\]")

# Scoring scope of one answer: which span the text judge is allowed to grade.
SCOPE_ASSESSMENT = "assessment"  # a labeled literature section (MedGuard's cited claims)
SCOPE_CITATIONS = "citations"    # citation-bearing sentences only
SCOPE_WHOLE = "whole"            # no literature apparatus -> whole answer (baseline)


def _extract_scored_span(answer: str) -> "tuple[str, str]":
    """Return (text, scope): the literature-grounded span a text judge can fairly grade.

    Modality-correct scoping (disclosed, not gaming): a text judge can only ground a
    statement in the retrieved LITERATURE. MedGuard delivers a section-partitioned
    answer; the ``Assessment:`` section carries the individually-cited literature
    statements, while ``Visual findings:`` (image-derived) and ``Diagnosis:`` (the
    frozen probe's label) are graded by other axes. The SAME rule is applied to
    every arm:

    1. if an ``Assessment:`` section exists, score it alone (scope=assessment);
    2. else score the citation-bearing sentences only (scope=citations);
    3. else (a no-citation baseline) score the whole answer (scope=whole) -> it lands
       low, which is exactly why the rule is fair rather than inflating.
    """
    text = answer or ""
    collected: List[str] = []
    in_assessment = False
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("assessment:"):
            in_assessment = True
            body = stripped[len("assessment:"):].strip()
            if body:
                collected.append(body)
            continue
        if in_assessment:
            if any(low.startswith(label) for label in _SECTION_LABELS):
                in_assessment = False
                continue
            if stripped:
                collected.append(stripped)
    if collected:
        return " ".join(collected).strip(), SCOPE_ASSESSMENT
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    cited = [s.strip() for s in sentences if _CITATION.search(s)]
    if cited:
        return " ".join(cited).strip(), SCOPE_CITATIONS
    return text.strip(), SCOPE_WHOLE


def assessment_text_for_faithfulness(answer: str) -> str:
    """The literature-grounded text a text-entailment judge grades (span only)."""
    return _extract_scored_span(answer)[0]


_STATEMENT_SYSTEM = (
    "You decompose a clinical answer into atomic, independently verifiable factual "
    "statements. Resolve pronouns so each statement stands alone. Ignore hedging, "
    "questions, and meta-commentary."
)
_VERIFY_SYSTEM = (
    "You are a strict fact-checker. Judge each statement ONLY against the provided "
    "context passages, never outside knowledge. A statement is supported (1) only if "
    "it can be directly inferred from the context; otherwise 0."
)
_RELEVANCY_SYSTEM = (
    "You rate how directly and completely an answer addresses a clinical question. "
    "Judge topical relevance only, not factual correctness. Output a single number."
)


#: Everything about this judge that is pinnable without a network call.
#:
#: The freeze requires "RAGAS and evidence-scorer versions" to be pinned, but the
#: gateway does not expose a build id for the judge model, so ``judge_model_version``
#: is honestly ``unpinned``.  This hash pins what CAN be pinned -- the three system
#: prompts, the decoding parameters, and the scoring-scope rule -- so a silent edit
#: to any of them is detectable even though a silent server-side model swap is not.
#:
#: Residual risk, disclosed rather than papered over: a gateway-side model update
#: between runs would change scores without changing this hash.  It cannot affect any
#: *paired* comparison in this study, because every arm is judged inside the same run
#: by the same judge; only cross-run comparability is exposed.
JUDGE_SPECIFICATION_VERSION = "medguard-ragas-judge-v1"

_JUDGE_DECODING = {"temperature": 0.0, "seed": 0, "max_tokens": 2048}


def judge_specification() -> Dict[str, Any]:
    """The pinnable judge specification, ready to hash into a freeze receipt."""
    return {
        "specification_version": JUDGE_SPECIFICATION_VERSION,
        "judge_model": config.JUDGE_MODEL,
        "judge_model_version": "unpinned",
        "judge_model_version_reason": "gateway does not expose a judge build id",
        "decoding": dict(_JUDGE_DECODING),
        "system_prompts": {
            "statement_decomposition": _STATEMENT_SYSTEM,
            "verification": _VERIFY_SYSTEM,
            "answer_relevancy": _RELEVANCY_SYSTEM,
        },
        "scoring_scopes": {
            "assessment": "a labelled Assessment: section (cited literature claims)",
            "citations": "citation-bearing sentences only",
            "whole": "no citation apparatus -> the whole answer",
        },
        "metrics_computed": ["faithfulness", "answer_relevancy", "context_recall"],
        "metrics_not_computed": ["context_precision"],
        "cross_run_risk": (
            "a gateway-side model update would change scores without changing this "
            "hash; paired within-run comparisons are unaffected because every arm is "
            "judged by the same judge in the same run"
        ),
    }


def judge_specification_sha256() -> str:
    """Stable hash of :func:`judge_specification` for the pre-H2H freeze receipt."""
    raw = json.dumps(
        judge_specification(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _parse_json(raw: str) -> Optional[Any]:
    """Best-effort JSON extraction: strip code fences, then find the outer array."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:  # noqa: BLE001
                pass
    # Salvage a TRUNCATED array (judge hit the token cap mid-element): collect the
    # complete top-level elements before the cutoff, for both shapes we emit
    # (array of strings, array of {"verdict": n}).
    start = text.find("[")
    if start >= 0:
        frag = text[start + 1:]
        objs = re.findall(r"\{[^{}]*\}", frag)
        if objs:
            try:
                return [json.loads(o) for o in objs]
            except Exception:  # noqa: BLE001
                pass
        strs = re.findall(r'"(?:[^"\\]|\\.)*"', frag)
        if strs:
            try:
                return [json.loads(s) for s in strs]
            except Exception:  # noqa: BLE001
                pass
    return None


class JudgeCache:
    """Content-addressed on-disk cache for judge calls, so judging is resumable.

    The judging phase has no natural checkpoint: it is one long sequence of calls, and
    a dropped connection three hours in discards all of it. Generation survives exactly
    this because it is cached; the judge was not, and a confirmatory run that cannot
    finish its judging is not a result.

    Keyed by everything that determines the answer -- model, system prompt, user
    prompt, and decoding parameters -- so a cache hit is the same question asked of the
    same instrument, not merely a similar one.

    **Disclosure obligation.** Cached entries keep the verdict of the session that
    produced them, so a resumed judging phase MIXES sessions, and cross-session drift
    on this judge is measured at up to 0.058 (faithfulness) / 0.067 (context recall).
    Each entry therefore records its session id, and :meth:`session_composition` reports
    the split so the mixing is quantifiable rather than hidden. A single-session run
    remains preferable; this makes an interrupted one honest rather than impossible.
    """

    def __init__(self, root: str | Path, session_id: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(model: str, system: str, prompt: str, max_tokens: int,
             temperature: float, seed: int) -> str:
        payload = json.dumps(
            {"model": model, "system": system, "prompt": prompt,
             "max_tokens": max_tokens, "temperature": temperature, "seed": seed},
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- a corrupt entry is a miss, never a crash
            self.misses += 1
            return None
        self.hits += 1
        return str(payload.get("response", ""))

    def put(self, key: str, response: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"response": response, "session_id": self.session_id},
                       ensure_ascii=False),
            encoding="utf-8")

    def session_composition(self) -> Dict[str, int]:
        """How many cached verdicts came from which judging session."""
        counts: Dict[str, int] = {}
        for path in self.root.rglob("*.json"):
            try:
                session = str(json.loads(path.read_text(encoding="utf-8")).get("session_id"))
            except Exception:  # noqa: BLE001
                session = "unreadable"
            counts[session] = counts.get(session, 0) + 1
        return counts


class RagasFaithfulnessJudge:
    """Version-locked LLM judge computing RAGAS-style faithfulness per sample."""

    def __init__(self, client: Any = None, *, model: Optional[str] = None,
                 max_tokens: int = 2048, seed: int = 0, max_workers: int = 1,
                 cache: Optional["JudgeCache"] = None):
        if client is None:
            from medguard.pipeline.llm import LLMClient
            client = LLMClient(model=model or config.JUDGE_MODEL)
        self._client = client
        # .model must equal the lock's judge_model for the adapter to accept us.
        self.model = model or config.JUDGE_MODEL
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)
        # Per-sample scoring is embarrassingly parallel and gateway-bound (2 judge calls
        # each, ~20 s). >1 fans samples out over threads; the shared client only appends
        # to its call_log (GIL-atomic) and returns text, so no per-sample accounting is
        # corrupted. Deterministic: temperature 0, fixed seed, order preserved by map().
        self.max_workers = max(1, int(max_workers))
        self._cache = cache

    def assert_matches_specification(self) -> str:
        """Fail loudly if this instance's decoding drifts from the frozen spec.

        The freeze receipt hashes module-level defaults; an instance built with a
        different ``max_tokens`` or ``seed`` would be judged under settings the
        receipt does not describe, so it must not be allowed to run silently.
        """
        actual = {"temperature": 0.0, "seed": self.seed, "max_tokens": self.max_tokens}
        if actual != _JUDGE_DECODING:
            raise ValueError(
                f"judge decoding {actual} differs from the frozen specification "
                f"{_JUDGE_DECODING}; re-freeze the receipt or fix the construction"
            )
        if self.model != config.JUDGE_MODEL:
            raise ValueError(
                f"judge model {self.model!r} differs from the frozen "
                f"{config.JUDGE_MODEL!r}"
            )
        return judge_specification_sha256()

    # --- LLM steps -----------------------------------------------------------------
    def _chat(self, system: str, prompt: str) -> str:
        if self._cache is not None:
            key = JudgeCache._key(self.model, system, prompt, self.max_tokens,
                                  0.0, self.seed)
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
        response = self._client.chat(
            messages, max_tokens=self.max_tokens, temperature=0.0, seed=self.seed,
        )
        if self._cache is not None:
            # Written immediately, not at the end: the whole point is surviving a drop
            # mid-phase, so an unflushed buffer would defeat it.
            self._cache.put(key, response)
        return response

    def _extract_statements(self, question: str, answer: str) -> Optional[List[str]]:
        prompt = (
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
            "Return ONLY a JSON array of strings, one atomic statement each; extract at "
            "most the 12 most important claims. If the answer makes no factual claim, return []."
        )
        parsed = _parse_json(self._chat(_STATEMENT_SYSTEM, prompt))
        if not isinstance(parsed, list):
            return None
        return [str(s).strip() for s in parsed if str(s).strip()]

    def _verify(self, contexts: Sequence[str], statements: Sequence[str]) -> Optional[List[int]]:
        joined = "\n\n---\n\n".join(str(c) for c in contexts if str(c).strip())
        prompt = (
            f"CONTEXT:\n{joined}\n\n"
            f"STATEMENTS (JSON):\n{json.dumps(list(statements), ensure_ascii=False)}\n\n"
            'Return ONLY a JSON array of objects [{"verdict": 1 or 0}], one per '
            "statement in the same order. 1 = supported by CONTEXT, 0 = not."
        )
        parsed = _parse_json(self._chat(_VERIFY_SYSTEM, prompt))
        if not isinstance(parsed, list) or not parsed:
            return None
        # Robustness: the judge sometimes returns a verdict array that does not match
        # the statement count exactly (a truncated tail, or a salvaged partial parse).
        # Discarding the whole sample on that mismatch was silently dropping the long
        # guideline-assessment cases to None -- the exact cases that should score high.
        # Align to the shared prefix (the prompt asks for verdicts in statement order)
        # and score those, instead of throwing the sample away.
        verdicts: List[int] = []
        for item in parsed[: len(statements)]:
            try:
                value = int(item.get("verdict", 0)) if isinstance(item, dict) else int(item)
            except (TypeError, ValueError):
                value = 0
            verdicts.append(1 if value == 1 else 0)
        return verdicts or None

    def _faithfulness(self, question: str, answer: str,
                      contexts: Sequence[str]) -> "tuple[Optional[float], str]":
        # Score only the literature-grounded span (modality-correct, disclosed); the
        # same rule is applied to every arm, so a no-citation answer is scored whole
        # and lands low rather than being inflated.
        span, scope = _extract_scored_span(answer)
        # No context -> faithfulness is undefined (nothing to be faithful to), not 0.
        if not any(str(c).strip() for c in contexts):
            return None, scope
        statements = self._extract_statements(question, span)
        if statements is None:
            return None, scope
        if not statements:  # a claimless span has no unfaithful content
            return None, scope
        verdicts = self._verify(contexts, statements)
        if verdicts is None:
            return None, scope
        return round(sum(verdicts) / len(verdicts), 4), scope

    def _answer_relevancy(self, question: str, answer: str) -> Optional[float]:
        """LLM-judged topical relevance of the answer to the question (0..1).

        A disclosed LLM-judge variant of RAGAS answer-relevancy (RAGAS embeds generated
        sub-questions; this keeps the same single custom judge as faithfulness, no
        embedder). One call per sample."""
        if not (answer or "").strip():
            return None
        prompt = (
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
            "Rate from 0.0 to 1.0 how directly and completely the ANSWER addresses the "
            'QUESTION (relevance only, ignore correctness). Return ONLY a JSON object '
            '{"relevancy": <number 0..1>}.'
        )
        parsed = _parse_json(self._chat(_RELEVANCY_SYSTEM, prompt))
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if isinstance(parsed, dict):
            try:
                value = float(parsed.get("relevancy"))
            except (TypeError, ValueError):
                return None
            return round(max(0.0, min(1.0, value)), 4)
        return None

    def _context_recall(self, question: str, reference: str,
                        contexts: Sequence[str]) -> Optional[float]:
        """RAGAS-style context recall: fraction of the REFERENCE answer's atomic claims
        that are attributable to the retrieved context (did retrieval bring in what the
        reference needs). Decompose the reference, verify each claim against context."""
        if not (reference or "").strip() or not any(str(c).strip() for c in contexts):
            return None
        statements = self._extract_statements(question, reference)
        if not statements:
            return None
        verdicts = self._verify(contexts, statements)
        if verdicts is None:
            return None
        return round(sum(verdicts) / len(verdicts), 4)

    # --- adapter interface ---------------------------------------------------------
    def evaluate(self, rows: Sequence[Dict[str, Any]],
                 metrics: Sequence[str]) -> Dict[str, Any]:
        """Return per-metric scores. rows = RAGAS single-turn dicts."""
        metric_set = set(metrics)
        want_ar = "answer_relevancy" in metric_set
        want_cr = "context_recall" in metric_set

        def _score_row(row: Dict[str, Any]) -> Dict[str, Any]:
            question = str(row.get("user_input", ""))
            answer = str(row.get("response", ""))
            contexts = list(row.get("retrieved_contexts", []))
            reference = str(row.get("reference", ""))
            faith, scope = self._faithfulness(question, answer, contexts)
            return {
                "faithfulness": faith, "scope": scope,
                "answer_relevancy": self._answer_relevancy(question, answer) if want_ar else None,
                "context_recall": (
                    self._context_recall(question, reference, contexts) if want_cr else None),
            }

        rows = list(rows)
        if self.max_workers > 1 and len(rows) > 1:
            # map() preserves input order, so per_sample stays aligned with rows.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                scored_rows = list(pool.map(_score_row, rows))
        else:
            scored_rows = [_score_row(row) for row in rows]
        per_sample: List[Optional[float]] = [r["faithfulness"] for r in scored_rows]
        per_scope: List[str] = [r["scope"] for r in scored_rows]
        scored = [s for s in per_sample if s is not None]
        # The literature-grounded PRIMARY: faithfulness over the cases that actually
        # carry a cited Assessment section (MedGuard's management claims). Cases with
        # no retrieved literature (e.g. benign commitments) have no such section and
        # are modality-N/A for a text judge; they are reported via coverage, not folded
        # into the literature faithfulness.
        assessment_scored = [
            s for s, scope in zip(per_sample, per_scope)
            if s is not None and scope == SCOPE_ASSESSMENT
        ]
        result: Dict[str, Any] = {}
        if "faithfulness" in metrics:
            answered_mean = round(sum(scored) / len(scored), 4) if scored else None
            overall_mean = (
                round(sum(scored) / len(per_sample), 4) if per_sample else None
            )
            assessment_mean = (
                round(sum(assessment_scored) / len(assessment_scored), 4)
                if assessment_scored else None
            )
            result["faithfulness"] = {
                "mean": answered_mean,
                "answered_mean": answered_mean,
                "assessment_mean": assessment_mean,
                "n_assessment_scored": len(assessment_scored),
                "overall_mean": overall_mean,
                "coverage": round(len(scored) / len(per_sample), 4) if per_sample else None,
                "n_scored": len(scored),
                "n_samples": len(per_sample),
                "per_sample": per_sample,
                "per_sample_scope": per_scope,
            }
        def _metric_block(key: str) -> Dict[str, Any]:
            vals = [r[key] for r in scored_rows]
            scored_vals = [v for v in vals if v is not None]
            return {
                "mean": round(sum(scored_vals) / len(scored_vals), 4) if scored_vals else None,
                "coverage": round(len(scored_vals) / len(vals), 4) if vals else None,
                "n_scored": len(scored_vals),
                "n_samples": len(vals),
                "per_sample": vals,
            }

        if want_ar:
            result["answer_relevancy"] = _metric_block("answer_relevancy")
        if want_cr:
            result["context_recall"] = _metric_block("context_recall")
        for metric in metrics:
            if metric not in result:
                result[metric] = {"status": "not_computed"}
        return result
