"""Ask a VLM to read the seven dermoscopic criteria off an image -- and nothing else.

This is the perception half of the v2 split.  The model judges *whether a criterion is
present*; :mod:`medguard_v2.criteria.aggregate` does the arithmetic.  The model is never shown
the point weights, never told the threshold, and never asked for a score, so a correct total can
only come from correct perception plus the host's rule -- not from the model recognising
"melanoma" and reverse-engineering criteria that justify it.

Five properties this module is built around
-------------------------------------------

**The schema is the gate, not a suggestion.**  Every response goes through
:func:`~medguard_v2.criteria.schema.validate_values`.  A value outside the vocabulary earns one
repair attempt with the specific error quoted back; if the repair also fails the case is recorded
as a ``parse_failure`` and carries **no values at all**.  It is never coerced to ``absent``.
Coercing to ``absent`` would silently manufacture a 7-point score of 0 -- a confident
"benign, no criteria present" verdict fabricated out of a parsing bug.  That is the single most
dangerous failure mode available to this component, so it is structurally impossible here.

**Every permitted value is defined, not just the ones that score.**  ``schema.py``'s
``description`` field is written from the checklist's point of view: for ``vascular_structures``
it reads "Atypical vascular pattern: dotted or linear-irregular vessels", which (a) defines two
of eight permitted values and leaves the other six undefined, and (b) tells the model exactly
which values the host's rule rewards.  Both are disqualifying in a readability gate: a model that
has never been told what "arborizing" looks like will not report it, and a model that knows which
values score can pattern-match a diagnosis and back-fill.  :data:`VALUE_DEFINITIONS` therefore
gives a neutral morphological definition for **every** value in every vocabulary, and
``_check_value_definitions()`` fails at import if the schema ever grows a value the table does not
cover.  The vocabularies themselves still come from ``schema.py`` verbatim -- nothing here
re-states them.

**Declining is allowed.**  ``INDETERMINATE`` is offered to the model as a first-class answer and
described as such in the prompt.  A model forced to guess on an unreadable region produces a
worse gate than one permitted to abstain: the guess pollutes the accuracy estimate with noise
that looks like signal, whereas an abstention is visible in the indeterminate rate and propagates
into a *score range* rather than a wrong point estimate.

**Surface normalisation is not semantic coercion.**  ``"Atypical"``, ``"atypical "`` and
``"linear-irregular"`` are lowercased / whitespace-collapsed / punctuation-normalised into the
vocabulary form, and every such rewrite is recorded in ``normalisations``.  No mapping between
*different* vocabulary values is ever performed.

**Re-runs are free.**  Every *completed* assessment -- success or schema failure -- is cached on
disk under a key derived from the image bytes, the exact prompt text, the model id, the decoding
parameters and the repair budget.  Caching schema failures is deliberate: it keeps a re-run
byte-identical instead of letting a flaky case silently resolve differently on the second pass.
Transport failures (``api_failure``) are **not** cached -- a 502 is a property of the network, not
of the request, and freezing it would make an outage permanent.  Pass ``refresh=True`` to
deliberately re-pay.

Because the key is over image bytes rather than paths, two cases sharing an identical image (the
nine Derm7pt cases whose clinical view is a copy of the dermoscopic one) share one cache entry.
Case identity is therefore attached on read, never trusted from the stored payload.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from medguard_v2.criteria.schema import (
    CRITERIA_BY_NAME,
    CRITERION_NAMES,
    CRITERION_SPECS,
    INDETERMINATE,
    validate_values,
)

__all__ = [
    "PROMPT_VERSION",
    "SCHEMA_VERSION",
    "CRITERION_FRAMING",
    "VALUE_DEFINITIONS",
    "DecodingParams",
    "AssessorRequest",
    "CallStats",
    "CriterionAssessment",
    "AssessorCache",
    "CriterionAssessor",
    "build_system_prompt",
    "build_user_prompt",
    "build_repair_prompt",
    "parse_response",
    "estimate_image_tokens",
    "estimate_text_tokens",
    "default_client",
]

#: Bump when the prompt text changes.  It is part of the cache key, so old answers are never
#: reused against a new prompt.
PROMPT_VERSION = "g1-criterion-assessor-v2"
SCHEMA_VERSION = 2

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "research" / "evidence" / "derm7pt_v1" / "g1_assessor_cache"

DEFAULT_MODEL = "qwen/qwen3-vl-32b-instruct"


# ---------------------------------------------------------------------------
# What each criterion *is*, and what each of its permitted values looks like.
#
# Deliberately neutral: no value is described as suspicious, atypical-in-the-scoring-sense,
# benign, or diagnostic.  The checklist's own framing lives in schema.py and is not sent.
# ---------------------------------------------------------------------------
CRITERION_FRAMING: Mapping[str, str] = {
    "pigment_network": "The reticular grid of pigmented lines ('network') over the lesion.",
    "blue_whitish_veil": "A confluent blue-grey area overlaid by a whitish haze.",
    "vascular_structures": "The morphology of blood vessels visible inside the lesion.",
    "streaks": "Radial linear projections or bulbous pseudopods at the lesion border.",
    "pigmentation": "Structureless areas of pigment (blotches), as opposed to structured patterns.",
    "dots_and_globules": "Round pigmented dots and globules within the lesion.",
    "regression_structures": "White scar-like areas and/or blue-grey granular peppering.",
}

VALUE_DEFINITIONS: Mapping[str, Mapping[str, str]] = {
    "pigment_network": {
        "absent": "no reticular grid of pigmented lines anywhere in the lesion",
        "typical": (
            "a regular grid of thin, evenly coloured brown lines with uniformly sized holes, "
            "thinning out gradually towards the edge"
        ),
        "atypical": (
            "a grid whose lines are thickened, broken or unevenly coloured, with holes that vary "
            "in size and shape, often ending abruptly at the edge"
        ),
    },
    "blue_whitish_veil": {
        "absent": "no confluent blue-grey area with an overlying whitish film",
        "present": (
            "a confluent blue-grey pigmented area covered by a white 'ground-glass' film, not "
            "occupying the whole lesion"
        ),
    },
    "vascular_structures": {
        "absent": "no vessels discernible inside the lesion",
        "arborizing": (
            "large-calibre bright red vessels branching into progressively thinner ramifications, "
            "sharply in focus"
        ),
        "comma": "short, gently curved, comma-shaped vessels of low branching complexity",
        "hairpin": (
            "elongated vessels looping back on themselves like a hairpin, often with a whitish halo"
        ),
        "within regression": "fine vessels seen inside white scar-like depigmented areas",
        "wreath": (
            "short vessels arranged in crown-like arcs around the periphery, not crossing the "
            "centre of the lesion"
        ),
        "dotted": (
            "numerous tiny pinpoint red dots at the limit of resolution, densely and unevenly "
            "distributed"
        ),
        "linear irregular": (
            "red linear vessels of irregular shape, calibre and distribution, fitting no regular "
            "arrangement"
        ),
    },
    "streaks": {
        "absent": "no radial lines or pseudopods at the lesion border",
        "regular": (
            "radial lines or bulbous pseudopods distributed symmetrically around the whole border"
        ),
        "irregular": (
            "radial lines or pseudopods present at only some segments of the border, unevenly "
            "distributed"
        ),
    },
    "pigmentation": {
        "absent": "no structureless pigmented blotch",
        "diffuse regular": (
            "structureless pigmentation covering most of the lesion, uniform in colour and "
            "symmetrically placed"
        ),
        "localized regular": (
            "structureless pigmentation confined to part of the lesion, uniform in colour and "
            "symmetrically placed"
        ),
        "diffuse irregular": (
            "structureless pigmentation over most of the lesion but uneven in colour or density "
            "and asymmetrically placed"
        ),
        "localized irregular": (
            "a structureless pigmented blotch confined to part of the lesion, eccentric or "
            "asymmetrically placed"
        ),
    },
    "dots_and_globules": {
        "absent": "no dots or globules",
        "regular": (
            "round-oval brown or black dots and globules of similar size and colour, evenly "
            "distributed"
        ),
        "irregular": (
            "dots and globules varying in size, shape or colour, unevenly distributed or clustered "
            "at one part of the lesion"
        ),
    },
    "regression_structures": {
        "absent": "no white scar-like areas and no blue-grey peppering",
        "blue areas": "blue-grey granular peppering, without white scar-like areas",
        "white areas": (
            "white scar-like areas lighter than the surrounding skin, without blue-grey peppering"
        ),
        "combinations": "both white scar-like areas and blue-grey peppering",
    },
}


def _check_value_definitions() -> None:
    """Fail loudly at import if the definition table and the schema have drifted apart.

    A value that exists in the schema but not here would be offered to the model as a bare word
    with no definition; a value here but not in the schema would be offered and then rejected by
    the validator.  Both produce a gate that measures the prompt rather than the model.
    """
    problems: List[str] = []
    for spec in CRITERION_SPECS:
        if spec.name not in CRITERION_FRAMING:
            problems.append(f"{spec.name}: no framing line")
        defined = set(VALUE_DEFINITIONS.get(spec.name, {}))
        missing = sorted(spec.vocabulary - defined)
        extra = sorted(defined - spec.vocabulary)
        if missing:
            problems.append(f"{spec.name}: undefined values {missing}")
        if extra:
            problems.append(f"{spec.name}: definitions for non-schema values {extra}")
    if problems:
        raise RuntimeError("VALUE_DEFINITIONS is out of sync with schema.py: " + "; ".join(problems))


_check_value_definitions()


# ---------------------------------------------------------------------------
# Prompt construction -- every permitted value comes from schema.py, never a literal here
# ---------------------------------------------------------------------------
def _ordered_vocabulary(name: str) -> Tuple[str, ...]:
    """Deterministic display order: ``absent`` first (the reference state), then sorted.

    Determinism matters more than it looks: the prompt text is part of the cache key, and
    ``frozenset`` iteration order is not stable across processes.
    """
    vocabulary = CRITERIA_BY_NAME[name].vocabulary
    rest = sorted(value for value in vocabulary if value != "absent")
    return (("absent",) if "absent" in vocabulary else ()) + tuple(rest)


def build_system_prompt() -> str:
    return (
        "You are a dermoscopy image reader. You judge individual morphological criteria that are "
        "visible in an image. You never diagnose, never name a disease, never compute or mention "
        "any score or total, and never state or imply a management recommendation. Those are "
        "decided elsewhere from your criterion judgements. Answer only with the requested JSON."
    )


def build_user_prompt(*, has_clinical: bool = False) -> str:
    views = (
        "You are given two images of the same lesion: the FIRST is the dermoscopic image and the "
        "SECOND is a clinical (macro) photograph. Judge the criteria from the DERMOSCOPIC image; "
        "use the clinical image only for context."
        if has_clinical
        else "You are given one dermoscopic image of a skin lesion."
    )

    blocks: List[str] = []
    for name in CRITERION_NAMES:
        lines = [f'"{name}" -- {CRITERION_FRAMING[name]}', "  permitted values (choose exactly one):"]
        lines += [
            f'    "{value}" = {VALUE_DEFINITIONS[name][value]}' for value in _ordered_vocabulary(name)
        ]
        blocks.append("\n".join(lines))
    criteria_block = "\n\n".join(blocks)

    example_keys = ",\n".join(
        f'  "{name}": {{"value": "<one permitted value or \\"{INDETERMINATE}\\">", '
        f'"confidence": <number between 0 and 1>, "justification": "<one short line>"}}'
        for name in CRITERION_NAMES
    )

    return f"""{views}

Assess each of the seven dermoscopic criteria below, independently of one another.

{criteria_block}

RULES
1. For every criterion return exactly one value copied verbatim from that criterion's permitted
   list. Do not invent values, do not abbreviate, do not translate, do not merge two values.
2. If a criterion genuinely cannot be judged from the image -- the region is out of frame, out of
   focus, obscured by hair, bubbles, glare or ink, or the image quality does not let you tell --
   return "{INDETERMINATE}". This is a legitimate and expected answer, not a failure. Declining is
   strictly better than guessing: a wrong confident value is worse than an honest abstention.
3. "confidence" is your probability that your chosen value is the one an expert dermoscopist
   would record, as a number between 0 and 1.
4. "justification" is ONE short line naming the visual feature you actually saw (or, for
   "{INDETERMINATE}", what prevented the judgement). No diagnosis, no score, no management advice.
5. Judge only what is visible. Do not infer one criterion from another, and do not let an
   impression of the overall lesion decide an individual criterion.

Return ONLY a JSON object with exactly these seven keys and no other text:

{{
{example_keys}
}}"""


def build_repair_prompt(*, previous: str, error: str, has_clinical: bool = False) -> str:
    return f"""{build_user_prompt(has_clinical=has_clinical)}

YOUR PREVIOUS RESPONSE WAS REJECTED BY THE SCHEMA VALIDATOR.

Previous response:
{previous}

Validator error:
{error}

Fix exactly this problem. Every value must be copied verbatim from that criterion's permitted
list, or be "{INDETERMINATE}". All seven keys must be present. Return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Token estimation (dry-run cost projection; no network)
# ---------------------------------------------------------------------------
def estimate_text_tokens(text: str) -> int:
    """Approximate token count. Uses ``tiktoken`` when available, else 4 chars/token.

    The served model is a Qwen VL variant whose tokeniser is not cl100k, so this is an estimate
    and is labelled as one everywhere it surfaces. The live run reports *observed* prompt tokens
    and a calibration ratio, so the projection can be corrected against reality rather than
    trusted.
    """
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # noqa: BLE001 -- estimation must never be the thing that fails
        return max(1, len(text) // 4)


def _smart_resize(
    height: int, width: int, *, factor: int = 28,
    min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28,
) -> Tuple[int, int]:
    """Qwen-VL dynamic-resolution grid: round each side to a multiple of ``factor``."""
    h_bar = max(factor, int(round(height / factor)) * factor)
    w_bar = max(factor, int(round(width / factor)) * factor)
    if h_bar * w_bar > max_pixels:
        beta = ((height * width) / max_pixels) ** 0.5
        h_bar = max(factor, int(height / beta // factor) * factor)
        w_bar = max(factor, int(width / beta // factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        h_bar = int((height * beta + factor - 1) // factor) * factor
        w_bar = int((width * beta + factor - 1) // factor) * factor
    return h_bar, w_bar


def estimate_image_tokens(path: Path | str, *, factor: int = 28) -> int:
    """Visual tokens for one image under Qwen-VL's 28x28-pixel-per-token grid, +2 delimiters."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001 -- fall back to the Derm7pt modal size
        width, height = 768, 512
    h_bar, w_bar = _smart_resize(height, width, factor=factor)
    return (h_bar * w_bar) // (factor * factor) + 2


# ---------------------------------------------------------------------------
# Request identity and on-disk cache
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecodingParams:
    model: str
    temperature: float = 0.0
    max_tokens: int = 900
    seed: Optional[int] = 20260810

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class AssessorRequest:
    """Everything that determines the answer -- and therefore the whole cache key."""

    case_index: int
    case_num: int
    image_paths: Tuple[Path, ...]
    image_roles: Tuple[str, ...]
    image_sha256: Tuple[str, ...]
    system_prompt: str
    user_prompt: str
    decoding: DecodingParams
    has_clinical: bool
    max_repair_attempts: int = 1

    def identity(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "decoding": self.decoding.to_dict(),
            # Part of the procedure, not of the request -- but it changes the answer for any case
            # whose first response fails the schema, and schema failures are cached.
            "max_repair_attempts": int(self.max_repair_attempts),
            "images": [
                {"role": role, "sha256": digest}
                for role, digest in zip(self.image_roles, self.image_sha256)
            ],
        }

    def cache_key(self) -> str:
        payload = json.dumps(
            self.identity(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class CallStats:
    """Per-case accounting. On a cache hit these are the *original* call's numbers, replayed;
    ``cache_hit`` is what tells a reporter not to bill them again."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    remote_calls: int = 0          # logical calls we issued (1 + repairs)
    transport_attempts: int = 0    # including the client's own transient retries
    repair_attempts: int = 0
    cache_hit: bool = False
    usage_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
            "latency_seconds": round(float(self.latency_seconds), 4),
            "remote_calls": int(self.remote_calls),
            "transport_attempts": int(self.transport_attempts),
            "repair_attempts": int(self.repair_attempts),
            "cache_hit": bool(self.cache_hit),
            "usage_available": bool(self.usage_available),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CallStats":
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
            latency_seconds=float(data.get("latency_seconds", 0.0) or 0.0),
            remote_calls=int(data.get("remote_calls", 0) or 0),
            transport_attempts=int(data.get("transport_attempts", 0) or 0),
            repair_attempts=int(data.get("repair_attempts", 0) or 0),
            cache_hit=bool(data.get("cache_hit", False)),
            usage_available=bool(data.get("usage_available", True)),
        )


@dataclass
class CriterionAssessment:
    """One case's outcome. ``values`` is ``None`` exactly when nothing usable was produced."""

    case_index: int
    case_num: int
    status: str                                   # "ok" | "parse_failure" | "api_failure"
    values: Optional[Dict[str, str]] = None
    confidences: Dict[str, Optional[float]] = field(default_factory=dict)
    justifications: Dict[str, str] = field(default_factory=dict)
    raw_values: Dict[str, str] = field(default_factory=dict)
    normalisations: Dict[str, str] = field(default_factory=dict)
    soft_defects: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    stats: CallStats = field(default_factory=CallStats)
    cache_key: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.values is not None

    @property
    def indeterminate_criteria(self) -> Tuple[str, ...]:
        if not self.values:
            return ()
        return tuple(name for name in CRITERION_NAMES if self.values[name] == INDETERMINATE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_index": self.case_index,
            "case_num": self.case_num,
            "status": self.status,
            "values": dict(self.values) if self.values else None,
            "confidences": dict(self.confidences),
            "justifications": dict(self.justifications),
            "raw_values": dict(self.raw_values),
            "normalisations": dict(self.normalisations),
            "soft_defects": list(self.soft_defects),
            "failure_reason": self.failure_reason,
            "attempts": list(self.attempts),
            "stats": self.stats.to_dict(),
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriterionAssessment":
        return cls(
            case_index=int(data.get("case_index", -1)),
            case_num=int(data.get("case_num", -1)),
            status=str(data["status"]),
            values=dict(data["values"]) if data.get("values") else None,
            confidences=dict(data.get("confidences") or {}),
            justifications=dict(data.get("justifications") or {}),
            raw_values=dict(data.get("raw_values") or {}),
            normalisations=dict(data.get("normalisations") or {}),
            soft_defects=list(data.get("soft_defects") or []),
            failure_reason=data.get("failure_reason"),
            attempts=list(data.get("attempts") or []),
            stats=CallStats.from_dict(data.get("stats") or {}),
            cache_key=str(data.get("cache_key", "")),
        )


class AssessorCache:
    """Atomic, content-addressed, self-verifying JSON cache."""

    def __init__(self, root: Path | str = DEFAULT_CACHE_DIR):
        self.root = Path(root)
        self._lock = threading.Lock()

    def entry_path(self, key: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[CriterionAssessment]:
        """Return the cached assessment, ``None`` on a miss or a superseded version.

        A *corrupt* entry raises instead of being treated as a miss: silently re-paying for a
        request whose stored answer no longer parses would hide the corruption.
        """
        path = self.entry_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SCHEMA_VERSION:
                return None
            if payload.get("prompt_version") != PROMPT_VERSION:
                return None
            if payload.get("request_sha256") != key:
                raise ValueError("request digest mismatch")
            assessment = CriterionAssessment.from_dict(payload["assessment"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid assessor cache entry {key}: {exc}") from exc
        # A cached answer is about the *inputs*, not the case that first asked for it.
        if assessment.values is not None:
            validate_values(assessment.values)
        assessment.cache_key = key
        assessment.stats.cache_hit = True
        return assessment

    def put(self, key: str, assessment: CriterionAssessment, identity: Mapping[str, Any]) -> None:
        path = self.entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "request_sha256": key,
            "request_identity": dict(identity),
            "first_seen_case_num": assessment.case_num,
            "assessment": assessment.to_dict(),
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with self._lock:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
            ) as handle:
                handle.write(encoded)
                temporary = Path(handle.name)
            try:
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()

    def count(self) -> int:
        if not self.root.is_dir():
            return 0
        return sum(1 for _ in self.root.rglob("*.json"))


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Pull the first balanced JSON object out of a completion. Raises ValueError on failure."""
    candidate = text.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    start = candidate.find("{")
    if start < 0:
        raise ValueError("response contains no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(candidate)):
        char = candidate[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(candidate[start:i + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("top-level JSON value is not an object")
                return parsed
    raise ValueError("response contains an unterminated JSON object")


def _normalise_value(raw: Any) -> str:
    """Surface-only normalisation. Never maps one vocabulary value onto a different one.

    Case, surrounding quotes, underscores, hyphens, commas, trailing full stops and repeated
    whitespace are cosmetic; ``"Linear-Irregular."`` and ``"linear irregular"`` are the same
    answer.  Nothing here can turn ``"typical"`` into ``"atypical"`` or ``"regular"`` into
    ``"irregular"``: no substring, synonym or prefix rule is applied.
    """
    text = str(raw).strip().strip('"').strip("'").lower()
    text = text.replace("_", " ").replace("-", " ").replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    return text.strip()


def _coerce_confidence(raw: Any) -> Tuple[Optional[float], Optional[str]]:
    if raw is None:
        return None, "missing"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"unparseable ({raw!r})"
    if value != value:  # NaN
        return None, "NaN"
    if value < 0.0 or value > 1.0:
        return min(1.0, max(0.0, value)), f"out of range ({value})"
    return value, None


@dataclass
class ParsedResponse:
    values: Dict[str, str]
    confidences: Dict[str, Optional[float]]
    justifications: Dict[str, str]
    raw_values: Dict[str, str]
    normalisations: Dict[str, str]
    soft_defects: List[str]


def parse_response(text: str) -> ParsedResponse:
    """Parse one completion into validated criterion values.

    Raises ``ValueError`` for anything the schema gate must reject: unparseable JSON, missing or
    unknown criteria, out-of-vocabulary values. Confidence/justification defects are *recorded*,
    not raised -- G1 measures whether criteria are readable, and discarding an otherwise valid
    seven-criterion judgement because the model wrote ``"high"`` instead of ``0.8`` would answer a
    different and much less interesting question.
    """
    payload = _extract_json_object(text)

    unknown = sorted(set(map(str, payload)) - set(CRITERION_NAMES))
    if unknown:
        raise ValueError(f"response contains unknown criteria: {unknown}")
    missing = sorted(set(CRITERION_NAMES) - set(map(str, payload)))
    if missing:
        raise ValueError(f"response is missing criteria: {missing}")

    parsed = ParsedResponse({}, {}, {}, {}, {}, [])
    for name in CRITERION_NAMES:
        entry = payload[name]
        if isinstance(entry, str):
            raw: Any = entry
            confidence_raw: Any = None
            justification = ""
            parsed.soft_defects.append(f"{name}: bare string, no confidence/justification object")
        elif isinstance(entry, Mapping):
            if "value" not in entry:
                raise ValueError(f"{name}: object has no 'value' key")
            raw = entry["value"]
            confidence_raw = entry.get("confidence")
            justification = str(entry.get("justification", "") or "").strip()
            if not justification:
                parsed.soft_defects.append(f"{name}: empty justification")
        else:
            raise ValueError(f"{name}: expected an object or a string, got {type(entry).__name__}")

        parsed.raw_values[name] = str(raw)
        normalised = _normalise_value(raw)
        if normalised != str(raw):
            parsed.normalisations[name] = f"{str(raw)!r} -> {normalised!r}"
        parsed.values[name] = normalised

        confidence, defect = _coerce_confidence(confidence_raw)
        parsed.confidences[name] = confidence
        if defect:
            parsed.soft_defects.append(f"{name}: confidence {defect}")
        parsed.justifications[name] = justification

    validate_values(parsed.values)  # the gate: raises on any out-of-vocabulary value
    return parsed


# ---------------------------------------------------------------------------
# The assessor
# ---------------------------------------------------------------------------
def default_client(model: Optional[str] = None, **kwargs: Any):
    """Build v1's gateway client. Imported lazily so this module has no heavy dependencies.

    v1's ``config`` reads credentials from the process environment (and, via ``load_env``, from
    its own dotenv file). No credential is read, printed or returned here.
    """
    from medguard.pipeline.llm import LLMClient  # noqa: PLC0415 -- lazy on purpose

    return LLMClient(model=model, **kwargs)


class CriterionAssessor:
    """Turn one Derm7pt case into seven validated criterion judgements (or an honest failure)."""

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 900,
        seed: Optional[int] = 20260810,
        include_clinical: bool = False,
        cache: Optional[AssessorCache] = None,
        cache_dir: Path | str | None = None,
        allow_remote: bool = True,
        refresh: bool = False,
        max_repair_attempts: int = 1,
    ):
        self.client = client
        self.decoding = DecodingParams(
            model=model, temperature=temperature, max_tokens=max_tokens, seed=seed
        )
        self.include_clinical = bool(include_clinical)
        self.cache = cache if cache is not None else AssessorCache(cache_dir or DEFAULT_CACHE_DIR)
        self.allow_remote = bool(allow_remote)
        self.refresh = bool(refresh)
        self.max_repair_attempts = int(max_repair_attempts)
        self.system_prompt = build_system_prompt()

    # -- request construction (used by --dry-run; touches no network) --------------------
    def build_request(self, case) -> AssessorRequest:
        paths: List[Path] = [Path(case.derm_path)]
        roles: List[str] = ["dermoscopic"]
        # Nine Derm7pt cases have clinic_path == derm_path. Sending it twice is both a wasted
        # image and a hard ValueError inside LLMClient.vision, so honour has_distinct_views.
        has_clinical = self.include_clinical and getattr(case, "has_distinct_views", True)
        if has_clinical:
            paths.append(Path(case.clinic_path))
            roles.append("clinical")

        return AssessorRequest(
            case_index=int(case.case_index),
            case_num=int(case.case_num),
            image_paths=tuple(paths),
            image_roles=tuple(roles),
            image_sha256=tuple(_file_sha256(path) for path in paths),
            system_prompt=self.system_prompt,
            user_prompt=build_user_prompt(has_clinical=has_clinical),
            decoding=self.decoding,
            has_clinical=has_clinical,
            max_repair_attempts=self.max_repair_attempts,
        )

    def estimate_request_tokens(self, request: AssessorRequest) -> Dict[str, int]:
        text_tokens = (
            estimate_text_tokens(request.system_prompt)
            + estimate_text_tokens(request.user_prompt)
        )
        image_tokens = sum(estimate_image_tokens(path) for path in request.image_paths)
        return {
            "text_tokens_estimated": text_tokens,
            "image_tokens_estimated": image_tokens,
            "prompt_tokens_estimated": text_tokens + image_tokens,
            "max_completion_tokens": int(self.decoding.max_tokens),
        }

    # -- one remote call ----------------------------------------------------------------
    def _call(self, request: AssessorRequest, prompt: str, stats: CallStats) -> str:
        if self.client is None:
            raise RuntimeError("no client configured; pass client= or run in --dry-run mode")
        before = len(getattr(self.client, "call_log", []) or [])
        started = time.perf_counter()
        try:
            text = self.client.vision(
                prompt,
                image_path=[str(path) for path in request.image_paths],
                system=request.system_prompt,
                model=self.decoding.model,
                max_tokens=self.decoding.max_tokens,
                temperature=self.decoding.temperature,
                seed=self.decoding.seed,
            )
        except Exception:
            # Count the attempt before re-raising, or a failed case reports zero effort spent.
            stats.remote_calls += 1
            stats.latency_seconds += time.perf_counter() - started
            after = len(getattr(self.client, "call_log", []) or [])
            stats.transport_attempts += max(1, after - before)
            raise
        elapsed = time.perf_counter() - started
        metadata = getattr(self.client, "last_call_metadata", None) or {}
        after = len(getattr(self.client, "call_log", []) or [])

        stats.remote_calls += 1
        stats.transport_attempts += max(1, after - before)
        stats.latency_seconds += float(metadata.get("latency_seconds") or elapsed)
        prompt_tokens = metadata.get("prompt_tokens")
        completion_tokens = metadata.get("completion_tokens")
        if prompt_tokens is None and completion_tokens is None:
            stats.usage_available = False
        stats.prompt_tokens += int(prompt_tokens or 0)
        stats.completion_tokens += int(completion_tokens or 0)
        stats.total_tokens += int(
            metadata.get("total_tokens") or ((prompt_tokens or 0) + (completion_tokens or 0))
        )
        return text or ""

    # -- the whole path -----------------------------------------------------------------
    def assess(self, case) -> CriterionAssessment:
        request = self.build_request(case)
        key = request.cache_key()

        if not self.refresh:
            cached = self.cache.get(key)
            if cached is not None:
                return replace(cached, case_index=request.case_index, case_num=request.case_num)

        if not self.allow_remote:
            raise RuntimeError(f"cache-only run missed on case {request.case_num} (key {key})")

        stats = CallStats()
        attempts: List[Dict[str, Any]] = []
        prompt = request.user_prompt
        last_error = "no attempt was made"

        def _record(index: int, error: Optional[str], response: Optional[str]) -> None:
            attempts.append({
                "attempt": index + 1,
                "kind": "repair" if index else "initial",
                "error": error,
                "response": response,
            })

        for attempt_index in range(self.max_repair_attempts + 1):
            if attempt_index:
                stats.repair_attempts += 1
            try:
                text = self._call(request, prompt, stats)
            except Exception as exc:  # noqa: BLE001 -- transport failure is not a parse failure
                _record(attempt_index, f"{type(exc).__name__}: {exc}", None)
                # Not cached: a gateway outage is a property of the network, not of the request.
                return CriterionAssessment(
                    case_index=request.case_index, case_num=request.case_num,
                    status="api_failure", values=None,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    attempts=attempts, stats=stats, cache_key=key,
                )

            try:
                parsed = parse_response(text)
            except ValueError as exc:
                last_error = str(exc)
                _record(attempt_index, last_error, text[:2000])
                prompt = build_repair_prompt(
                    previous=text[:2000], error=last_error, has_clinical=request.has_clinical
                )
                continue

            _record(attempt_index, None, text[:2000])
            assessment = CriterionAssessment(
                case_index=request.case_index, case_num=request.case_num, status="ok",
                values=parsed.values, confidences=parsed.confidences,
                justifications=parsed.justifications, raw_values=parsed.raw_values,
                normalisations=parsed.normalisations, soft_defects=parsed.soft_defects,
                attempts=attempts, stats=stats, cache_key=key,
            )
            self.cache.put(key, assessment, request.identity())
            return assessment

        # Every attempt failed the schema gate. Record it as such and emit NO values:
        # coercing to "absent" here would fabricate a confident 7-point score of 0.
        assessment = CriterionAssessment(
            case_index=request.case_index, case_num=request.case_num,
            status="parse_failure", values=None,
            failure_reason=last_error, attempts=attempts, stats=stats, cache_key=key,
        )
        self.cache.put(key, assessment, request.identity())
        return assessment

    def assess_cases(self, cases: Sequence[Any], *, progress=None) -> List[CriterionAssessment]:
        out: List[CriterionAssessment] = []
        for index, case in enumerate(cases, start=1):
            assessment = self.assess(case)
            out.append(assessment)
            if progress is not None:
                progress(index, len(cases), assessment)
        return out
