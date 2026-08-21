"""RAGAS field mapping and judge/version lock -- a SUPPORTING endpoint only.

RAGAS is demoted to supporting in this study (prereg v2 / spec v4): it is
Wikipedia-validated with ~0.70 annotator agreement, and its faithfulness metric
is an LLM-judge score, not a host-verifiable one.  The primary evidence endpoints
are citation integrity and evidence-contract completion, which this module never
touches.

What this module does, and deliberately does not do.  It (1) maps one H2H
arm/case output into the ``(user_input, response, retrieved_contexts,
reference)`` shape RAGAS expects, and (2) pins the judge model and the RAGAS
package version into a lock whose hash a caller can compare against a registered
value.  It does NOT call a judge here: with no judge injected, ``evaluate``
returns a mapped-but-not-executed payload, so importing or using this module can
never issue a paid judge call.  A test may inject a fake judge to exercise the
execution path; the real judge is a separate, later, authorized step the main
session owns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from medguard import config


RAGAS_ROLE = "supporting"

# The judge is version-locked because an LLM-judge score is only comparable across
# arms if every arm is judged by the same model.  The model id comes from config
# (openai/gpt-oss-20b); the version string is supplied by the caller because the
# gateway does not expose a stable judge build id through config.
DEFAULT_JUDGE_MODEL = config.JUDGE_MODEL

# The RAGAS metrics this study would report, all answer/context faithfulness style
# and all supporting.  Diagnostic accuracy, actionability, provenance integrity,
# and contract completion are scored elsewhere and are NOT RAGAS metrics.
SUPPORTING_METRICS: Tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


class RagasAdapterError(ValueError):
    """Raised when a RAGAS mapping or judge lock is inconsistent."""


def installed_ragas_version() -> Optional[str]:
    """Best-effort RAGAS version for the lock; ``None`` if RAGAS is absent.

    Imported lazily and defensively so this module has no hard RAGAS dependency
    and never fails to import in an environment without it.
    """
    try:
        import ragas  # type: ignore
    except Exception:  # noqa: BLE001 -- any import failure means "not available"
        return None
    return str(getattr(ragas, "__version__", "") or "") or None


@dataclass(frozen=True)
class RagasSample:
    """One RAGAS evaluation sample mapped from an H2H arm/case output."""

    case_id: str
    arm_id: str
    user_input: str
    response: str
    retrieved_contexts: Tuple[str, ...]
    reference: str = ""

    def __post_init__(self) -> None:
        for name in ("case_id", "arm_id", "user_input", "response"):
            if not str(getattr(self, name)).strip():
                raise RagasAdapterError(f"RAGAS sample {name} must be non-empty")
        object.__setattr__(
            self, "retrieved_contexts",
            tuple(str(item) for item in self.retrieved_contexts),
        )

    def to_ragas_row(self) -> Dict[str, Any]:
        """The exact field names the RAGAS 0.2+ single-turn sample expects."""
        row: Dict[str, Any] = {
            "user_input": self.user_input,
            "response": self.response,
            "retrieved_contexts": list(self.retrieved_contexts),
        }
        if self.reference:
            row["reference"] = self.reference
        return row


def map_arm_case_to_sample(
    *,
    case_id: str,
    arm_id: str,
    question: str,
    answer: str,
    contexts: Sequence[Mapping[str, Any] | str],
    reference: str = "",
) -> RagasSample:
    """Field-map one arm/case into a RAGAS sample.

    ``contexts`` accepts either raw strings or evidence-record mappings; a mapping
    contributes its ``text`` field.  This is the only place field names cross from
    the H2H schema into the RAGAS schema, so the mapping stays auditable in one
    spot.
    """
    rendered_contexts: list[str] = []
    for item in contexts:
        if isinstance(item, Mapping):
            text = str(item.get("text") or "").strip()
        else:
            text = str(item).strip()
        if text:
            rendered_contexts.append(text)
    return RagasSample(
        case_id=str(case_id), arm_id=str(arm_id), user_input=str(question),
        response=str(answer), retrieved_contexts=tuple(rendered_contexts),
        reference=str(reference),
    )


@dataclass(frozen=True)
class RagasJudgeLock:
    """A pinned judge identity + RAGAS version, hashable for registration."""

    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_model_version: str = "unpinned"
    metrics: Tuple[str, ...] = SUPPORTING_METRICS
    ragas_version: Optional[str] = field(default_factory=installed_ragas_version)

    def __post_init__(self) -> None:
        if not str(self.judge_model).strip():
            raise RagasAdapterError("judge_model must be non-empty")
        if not self.metrics:
            raise RagasAdapterError("at least one supporting metric must be locked")
        object.__setattr__(self, "metrics", tuple(str(m) for m in self.metrics))

    def payload(self) -> Dict[str, Any]:
        return {
            "role": RAGAS_ROLE,
            "judge_model": self.judge_model,
            "judge_model_version": self.judge_model_version,
            "metrics": list(self.metrics),
            "ragas_version": self.ragas_version,
        }

    @property
    def lock_sha256(self) -> str:
        raw = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class RagasSupportingAdapter:
    """Maps and (only if a judge is injected) scores RAGAS supporting metrics.

    The default construction has ``judge=None`` and therefore cannot make a judge
    call.  A test injects a fake judge exposing ``model`` and
    ``evaluate(rows, metrics)``; the adapter checks the fake's ``model`` against
    the lock before using it, so an arm can never be judged by an unlocked model.
    """

    def __init__(self, judge_lock: Optional[RagasJudgeLock] = None, *, judge: Any = None):
        self.lock = judge_lock or RagasJudgeLock()
        self.judge = judge

    def evaluate(self, samples: Sequence[RagasSample]) -> Dict[str, Any]:
        rows = [sample.to_ragas_row() for sample in samples]
        base: Dict[str, Any] = {
            "role": RAGAS_ROLE,
            "judge_lock": self.lock.payload(),
            "judge_lock_sha256": self.lock.lock_sha256,
            "n_samples": len(rows),
            "sample_rows": rows,
        }
        if self.judge is None:
            base["status"] = "mapped_not_executed"
            base["scores"] = None
            return base
        judge_model = str(getattr(self.judge, "model", "") or "")
        if judge_model != self.lock.judge_model:
            raise RagasAdapterError(
                f"injected judge model {judge_model!r} differs from the locked "
                f"judge {self.lock.judge_model!r}"
            )
        base["status"] = "executed"
        base["scores"] = self.judge.evaluate(rows, self.lock.metrics)
        return base
