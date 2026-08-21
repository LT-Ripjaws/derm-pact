"""One-shot guard and access log for the sealed HAM10000 test partition.

The preregistration allows the sealed test to be evaluated **once**, and requires a
sealed-test access log among the freeze requirements.  Both are enforced here rather
than left to discipline, because "we only ran it once" is not a claim anyone can check
afterwards unless the code refused to run it twice.

How it works
------------
Every sealed read goes through :func:`open_sealed_access`, which appends an immutable,
hash-chained entry to ``research/SEALED_ACCESS_LOG.jsonl`` before returning.  A second
attempt to open the same ``purpose`` raises :class:`SealedAccessExhausted` unless the
caller passes an explicit override token, which is itself logged as an override.

The log is hash-chained: each entry carries the digest of the previous one, so a
deleted or edited entry breaks the chain and :func:`verify_log` reports it.  Deleting
the whole file is still possible -- nothing stops a determined author -- but it cannot
be done silently, because the runtime lock pins this module and the manuscript is
expected to quote the chain head.

What this deliberately does NOT do
----------------------------------
It does not stop you reading the sealed manifest with a text editor.  It is a guard for
the *evaluation pipeline*, not a security boundary, and the honest framing in the paper
is "the pipeline refuses a second sealed evaluation and logs the first", not "the sealed
set was cryptographically sealed".
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "research" / "SEALED_ACCESS_LOG.jsonl"

#: Purposes the protocol anticipates.  A purpose not in this set is allowed but is
#: recorded as ``registered=False`` so an unplanned sealed read is visible in the log.
REGISTERED_PURPOSES = (
    "sealed_perception_freeze",   # encode test images with the frozen probe
    "sealed_h2h_run",             # the single paid evaluation
    "sealed_scoring",             # truth join for the scorer, after the run
)

#: Purposes that may be opened repeatedly WITHOUT an override -- but are still logged
#: every single time.
#:
#: One-shot exists to protect a *claim*: the confirmatory evaluation may happen once
#: because reporting the best of several runs would be cherry-picking. A pipeline smoke
#: test carries no claim -- it reports nothing -- so making it one-shot adds friction
#: with no protective value, and worse, it pushes an operator toward routine overrides,
#: which is exactly the habit that makes a real override unremarkable.
#:
#: Every smoke access is still appended to the chain, so "how many times did you touch
#: the sealed set while debugging" remains answerable from the log.
REPEATABLE_PURPOSES = (
    "sealed_smoke", "sealed_smoke_scoring",
    # Resuming an interrupted evaluation is CONTINUING it, not starting a second one.
    # Forcing an override here would stamp override=True against the confirmatory
    # purpose -- the one entry a reviewer looks hardest at -- to record something
    # entirely innocent. A distinct, repeatable purpose keeps sealed_h2h_run at exactly
    # one clean entry and records each continuation as what it is.
    "sealed_h2h_resume", "sealed_scoring_resume",
)

#: A resume purpose is only meaningful if the thing it resumes was actually started.
#: Opening one without its parent would let a run masquerade as a continuation of an
#: evaluation that never happened.
RESUME_PARENTS = {
    "sealed_h2h_resume": "sealed_h2h_run",
    "sealed_scoring_resume": "sealed_scoring",
}

GENESIS = "0" * 64


class SealedAccessError(RuntimeError):
    """Base class for sealed-access failures."""


class SealedAccessExhausted(SealedAccessError):
    """Raised when a purpose has already been used and no override was supplied."""


class SealedLogTampered(SealedAccessError):
    """Raised when the hash chain does not verify."""


@dataclass(frozen=True)
class SealedAccessGrant:
    """Proof that a sealed read was logged.  Carries the chain head it created."""

    purpose: str
    sequence: int
    entry_sha256: str
    timestamp: str
    override: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "purpose": self.purpose, "sequence": self.sequence,
            "entry_sha256": self.entry_sha256, "timestamp": self.timestamp,
            "override": self.override,
        }


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_log(log_path: str | Path = DEFAULT_LOG) -> List[Dict[str, Any]]:
    """Every logged sealed access, oldest first.  Missing file means never accessed."""
    path = Path(log_path)
    if not path.is_file():
        return []
    entries: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def verify_log(log_path: str | Path = DEFAULT_LOG) -> Dict[str, Any]:
    """Walk the hash chain.  Reports the head, or the first index that breaks it."""
    entries = read_log(log_path)
    previous = GENESIS
    for index, entry in enumerate(entries):
        if entry.get("previous_sha256") != previous:
            return {"valid": False, "broken_at": index, "n_entries": len(entries),
                    "reason": "previous_sha256 does not match the preceding entry"}
        body = {k: v for k, v in entry.items() if k != "entry_sha256"}
        if _digest(body) != entry.get("entry_sha256"):
            return {"valid": False, "broken_at": index, "n_entries": len(entries),
                    "reason": "entry digest does not match its contents"}
        previous = str(entry["entry_sha256"])
    return {"valid": True, "n_entries": len(entries), "head_sha256": previous,
            "purposes_used": sorted({str(e["purpose"]) for e in entries})}


def purposes_used(log_path: str | Path = DEFAULT_LOG) -> Dict[str, int]:
    """How many times each purpose has been opened."""
    counts: Dict[str, int] = {}
    for entry in read_log(log_path):
        key = str(entry["purpose"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def open_sealed_access(
    purpose: str,
    *,
    detail: str,
    log_path: str | Path = DEFAULT_LOG,
    override_token: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> SealedAccessGrant:
    """Log a sealed read and return the grant, or refuse if the purpose is spent.

    ``detail`` is a free-text description of what is about to be read and why; it is
    written verbatim into the log, so it is the thing a reader will judge the run by.

    ``override_token`` must equal ``"OVERRIDE:<purpose>"`` to re-open a spent purpose.
    Requiring the purpose to be spelled out again is deliberate: it makes an override
    impossible to perform by reflex, and every override is logged as one.
    """
    if not str(purpose).strip():
        raise SealedAccessError("purpose must be non-empty")
    if not str(detail).strip():
        raise SealedAccessError("detail must be non-empty: an unexplained sealed read "
                                "is exactly what the log exists to prevent")
    chain = verify_log(log_path)
    if not chain["valid"]:
        raise SealedLogTampered(
            f"sealed access log fails verification at entry {chain['broken_at']}: "
            f"{chain['reason']}. Refusing to extend a broken chain."
        )
    counts = purposes_used(log_path)
    parent = RESUME_PARENTS.get(purpose)
    if parent is not None and not counts.get(parent):
        raise SealedAccessError(
            f"cannot open {purpose!r}: its parent {parent!r} was never opened. A resume "
            f"continues an evaluation that started; it cannot stand in for one that did "
            f"not."
        )
    used = counts.get(purpose, 0)
    override = False
    if used and purpose in REPEATABLE_PURPOSES:
        pass  # repeatable by design; still logged below
    elif used:
        expected = f"OVERRIDE:{purpose}"
        if override_token != expected:
            raise SealedAccessExhausted(
                f"sealed purpose {purpose!r} was already used {used} time(s). The "
                f"protocol allows the sealed test to be evaluated ONCE. To override "
                f"deliberately, pass override_token={expected!r}; the override will be "
                f"recorded in the log as such."
            )
        override = True

    entries = read_log(log_path)
    body: Dict[str, Any] = {
        "sequence": len(entries),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
        "registered_purpose": purpose in REGISTERED_PURPOSES,
        "detail": str(detail),
        "override": override,
        "previous_sha256": chain.get("head_sha256", GENESIS) if entries else GENESIS,
        "extra": dict(extra or {}),
    }
    body["entry_sha256"] = _digest(body)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(body, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return SealedAccessGrant(
        purpose=purpose, sequence=int(body["sequence"]),
        entry_sha256=str(body["entry_sha256"]), timestamp=str(body["timestamp"]),
        override=override,
    )


def assert_never_accessed(
    purpose: str = "sealed_h2h_run", log_path: str | Path = DEFAULT_LOG,
) -> None:
    """Guard for development code paths that must never touch the sealed set."""
    if purposes_used(log_path).get(purpose):
        raise SealedAccessExhausted(
            f"sealed purpose {purpose!r} has already been used; this code path claims "
            f"the sealed set is untouched and that is no longer true"
        )
