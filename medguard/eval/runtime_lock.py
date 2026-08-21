"""The pre-sealed runtime lock: everything that must not move once the sealed run starts.

The design freeze (EXPERIMENT_SPEC_V6) fixes *what* will be measured.  This fixes *what
will do the measuring*: the source tree, the prompts, the decoding parameters, the
scorers, the frozen artifacts, the abstention gate and the judge specification.

Why a second freeze at all.  Between the design freeze and the sealed run there is a
window in which a well-meaning edit -- a prompt tweak, a re-fitted threshold, a scorer
"fix" -- would silently change what the sealed numbers mean, and nothing in the design
freeze would notice.  The lock closes that window by hashing the runtime and refusing to
proceed if any of it moved.

Two operations, and the asymmetry is deliberate:

* :func:`build_lock` writes the lock.  It is run ONCE, before the sealed run.
* :func:`verify_lock` checks the current tree against it.  It is run immediately before
  the sealed run and again when the results are scored, and it **raises** rather than
  warning: a drifted runtime invalidates the run, and a warning is something a tired
  person scrolls past at 2am.

The lock deliberately does NOT hash the results ledger, the notes, or the evidence
directory.  Those are append-only working records; pinning them would guarantee the
lock breaks for reasons that have nothing to do with the runtime.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class RuntimeLockError(RuntimeError):
    """Raised when the runtime has drifted from the lock."""


#: Source files whose behaviour determines a sealed number.  Everything that builds a
#: contract, retrieves evidence, generates, verifies, or scores.
LOCKED_SOURCE_GLOBS: Sequence[str] = (
    "medguard/evidence/*.py",
    "medguard/pipeline/*.py",
    "medguard/eval/*.py",
    "medguard/agents/*.py",
    "medguard/config.py",
)

#: Frozen inputs.  A changed manifest or corpus silently changes the population.
LOCKED_ARTIFACTS: Sequence[str] = (
    "research/manifests/ham10000_split_v2.json",
    "research/manifests/medguard_h2h_development_v1.json",
    "medguard/data/corpus/derm_guideline_recommendations_v1.jsonl",
    "medguard/data/corpus/derm_guideline_inventory_v1.json",
)

#: Implementation cards become design-binding at this lock, per the v5/v6 receipts.
LOCKED_CARDS: Sequence[str] = (
    "research/implementation_cards/C1_EVIDENCE_CONTRACT_CURRENT.json",
    "research/implementation_cards/C2_RETRIEVAL_COMPONENT_V1.json",
    "research/implementation_cards/H2H_IMPLEMENTATION_CARDS_V1.json",
)

#: Excluded on purpose: append-only records and working notes.  Pinning them would make
#: the lock break for reasons unrelated to the runtime, which trains people to ignore it.
DELIBERATELY_NOT_LOCKED: Mapping[str, str] = {
    "research/RESULTS_LEDGER.json": "append-only; new records are expected",
    "research/RIVAL_LEDGER.json": "reference-only positioning, touches no runtime path",
    "notes/": "working notes",
    "research/evidence/": "run outputs, written BY the locked runtime",
    "medguard/tests/": "tests cannot change a sealed number; locking them would block "
                       "adding a test that catches a bug before the run",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect(root: Path, globs: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            rows.append({"path": rel, "bytes": path.stat().st_size,
                         "sha256": _sha256(path)})
    return rows


def _tree_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """One hash over the whole locked set, so drift anywhere shows up in one field."""
    joined = "\n".join(f"{r['path']}:{r['sha256']}" for r in rows)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_lock(
    root: Path,
    *,
    abstention_gate_path: str,
    judge_specification_sha256: str,
    spec_id: str,
    identical_arms_for_noise_floor: Sequence[str],
    created: str,
) -> Dict[str, Any]:
    """Hash the runtime that will produce the sealed numbers.

    ``abstention_gate_path`` is recorded as a path plus a hash on purpose: the gate must
    be LOADED at sealed time, never re-fitted. A re-fit would tune the operating point on
    the sealed distribution, which is the exact failure the sealed protocol exists to
    prevent.
    """
    root = Path(root)
    gate = root / abstention_gate_path
    if not gate.is_file():
        raise RuntimeLockError(
            f"abstention gate {abstention_gate_path} not found; the sealed run must load "
            f"a frozen gate, never fit one"
        )
    source = _collect(root, LOCKED_SOURCE_GLOBS)
    artifacts = _collect(root, LOCKED_ARTIFACTS)
    cards = _collect(root, LOCKED_CARDS)
    if not source:
        raise RuntimeLockError("no source files matched the lock globs")
    missing = [p for p in LOCKED_ARTIFACTS if not (root / p).is_file()]
    if missing:
        raise RuntimeLockError(f"locked artifacts missing: {missing}")

    return {
        "schema_version": "medguard-runtime-lock-v1",
        "spec_id": spec_id,
        "created": created,
        "purpose": (
            "Fixes what will DO the measuring for the sealed run. The design freeze fixes "
            "what will be measured; this fixes the machinery."
        ),
        "source_tree": {
            "files": source,
            "n_files": len(source),
            "digest": _tree_digest(source),
            "globs": list(LOCKED_SOURCE_GLOBS),
        },
        "frozen_artifacts": {"files": artifacts, "digest": _tree_digest(artifacts)},
        "implementation_cards": {
            "files": cards, "digest": _tree_digest(cards),
            "note": "promoted to design-binding at this lock, per the v6 receipt",
        },
        "abstention_gate": {
            "path": abstention_gate_path,
            "sha256": _sha256(gate),
            "rule": (
                "LOAD this file at sealed time. Re-fitting the gate on sealed confidences "
                "would tune the operating point on the confirmatory set."
            ),
        },
        "scorers": {
            "ragas_judge_specification_sha256": judge_specification_sha256,
            "judge_model_version": "unpinned -- the gateway exposes no build id",
            "judge_nondeterminism": (
                "The judge disagrees with itself at temperature 0. The sealed run MUST "
                "re-measure the test-retest floor from the identical-arm group below and "
                "report it with the results; the development floor may not be carried over."
            ),
            "identical_arms_for_noise_floor": list(identical_arms_for_noise_floor),
        },
        "not_locked": dict(DELIBERATELY_NOT_LOCKED),
        "verification_rule": (
            "verify_lock() runs immediately before the sealed run and again when the "
            "results are scored. It raises on drift. A sealed run whose runtime does not "
            "match this lock is void."
        ),
        "authorization": (
            "This lock authorizes nothing. The sealed run remains a separate, explicit, "
            "single-use user authorization."
        ),
    }


def verify_lock(root: Path, lock: Mapping[str, Any]) -> Dict[str, Any]:
    """Check the tree against the lock. Raises :class:`RuntimeLockError` on drift."""
    root = Path(root)
    drift: Dict[str, List[str]] = {"changed": [], "missing": [], "added": []}

    for section, globs in (
        ("source_tree", LOCKED_SOURCE_GLOBS),
        ("frozen_artifacts", LOCKED_ARTIFACTS),
        ("implementation_cards", LOCKED_CARDS),
    ):
        locked = {row["path"]: row["sha256"] for row in lock[section]["files"]}
        current = {row["path"]: row["sha256"] for row in _collect(root, globs)}
        for path, digest in locked.items():
            if path not in current:
                drift["missing"].append(path)
            elif current[path] != digest:
                drift["changed"].append(path)
        # A NEW file under a locked glob can change behaviour just as much as an edit.
        for path in current:
            if path not in locked:
                drift["added"].append(path)

    gate = root / lock["abstention_gate"]["path"]
    if not gate.is_file():
        drift["missing"].append(lock["abstention_gate"]["path"])
    elif _sha256(gate) != lock["abstention_gate"]["sha256"]:
        drift["changed"].append(lock["abstention_gate"]["path"])

    if any(drift.values()):
        raise RuntimeLockError(
            "runtime drifted from the lock; the sealed run would not measure what was "
            f"frozen. changed={drift['changed']} missing={drift['missing']} "
            f"added={drift['added']}"
        )
    return {"verified": True, "source_files": lock["source_tree"]["n_files"],
            "source_digest": lock["source_tree"]["digest"]}


def verify_judge_specification(lock: Mapping[str, Any], current_sha256: str) -> None:
    """The judge is a measuring instrument; a changed one invalidates comparability."""
    locked = lock["scorers"]["ragas_judge_specification_sha256"]
    if locked != current_sha256:
        raise RuntimeLockError(
            f"RAGAS judge specification changed since the lock ({locked[:12]} -> "
            f"{current_sha256[:12]}); sealed scores would not be comparable to the "
            f"registered instrument"
        )
