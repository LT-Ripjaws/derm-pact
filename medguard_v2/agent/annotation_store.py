"""Deployment-time annotation capture: every clinician override IS a per-criterion label.

Why this exists
---------------
The world's supply of expert 7-point criterion labels is tiny -- Derm7pt's train split holds 413
cases, and that scarcity is the field's bottleneck. Yet the deployed system already contains a
labelling event: when a clinician calls ``override_criterion`` and says "criterion X is VALUE, you
are wrong", that is exactly one expert-verified (case, criterion, value) annotation, produced as a
side effect of routine use. Today the override changes the score and evaporates. This module makes
it durable: an append-only JSONL log of every successful override, validated against the same
criterion vocabularies the aggregator trusts, replayable into label updates a future head re-fit
can consume.

What it deliberately is NOT
---------------------------
* Not a change to any decision path. The store is opt-in (``AgentTools`` takes it as an optional
  constructor argument, default ``None``); when absent, the agent's behaviour is byte-identical.
* Not a training loop. It captures and replays labels; nothing here re-fits a head. The measured
  value of accumulated annotations lives in
  ``scripts/dev/build_annotation_learning_curve.py`` -> ``annotation_learning_curve.json``.
* Not a claim that clinician labels equal dataset gold. Real overrides are noisier than Derm7pt's
  consensus annotations (interobserver kappa on dermoscopic criteria is fair-to-moderate); the
  store records provenance per annotation so a consumer can weigh sources.

Integrity rules
---------------
* Append-only: ``record`` opens the file in append mode and writes one JSON line; nothing here
  rewrites or deletes existing lines.
* Validated on the way IN (``record`` rejects unknown criteria, out-of-vocabulary values, empty
  ids) and on the way OUT (``load`` refuses a tampered or hand-mangled file rather than silently
  skipping lines) -- garbage is never silently stored or served.
* ``INDETERMINATE`` is accepted as a corrected value (it is a first-class override: "cannot
  judge"), but it is an abstention, not a label -- ``as_label_updates`` excludes those pairs by
  default, and a FINAL indeterminate suppresses any earlier concrete value for that pair (the
  clinician's last word was "cannot judge", so no label survives).
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from medguard_v2.criteria.schema import CRITERIA_BY_NAME, CRITERION_NAMES, INDETERMINATE

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default location, alongside the other derm7pt_v1 evidence artefacts. Constructor-injectable so
#: tests and alternative deployments never touch the real log.
DEFAULT_PATH = REPO_ROOT / "research" / "evidence" / "derm7pt_v1" / "override_annotations.jsonl"

#: Stamped on every line and required back on load, so a foreign or future-format file is refused
#: loudly instead of being half-understood.
RECORD_PROTOCOL = "override-annotation-v1"

#: Receipt keys are trimmed to SCALAR values only ("minimal contract provenance"): enough to say
#: which perception produced the prior value (split, feature_set, seed, fit hyperparameters as
#: flat values), without dragging nested config blobs into every annotation line.
_SCALAR_TYPES = (str, int, float, bool)

_REQUIRED_KEYS = (
    "protocol",
    "recorded_at",
    "case_id",
    "criterion",
    "corrected_value",
    "prior_value",
    "source",
    "contract_provenance",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_criterion_value(criterion: str, value: Any, *, field: str, allow_none: bool) -> None:
    """Reject anything outside ``vocabulary | {INDETERMINATE}`` for this criterion."""
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{field} must not be None")
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    spec = CRITERIA_BY_NAME[criterion]
    if value != INDETERMINATE and value not in spec.vocabulary:
        raise ValueError(
            f"{criterion}: {field} {value!r} is not a permitted value; "
            f"allowed: {sorted(spec.vocabulary)} or {INDETERMINATE!r}"
        )


def _validate_record(record: Mapping[str, Any], *, where: str) -> None:
    """Full validation of one record dict -- shared by ``record`` (before writing) and ``load``
    (after reading), so the file can never hold a line ``load`` would not vouch for."""
    missing = [k for k in _REQUIRED_KEYS if k not in record]
    if missing:
        raise ValueError(f"{where}: missing keys {missing}")
    if record["protocol"] != RECORD_PROTOCOL:
        raise ValueError(
            f"{where}: unknown protocol {record['protocol']!r} (expected {RECORD_PROTOCOL!r})"
        )
    case_id = record["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{where}: case_id must be a non-empty string, got {case_id!r}")
    criterion = record["criterion"]
    if criterion not in CRITERION_NAMES:
        raise ValueError(
            f"{where}: unknown criterion {criterion!r}; valid: {list(CRITERION_NAMES)}"
        )
    _validate_criterion_value(
        criterion, record["corrected_value"], field="corrected_value", allow_none=False
    )
    _validate_criterion_value(
        criterion, record["prior_value"], field="prior_value", allow_none=True
    )
    source = record["source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{where}: source must be a non-empty string, got {source!r}")
    provenance = record["contract_provenance"]
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError(
            f"{where}: contract_provenance must be an object or null, got "
            f"{type(provenance).__name__}"
        )


def _minimal_provenance(contract_receipt: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Trim a contract receipt to its scalar entries; ``None`` when nothing usable is given."""
    if not contract_receipt:
        return None
    trimmed = {
        str(k): v for k, v in contract_receipt.items() if isinstance(v, _SCALAR_TYPES)
    }
    return trimmed or None


class AnnotationStore:
    """Append-only JSONL store of clinician override annotations."""

    def __init__(self, path: Union[str, Path, None] = None):
        self.path = Path(path) if path is not None else DEFAULT_PATH

    # -- capture -------------------------------------------------------------

    def record(
        self,
        case_id: str,
        criterion: str,
        corrected_value: str,
        prior_value: Optional[str],
        *,
        source: str = "clinician_override",
        contract_receipt: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate and append ONE annotation line; returns the stored record.

        Validation happens entirely before the file is opened, so a rejected annotation leaves
        the log untouched -- invalid input is refused, never stored.
        """
        entry: Dict[str, Any] = {
            "protocol": RECORD_PROTOCOL,
            "recorded_at": _utc_now(),
            "case_id": case_id,
            "criterion": criterion,
            "corrected_value": corrected_value,
            "prior_value": prior_value,
            "source": source,
            "contract_provenance": _minimal_provenance(contract_receipt),
        }
        _validate_record(entry, where="record()")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    # -- replay --------------------------------------------------------------

    def load(self) -> List[Dict[str, Any]]:
        """All records in append (chronological) order; ``[]`` when the file does not exist.

        Every line is re-validated: a tampered or truncated file raises ``ValueError`` naming the
        offending line, rather than silently serving a subset.
        """
        if not self.path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.path}:{lineno}: not valid JSON ({exc})") from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{self.path}:{lineno}: line is {type(record).__name__}, not an object"
                    )
                _validate_record(record, where=f"{self.path}:{lineno}")
                records.append(record)
        return records

    def as_label_updates(
        self, *, include_indeterminate: bool = False
    ) -> Dict[str, Dict[str, str]]:
        """``{case_id: {criterion: value}}`` with the LAST write per (case, criterion) winning.

        ``INDETERMINATE`` finals are dropped by default: an override to "cannot judge" is an
        abstention, not a trainable label -- and because last-write-wins is applied FIRST, a final
        indeterminate also suppresses any earlier concrete value for that pair.
        """
        updates: Dict[str, Dict[str, str]] = {}
        for record in self.load():
            updates.setdefault(record["case_id"], {})[record["criterion"]] = (
                record["corrected_value"]
            )
        if include_indeterminate:
            return updates
        filtered = {
            case_id: {c: v for c, v in values.items() if v != INDETERMINATE}
            for case_id, values in updates.items()
        }
        return {case_id: values for case_id, values in filtered.items() if values}

    def summary(self) -> Dict[str, Any]:
        """Counts a paper table or a monitoring dashboard would want."""
        records = self.load()
        pairs = {(r["case_id"], r["criterion"]) for r in records}
        label_updates = self.as_label_updates()
        return {
            "path": str(self.path),
            "n_records": len(records),
            "n_cases": len({r["case_id"] for r in records}),
            "n_case_criterion_pairs": len(pairs),
            "n_label_updates": sum(len(v) for v in label_updates.values()),
            "per_criterion": dict(Counter(r["criterion"] for r in records)),
            "per_source": dict(Counter(r["source"] for r in records)),
        }
