"""WP-D: typed graph state."""
from __future__ import annotations

import hashlib
import json
import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class PrototypeState(TypedDict, total=False):
    run_id: str
    case_id: str
    scenario: str
    mode: str  # "offline" | "live-deferred"
    inference: str  # "cached" | "worker"
    output_dir: str
    session_id: str
    actor: str

    record_version: int
    record_sha256: str

    request: str
    ask_first: bool
    route: str
    pending_interrupt: Optional[Dict[str, Any]]

    turn_result: Optional[Dict[str, Any]]
    checker_ok: bool
    live_status: Optional[str]

    clarification: Optional[Dict[str, Any]]
    memory: Optional[Dict[str, Any]]
    xai: Optional[Dict[str, Any]]
    episode: Optional[Dict[str, Any]]

    report_paths: Dict[str, str]
    terminal_state: str
    error: Optional[str]

    trace: Annotated[List[Dict[str, Any]], operator.add]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def trace_entry(node: str, *, record_version: Optional[int], inputs: Any, outputs_summary: Any) -> Dict[str, Any]:
    """One append-only trace entry: ``{node, at, record_version, inputs_sha256, outputs_summary}``."""
    return {
        "node": node,
        "at": now_iso(),
        "record_version": record_version,
        "inputs_sha256": sha256_of(inputs),
        "outputs_summary": outputs_summary,
    }


__all__ = ["PrototypeState", "now_iso", "sha256_of", "trace_entry"]
