"""Per-case feature logging -- the data source for training the gate policy.

Every gate decision appends one JSONL record: the feature vector `x`, the action
taken, and (when known) the outcome. The offline counterfactual-rollout harness
(Phase 1) reads these back to build the training matrix. Kept separate from
legacy baseline's QEBLogger so the two logs stay independent.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .gate.features import FeatureVector


class FeatureLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(
        self,
        case_id: str,
        features: FeatureVector,
        action: str,
        *,
        outcome: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "case_id": case_id,
            "action": action,
            "features": features.to_dict(),
        }
        if outcome is not None:
            record["outcome"] = outcome
        if extra:
            record.update(extra)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> list[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
