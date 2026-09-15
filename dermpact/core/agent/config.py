"""WP-AG: ``WORKFLOW_CONFIG`` -- the one versioned set of bounded-agent budgets."""
from __future__ import annotations

from dataclasses import dataclass

from dermpact.core.versions import WORKFLOW_CONFIG_VERSION


@dataclass(frozen=True)
class WorkflowConfig:
    simple_route_max_tools: int = 3
    complex_route_max_tools: int = 6
    max_turns: int = 8
    max_duplicate_calls: int = 1
    max_explanation_repairs: int = 1
    pending_confirmations: int = 1
    provider_timeout_s: float = 60.0
    provider_retries: int = 2
    config_version: str = WORKFLOW_CONFIG_VERSION

    def to_dict(self) -> dict:
        return {
            "simple_route_max_tools": self.simple_route_max_tools,
            "complex_route_max_tools": self.complex_route_max_tools,
            "max_turns": self.max_turns,
            "max_duplicate_calls": self.max_duplicate_calls,
            "max_explanation_repairs": self.max_explanation_repairs,
            "pending_confirmations": self.pending_confirmations,
            "provider_timeout_s": self.provider_timeout_s,
            "provider_retries": self.provider_retries,
            "config_version": self.config_version,
        }


#: The single shared configuration instance every module in this package imports.
WORKFLOW_CONFIG = WorkflowConfig()

__all__ = ["WorkflowConfig", "WORKFLOW_CONFIG"]
