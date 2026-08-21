"""Bounded action space for the new thesis; HANDOFF is intentionally absent."""
from enum import Enum


class Action(str, Enum):
    PASS = "PASS"
    EXPAND = "EXPAND"
    REFORMULATE = "REFORMULATE"
    INDEPENDENT_SOURCE = "INDEPENDENT_SOURCE"
    ABSTAIN = "ABSTAIN"

    @property
    def terminal(self) -> bool:
        return self in (Action.PASS, Action.ABSTAIN)
