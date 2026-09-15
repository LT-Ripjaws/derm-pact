"""Deep perception seam for the DermPact Phase 2.4 model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Mapping, TypeVar


_RecordT = TypeVar("_RecordT")

@dataclass(frozen=True)
class ObservedCase:
    """Inputs supplied for one DermPact assessment."""

    case_id: str
    dermoscopic_image: Any
    clinical_image: Any | None = None
    clinical_image_present: bool = False
    clinical_facts: Mapping[str, object] = field(default_factory=dict)


class DermPactPerception(Generic[_RecordT]):
    """Validate observed inputs, then delegate inference behind one small interface."""

    def __init__(self, inference_adapter: Callable[[ObservedCase], _RecordT]) -> None:
        if not callable(inference_adapter):
            raise TypeError("inference_adapter must be callable")
        self._inference_adapter = inference_adapter

    def assess(self, case: ObservedCase) -> _RecordT:
        if not isinstance(case, ObservedCase):
            raise TypeError("case must be an ObservedCase")
        if not case.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if case.dermoscopic_image is None:
            raise ValueError("one dermoscopic image is required")
        observed_clinical_image = case.clinical_image is not None
        if case.clinical_image_present is not observed_clinical_image:
            raise ValueError(
                "clinical image presence mask must match the observed clinical image"
            )
        return self._inference_adapter(case)
