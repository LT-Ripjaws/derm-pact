"""Dependency-free DERM-PACT exception types."""


class StructuredOutputError(ValueError):
    """Generated clinical content failed the host-side output contract."""


class CacheIntegrityError(ValueError):
    """A content-addressed generation cache entry failed validation."""


class LLMCallError(RuntimeError):
    """Remote generation failed after retries, with non-secret usage telemetry."""

    def __init__(self, message: str, *, usage_metadata=None):
        super().__init__(message)
        self.usage_metadata = dict(usage_metadata or {})
