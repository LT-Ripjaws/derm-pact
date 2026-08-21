"""Content-addressed cache for exact structured-generation requests."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

from ..evidence import EvidenceContract
from ..errors import CacheIntegrityError


SCHEMA_VERSION = 1


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical(asdict(value))
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("generation request contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item) for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    raise TypeError(f"unsupported generation-cache value type: {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical(value), ensure_ascii=False, allow_nan=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")


class GenerationCache:
    """Validated, atomic on-disk cache keyed by every generation input."""

    def __init__(self, root: Path | str, *, namespace: str):
        if not str(namespace).strip():
            raise ValueError("generation cache requires a non-empty model/protocol namespace")
        self.root = Path(root)
        self.namespace = str(namespace).strip()
        self._lock = threading.Lock()

    def request_key(self, kwargs: Mapping[str, Any]) -> str:
        request: Dict[str, Any] = dict(kwargs)
        image_path = request.get("image_path")
        if image_path:
            raw_paths = (
                (str(image_path),) if isinstance(image_path, (str, Path))
                else tuple(str(path) for path in image_path)
            )
            fingerprints = []
            for raw_path in raw_paths:
                path = Path(raw_path).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"generation image not found: {path}")
                fingerprints.append({
                    "resolved_path": str(path),
                    "sha256": _file_sha256(path),
                })
            request["image_path"] = fingerprints
        payload = {
            "schema_version": SCHEMA_VERSION,
            "namespace": self.namespace,
            "request": request,
        }
        return _sha256_bytes(_json_bytes(payload))

    def entry_path(self, key: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Tuple[EvidenceContract, Dict[str, Any]] | None:
        path = self.entry_path(key)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("schema version mismatch")
            if value.get("request_sha256") != key:
                raise ValueError("request digest mismatch")
            output = value.get("evidence_contract")
            if not isinstance(output, dict):
                raise ValueError("evidence_contract is missing")
            observed_output_hash = _sha256_bytes(_json_bytes(output))
            if observed_output_hash != value.get("output_sha256"):
                raise ValueError("output digest mismatch")
            contract = EvidenceContract.from_mapping(output)
            metadata = value.get("source_generation_metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("source generation metadata is malformed")
            return contract, dict(metadata)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise CacheIntegrityError(f"invalid generation cache entry {key}: {exc}") from exc

    def put(
        self, key: str, output: EvidenceContract,
        source_generation_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        path = self.entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        contract = output.to_dict()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "request_sha256": key,
            "output_sha256": _sha256_bytes(_json_bytes(contract)),
            "evidence_contract": contract,
            "source_generation_metadata": _canonical(
                dict(source_generation_metadata or {})
            ),
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as handle:
            handle.write(encoded)
            temporary = Path(handle.name)
        try:
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


class CachedStructuredGenerator:
    """Generator adapter that reuses only byte-identical evidence-contract requests."""

    def __init__(self, delegate: Any, cache: GenerationCache, *, allow_remote: bool = True):
        self.delegate = delegate
        self.cache = cache
        self.allow_remote = bool(allow_remote)
        self.last_metadata: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def _generate_contract_at_key(self, key: str, **kwargs: Any) -> EvidenceContract:
        with self._lock:
            cached = self.cache.get(key)
            if cached is not None:
                output, source_metadata = cached
                self.last_metadata = {
                    "cache_hit": True, "cache_key": key, "remote_calls": 0,
                    "failed_calls": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "total_tokens": 0,
                    "latency_seconds": 0.0, "usage_available": True,
                    "logical_remote_calls": int(source_metadata.get("remote_calls", 0) or 0),
                    "logical_prompt_tokens": int(source_metadata.get("prompt_tokens", 0) or 0),
                    "logical_completion_tokens": int(
                        source_metadata.get("completion_tokens", 0) or 0
                    ),
                    "logical_total_tokens": int(source_metadata.get("total_tokens", 0) or 0),
                    "logical_latency_seconds": float(
                        source_metadata.get("latency_seconds", 0.0) or 0.0
                    ),
                    "source_generation_metadata": source_metadata,
                }
                return output
            if not self.allow_remote:
                raise CacheIntegrityError(f"cache-only generation miss for request {key}")
            try:
                output = self.delegate.generate_contract(**kwargs)
            except Exception:
                metadata = getattr(self.delegate, "last_metadata", {})
                self.last_metadata = {
                    **(dict(metadata) if isinstance(metadata, dict) else {}),
                    "cache_hit": False, "cache_key": key, "generation_failed": True,
                }
                raise
            metadata = getattr(self.delegate, "last_metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            self.cache.put(key, output, metadata)
            self.last_metadata = {
                **dict(metadata), "cache_hit": False, "cache_key": key,
                "logical_remote_calls": int(metadata.get("remote_calls", 0) or 0),
                "logical_prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                "logical_completion_tokens": int(
                    metadata.get("completion_tokens", 0) or 0
                ),
                "logical_total_tokens": int(metadata.get("total_tokens", 0) or 0),
                "logical_latency_seconds": float(
                    metadata.get("latency_seconds", 0.0) or 0.0
                ),
            }
            return output

    def generate_contract(self, **kwargs: Any) -> EvidenceContract:
        return self._generate_contract_at_key(self.cache.request_key(kwargs), **kwargs)

    def generate_contract_with_key(
        self, key: str, *, namespace: str, **kwargs: Any,
    ) -> EvidenceContract:
        """Replay a request key pre-bound by the H2H protocol.

        The normal path hashes generic generator kwargs.  The H2H protocol has a
        stricter identity that also binds arm, code snapshot, frozen perception,
        decoding, and rendered prompt; it supplies that already-validated key.
        A namespace mismatch is rejected before the cache is read, preventing an
        arm from looking up another arm's cache directory.
        """
        if str(namespace) != self.cache.namespace:
            raise CacheIntegrityError(
                "generation cache namespace does not match the H2H request identity"
            )
        self.cache.entry_path(key)  # validates the supplied digest before use
        return self._generate_contract_at_key(key, **kwargs)

    def generate(self, **kwargs: Any) -> str:
        """Free-text compatibility path is intentionally uncached."""
        result = self.delegate.generate(**kwargs)
        metadata = getattr(self.delegate, "last_metadata", {})
        self.last_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        return result
