"""WP-D: host services bundle."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from dermpact.guidelines.reviewed_v2 import ReviewedApplicabilityEngine
from dermpact.reliability.bundle import ReliabilityBundle
from dermpact.pipeline.guideline_retriever import FrozenGuidelineRetriever, GuidelineCatalogError

from dermpact.core import memory as vcm2
from dermpact.core import versions as V
from dermpact.core.episode import EpisodeEvent
from dermpact.core.inference_client import CachedInferenceClient, WorkerClient
from dermpact.core.ledger_v2 import AmendmentLedgerV2
from dermpact.core.record_v2 import StructuredCaseRecordV2
from dermpact.core.store import CaseState, CaseStore, StoreErrorV2
from dermpact.core.agent.tools import ToolContext
from dermpact.core.prototype import inventory as inv

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATE_DIR = ROOT / "research/experiments/dermpact_phase6/prototype_state_v2"
DEFAULT_MEMORY_BANK_DIR = ROOT / "research/experiments/dermpact_phase6/memory_bank_dev_v1"
DEFAULT_GUIDELINE_LIBRARY = ROOT / "dermpact/data/corpus/derm_guideline_recommendations_v1.jsonl"

MODES = ("offline", "live-deferred", "live")
INFERENCE_KINDS = ("cached", "worker")


class HostBuildError(RuntimeError):
    """A HostServices.build() dependency (bundle/catalogue/bank) failed to load."""


class _CaseAwareInferenceClient:
    """Wraps the real inference client so a handful of fixture case ids can be served without ever resolving to a real image path."""

    def __init__(self, base: Any, overrides: Mapping[str, Dict[str, Any]]):
        self._base = base
        self._overrides = dict(overrides)

    def predict(self, *, case_id: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        if case_id in self._overrides:
            return self._overrides[case_id]
        return self._base.predict(case_id=case_id, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._base, item)


class _WorkerCaseClient:
    """Adapts :class:`WorkerClient` (which predicts from explicit file paths) to the ``predict(case_id=...)`` surface :mod:`agent.tools` expects, resolving paths through the same allowlisted-case resolution :mod:`inference_client` uses."""

    def __init__(self, worker: WorkerClient, resolve_paths):
        self._worker = worker
        self._resolve_paths = resolve_paths

    def predict(self, *, case_id: Optional[str] = None, **_ignored: Any) -> Dict[str, Any]:
        if not case_id:
            raise ValueError("case_id required")
        paths = self._resolve_paths(case_id)
        result = self._worker.predict(
            case_id=case_id, dermoscopic_path=paths["dermoscopic_path"], clinical_path=paths.get("clinical_path"),
            expected_dermoscopic_sha256=paths["dermoscopic_sha256"], expected_clinical_sha256=paths.get("clinical_sha256"),
        )
        return {"prediction": result}

    def __getattr__(self, item: str) -> Any:
        return getattr(self._worker, item)


@dataclass
class HostServices:
    state_dir: Path
    inventory_dir: Path
    mode: str
    inference: str
    catalogue: Any
    engine: Any
    guideline_retriever: FrozenGuidelineRetriever
    bundle: ReliabilityBundle
    store: CaseStore
    inference_client: Any
    memory_bank: Optional[vcm2.MemoryBank]
    memory_policy: Optional[vcm2.ComparisonPolicy]
    jobs: Any
    manifest: Optional[inv.InventoryManifest]
    provider_factory: Optional[Callable[[str, Any], Any]] = field(default=None, repr=False)
    live_model: Optional[str] = field(default=None, repr=False)
    gate: Any = field(default=None, repr=False)
    _app: Any = field(default=None, repr=False)
    _checkpoint_conn: Any = field(default=None, repr=False)
    _reserved_cases_cache: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _revoked_overlay_bank: Optional[vcm2.MemoryBank] = field(default=None, repr=False)

    # -- construction -------------------------------------------------------

    @classmethod
    def build(
        cls, state_dir: Any, *, inventory_dir: Any = inv.DEFAULT_INVENTORY_DIR, mode: str = "offline",
        inference: str = "cached", memory_bank_dir: Any = DEFAULT_MEMORY_BANK_DIR,
    ) -> "HostServices":
        if mode not in MODES:
            raise HostBuildError(f"unknown mode {mode!r}; expected one of {MODES}")
        if inference not in INFERENCE_KINDS:
            raise HostBuildError(f"unknown inference {inference!r}; expected one of {INFERENCE_KINDS}")
        engine = ReviewedApplicabilityEngine()
        catalogue = engine.catalogue
        try:
            library_sha = Path(str(DEFAULT_GUIDELINE_LIBRARY) + ".sha256").read_text(encoding="ascii").strip()
            guideline_retriever = FrozenGuidelineRetriever.load(DEFAULT_GUIDELINE_LIBRARY, expected_sha256=library_sha)
        except (OSError, GuidelineCatalogError) as exc:
            raise HostBuildError(f"could not load the hash-pinned 66-record guideline library: {exc}") from exc
        bundle = ReliabilityBundle(ROOT / V.RELIABILITY_BUNDLE)
        store = CaseStore(state_dir, catalogue=catalogue, engine=engine, calibrators=bundle.calibration, bundle=bundle)

        try:
            manifest = inv.load_manifest(inventory_dir)
        except FileNotFoundError:
            manifest = None

        from dermpact.core.prototype.jobs import JobCoordinator
        jobs = JobCoordinator()

        from dermpact.core.gate_v2 import VerificationGateV2
        host = cls(
            state_dir=Path(state_dir), inventory_dir=Path(inventory_dir), mode=mode, inference=inference,
            catalogue=catalogue, engine=engine, guideline_retriever=guideline_retriever,
            bundle=bundle, store=store, inference_client=None,
            memory_bank=None, memory_policy=None, jobs=jobs, manifest=manifest,
            gate=VerificationGateV2(catalogue, engine, calibrators=bundle.calibration, bundle=bundle),
        )

        overrides: Dict[str, Dict[str, Any]] = {}
        if inference == "cached":
            base_client = CachedInferenceClient(inventory_dir)
        else:
            base_client = _WorkerCaseClient(WorkerClient(), host.resolve_case_image_paths)
        try:
            base_result = base_client.predict(case_id=inv.BASE_FIXTURE_CASE_ID)
            base_embedding = base_result["prediction"]["sentinel_embedding"]
            overrides[inv.FIXTURE_MEMORY_NO_MATCH] = {
                "prediction": {**base_result["prediction"], "sentinel_embedding": _orthogonal_vector(base_embedding)},
            }
            overrides[inv.FIXTURE_REVOKED_MEMORY_EPISODE] = {"prediction": dict(base_result["prediction"])}
        except Exception:  # noqa: BLE001 - the fixture override is best-effort; absence just disables that one fixture
            pass
        host.inference_client = _CaseAwareInferenceClient(base_client, overrides)

        try:
            memory_bank = vcm2.MemoryBank.load(memory_bank_dir)
            memory_policy = vcm2.ComparisonPolicy.from_manifest(memory_bank_dir)
            host.memory_bank = memory_bank
            host.memory_policy = memory_policy
        except (FileNotFoundError, ValueError):
            host.memory_bank = None
            host.memory_policy = None

        return host

    # -- compiled graph lifecycle -------------------------------------------

    def compiled_app(self, make_graph: Callable[["HostServices"], Any], checkpoint_filename: str):
        """Compile the graph once per host with a single SQLite checkpoint connection under ``state_dir``; later calls return the same app."""
        if self._app is None:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = Path(self.state_dir) / checkpoint_filename
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._app = make_graph(self).compile(checkpointer=SqliteSaver(self._checkpoint_conn))
        return self._app

    def close(self) -> None:
        """Release the checkpoint connection (and any inference worker) so the state directory can be removed or reopened by another host."""
        self._app = None
        if self._checkpoint_conn is not None:
            try:
                self._checkpoint_conn.close()
            finally:
                self._checkpoint_conn = None
        closer = getattr(self.inference_client, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001 - best-effort worker shutdown
                pass

    # -- case lifecycle -------------------------------------------------

    def ensure_case(self, case_id: str) -> CaseState:
        try:
            return self.store.load_case(case_id)
        except StoreErrorV2:
            pass
        if case_id == inv.FIXTURE_NO_IMAGE:
            v1, v2, events = inv.build_fixture_no_image_versions(
                inventory_dir=self.inventory_dir, catalogue=self.catalogue, engine=self.engine, bundle=self.bundle,
            )
            self.store.create_case(v1)
            for ev in events:
                self.store.append_ledger_event(case_id, ev)
            return self.store.append_record(case_id, v2, expected_record_version=1)
        if case_id == inv.FIXTURE_IMAGE_QUALITY_FLAG:
            record = inv.build_fixture_image_quality_flag(
                inventory_dir=self.inventory_dir, catalogue=self.catalogue, engine=self.engine, bundle=self.bundle,
            )
            return self.store.create_case(record)
        if case_id == inv.FIXTURE_MEMORY_NO_MATCH:
            record = inv.build_fixture_memory_no_match(
                inventory_dir=self.inventory_dir, catalogue=self.catalogue, engine=self.engine, bundle=self.bundle,
            )
            return self.store.create_case(record)
        if case_id == inv.FIXTURE_REVOKED_MEMORY_EPISODE:
            record = inv.build_fixture_revoked_memory_episode(
                inventory_dir=self.inventory_dir, catalogue=self.catalogue, engine=self.engine, bundle=self.bundle,
            )
            return self.store.create_case(record)
        if case_id == inv.FIXTURE_CLINICAL_ABSENT:
            record = inv.build_fixture_clinical_absent(
                inventory_dir=self.inventory_dir, catalogue=self.catalogue, engine=self.engine, bundle=self.bundle,
            )
            return self.store.create_case(record)
        if case_id == inv.FIXTURE_AWAITING_REVIEW_RESUME:
            record = inv.build_record_from_inventory(
                inventory_dir=self.inventory_dir, case_id=inv.BASE_FIXTURE_CASE_ID, catalogue=self.catalogue,
                engine=self.engine, bundle=self.bundle,
            )
            import dataclasses as _dc
            record = _dc.replace(
                record, identity=_dc.replace(record.identity, case_id=case_id, source_label="synthetic_workflow_fixture"),
            )
            record = _dc.replace(record, record_sha256=record.compute_sha256())
            self.store.create_case(record)
            for ev in inv.build_fixture_awaiting_review_resume_events(case_id):
                self.store.append_episode_event(case_id, ev)
            return self.store.load_case(case_id)
        record = inv.build_record_from_inventory(
            inventory_dir=self.inventory_dir, case_id=case_id, catalogue=self.catalogue, engine=self.engine, bundle=self.bundle,
        )
        return self.store.create_case(record)

    # -- ledger -----------------------------------------------------------

    def ledger_for_case(self, case_id: str) -> Tuple[AmendmentLedgerV2, int]:
        try:
            case_state = self.store.load_case(case_id)
            events = case_state.ledger_events
        except StoreErrorV2:
            events = ()
        ledger = AmendmentLedgerV2.replay(
            events, catalogue=self.catalogue, engine=self.engine, calibrators=self.bundle.calibration, bundle=self.bundle,
        )
        return ledger, len(events)

    def flush_ledger_events(self, case_id: str, ledger: AmendmentLedgerV2, n_before: int) -> int:
        events = ledger.to_events()
        for ev in events[n_before:]:
            self.store.append_ledger_event(case_id, ev)
        return len(events)

    # -- memory -------------------------------------------------------------

    def memory_bank_for_case(self, case_id: str) -> Optional[vcm2.MemoryBank]:
        if case_id == inv.FIXTURE_REVOKED_MEMORY_EPISODE:
            return self._revoked_overlay_bank  # populated by scenarios.case_comparison before the tool call
        return self.memory_bank

    def set_revoked_overlay_bank(self, bank: Optional[vcm2.MemoryBank]) -> None:
        self._revoked_overlay_bank = bank

    # -- tool context -------------------------------------------------------

    def tool_context(self, case_id: str, *, session_id: str, actor: str, agent_mode: str = "offline_deterministic",
                     memory_request_explicit: bool = False) -> Tuple[ToolContext, AmendmentLedgerV2, int]:
        ledger, n0 = self.ledger_for_case(case_id)

        def xai_job_api(*, case_id: str, record_sha256: str, kind: str) -> Dict[str, Any]:
            return self.jobs.run(self, case_id=case_id, record_sha256=record_sha256, kind=kind)

        available_xai_job_api = xai_job_api if self.jobs is not None else None

        ctx = ToolContext(
            case_id=case_id, store=self.store, ledger=ledger, engine=self.engine, catalogue=self.catalogue,
            guideline_retriever=self.guideline_retriever,
            bundle_calibrators=self.bundle.calibration, memory_bank=self.memory_bank_for_case(case_id),
            memory_policy=self.memory_policy, memory_request_explicit=memory_request_explicit,
            inference_client=self.inference_client, xai_job_api=available_xai_job_api,
            session_id=session_id, actor=actor,
            mode=agent_mode,
        )
        return ctx, ledger, n0

    # -- real-image path resolution (worker mode only) -----------------------

    def resolve_case_image_paths(self, case_id: str) -> Dict[str, Any]:
        if case_id in inv.FIXTURE_CASE_IDS:
            raise ValueError(f"{case_id!r} is a synthetic fixture; it has no real image and cannot use --inference worker")
        if self._reserved_cases_cache is None:
            from dermpact.core.inference_client import _resolve_reserved_cases
            self._reserved_cases_cache = _resolve_reserved_cases()
        case = self._reserved_cases_cache.get(case_id)
        if case is None:
            raise ValueError(f"{case_id!r} is not one of the 50 reserved development cases")
        out: Dict[str, Any] = {"dermoscopic_path": case.dermoscopic_path, "dermoscopic_sha256": case.dermoscopic_sha256}
        if case.clinical_path is not None:
            out["clinical_path"] = case.clinical_path
            out["clinical_sha256"] = case.clinical_sha256
        return out


def _orthogonal_vector(reference: Any, seed: int = 20260906) -> list:
    """A deterministic unit vector whose expected cosine similarity with ``reference`` (and, in a high-dimensional embedding space, with every real bank episode) is ~0 -- used only by the ``fixture-memory-no-match`` override so ``retrieve_case_comparisons``/``compare()`` deterministically reports ``no_match``."""
    import numpy as np
    rng = np.random.default_rng(seed)
    dim = len(reference)
    v = rng.normal(size=dim)
    ref = np.asarray(reference, dtype=float)
    ref_norm = ref / max(float(np.linalg.norm(ref)), 1e-12)
    v = v - float(np.dot(v, ref_norm)) * ref_norm  # exactly orthogonal to the base embedding
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return v.tolist()


__all__ = ["DEFAULT_MEMORY_BANK_DIR", "DEFAULT_STATE_DIR", "HostBuildError", "HostServices", "INFERENCE_KINDS", "MODES"]
