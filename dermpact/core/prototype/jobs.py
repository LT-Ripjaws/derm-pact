"""WP-D: bounded XAI job coordination with terminal progress."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from dermpact.core import xai
from dermpact.core.versions import XAI_PROTOCOL_VERSION
from dermpact.core.prototype.state import now_iso

XAI_KINDS = ("region", "clinical")


class JobCoordinator:
    def __init__(self, *, clock: Callable[[], str] = now_iso,
                 runtime_clock: Callable[[], float] = time.perf_counter,
                 write_overlay: bool = True) -> None:
        self._busy = False
        self._clock = clock
        self._runtime_clock = runtime_clock
        self._write_overlay = write_overlay

    # -- per-case job-status file -----------------------------------------

    def _jobs_path(self, host: Any, case_id: str) -> Path:
        return host.store._case_dir(case_id) / "xai_jobs.json"

    def _read_jobs(self, host: Any, case_id: str) -> Dict[str, Any]:
        p = self._jobs_path(host, case_id)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def _write_jobs(self, host: Any, case_id: str, jobs: Dict[str, Any]) -> None:
        p = self._jobs_path(host, case_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(jobs, sort_keys=True, indent=2), encoding="utf-8")

    def _mark_interrupted(self, host: Any, case_id: str) -> None:
        jobs = self._read_jobs(host, case_id)
        changed = False
        for rec in jobs.values():
            if rec.get("status") in ("queued", "running"):
                rec["status"] = "failed:interrupted"
                rec["finished_at"] = self._clock()
                changed = True
        if changed:
            self._write_jobs(host, case_id, jobs)

    def status(self, host: Any, case_id: str) -> Dict[str, Any]:
        return self._read_jobs(host, case_id)

    # -- run ----------------------------------------------------------------

    def run(self, host: Any, *, case_id: str, record_sha256: str, kind: str) -> Dict[str, Any]:
        if kind not in XAI_KINDS:
            return {"job_id": None, "status": "unavailable", "reason": f"unknown xai kind {kind!r}"}
        if self._busy:
            return {"job_id": None, "status": "unavailable", "reason": "another XAI job is already running in this process"}
        self._busy = True
        try:
            return self._run_locked(host, case_id=case_id, record_sha256=record_sha256, kind=kind)
        finally:
            self._busy = False

    def _run_locked(self, host: Any, *, case_id: str, record_sha256: str, kind: str) -> Dict[str, Any]:
        self._mark_interrupted(host, case_id)
        case_state = host.store.load_case(case_id)
        record = case_state.current_record
        if record.record_sha256 != record_sha256:
            return {"job_id": None, "status": "unavailable", "reason": "record changed since the request was made"}

        job_id = f"{kind}-{record_sha256[:16]}"
        jobs = self._read_jobs(host, case_id)
        jobs[job_id] = {"status": "queued", "kind": kind, "case_id": case_id, "started_at": self._clock(), "finished_at": None}
        self._write_jobs(host, case_id, jobs)

        if kind == "clinical" and record.inputs.clinical_availability != "present":
            receipt = self._clinical_receipt(
                record, {"status": "not_applicable", "n_members": 5, "members_measured": 0,
                         "reason": f"clinical_availability={record.inputs.clinical_availability}"},
                started_at=self._clock(), runtime_seconds=0.0,
            )
            host.store.save_xai_receipt(case_id, receipt)
            jobs[job_id].update(status="complete", finished_at=self._clock())
            self._write_jobs(host, case_id, jobs)
            return {
                "job_id": job_id, "status": "complete", "receipt_id": job_id,
                "receipt_status": "not_applicable", "kind": "clinical",
                "clinical_availability": record.inputs.clinical_availability,
                "members_measured": 0,
                "reason": f"clinical_availability={record.inputs.clinical_availability}",
                "measurement_effect": "no clinical-photo sensitivity or contribution was computed",
            }

        client = host.inference_client
        if not hasattr(client, "xai_region"):
            jobs[job_id].update(status="failed", finished_at=self._clock())
            self._write_jobs(host, case_id, jobs)
            return {"job_id": job_id, "status": "unavailable",
                    "reason": "no worker inference client configured for measured XAI; re-run with --inference worker"}
        try:
            paths = host.resolve_case_image_paths(case_id)
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed, honest unavailability
            jobs[job_id].update(status="failed", finished_at=self._clock())
            self._write_jobs(host, case_id, jobs)
            return {"job_id": job_id, "status": "unavailable", "reason": str(exc)}

        jobs[job_id]["status"] = "running"
        self._write_jobs(host, case_id, jobs)

        def progress(payload: Dict[str, Any]) -> None:
            fold = payload.get("fold")
            n_folds = payload.get("n_folds", 5)
            stage = payload.get("stage", kind)
            if fold is not None:
                print(f"[xai] {stage} fold {fold + 1}/{n_folds}", file=sys.stderr, flush=True)

        started_at = self._clock()
        t0 = self._runtime_clock()
        try:
            if kind == "region":
                result = client.xai_region(
                    dermoscopic_path=paths["dermoscopic_path"], clinical_path=paths.get("clinical_path"),
                    expected_dermoscopic_sha256=paths["dermoscopic_sha256"],
                    expected_clinical_sha256=paths.get("clinical_sha256"), progress_cb=progress,
                )
                receipt = self._region_receipt(host, record, result, started_at=started_at, runtime_seconds=self._runtime_clock() - t0)
            else:
                result = client.xai_clinical(
                    dermoscopic_path=paths["dermoscopic_path"], clinical_path=paths.get("clinical_path"),
                    expected_dermoscopic_sha256=paths["dermoscopic_sha256"],
                    expected_clinical_sha256=paths.get("clinical_sha256"),
                    clinical_availability=record.inputs.clinical_availability, progress_cb=progress,
                )
                receipt = self._clinical_receipt(record, result, started_at=started_at, runtime_seconds=self._runtime_clock() - t0)
        except Exception as exc:  # noqa: BLE001 - never let a worker crash escape the CLI as a traceback
            jobs = self._read_jobs(host, case_id)
            jobs[job_id].update(status="failed", finished_at=self._clock())
            self._write_jobs(host, case_id, jobs)
            return {"job_id": job_id, "status": "failed", "reason": str(exc)}

        if self._write_overlay and kind == "region" and receipt.get("status") == "complete":
            overlay_path = self._write_region_overlay_png(host, case_id, receipt)
            if overlay_path is not None:
                receipt["overlay_png"] = str(overlay_path)
                receipt = xai.seal_receipt(receipt)
        xai.verify_receipt_content(receipt)
        host.store.save_xai_receipt(case_id, receipt)
        jobs = self._read_jobs(host, case_id)
        jobs[job_id].update(status=("complete" if receipt["status"] == "complete" else "failed"), finished_at=self._clock())
        self._write_jobs(host, case_id, jobs)
        return {"job_id": job_id, "status": jobs[job_id]["status"], "receipt_id": receipt["receipt_id"], "receipt_status": receipt["status"]}

    def _write_region_overlay_png(self, host: Any, case_id: str, receipt: Dict[str, Any]) -> Optional[Path]:
        """Best-effort 4x4 region-sensitivity heat overlay (``local_mean`` / ``melanoma_head``), written as a scientific output alongside the receipt -- never an application screenshot."""
        try:
            from PIL import Image
            grid = ((receipt.get("heatmap") or {}).get("local_mean") or {}).get("melanoma_head")
            if not grid:
                return None
            flat = [v for row in grid for v in row if v is not None]
            if not flat:
                return None
            lo, hi = min(flat), max(flat)
            span = (hi - lo) or 1.0
            cell = 56
            img = Image.new("RGB", (cell * len(grid[0]), cell * len(grid)), color=(32, 32, 32))
            px = img.load()
            for r, row in enumerate(grid):
                for c, v in enumerate(row):
                    if v is None:
                        color = (64, 64, 64)
                    else:
                        t = (v - lo) / span
                        color = (int(255 * t), int(32 * (1 - t)), int(255 * (1 - t)))
                    for dy in range(cell):
                        for dx in range(cell):
                            px[c * cell + dx, r * cell + dy] = color
            path = host.store._xai_dir(case_id) / f"{receipt['receipt_id']}_overlay.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path)
            return path
        except Exception:  # noqa: BLE001 - the overlay is an optional scientific extra, never load-bearing
            return None

    def _region_receipt(self, host: Any, record: Any, result: Dict[str, Any], *, started_at: str, runtime_seconds: float) -> Dict[str, Any]:
        cache_key = xai.build_cache_key(
            dermoscopic_sha256=record.inputs.dermoscopic.sanitized_sha256,
            clinical_sha256=(record.inputs.clinical.sanitized_sha256 if record.inputs.clinical_availability == "present" else None),
            case_id=record.identity.case_id, record_version=record.identity.record_version, record_sha256=record.record_sha256,
            checkpoint_shas=[b["sha256"] for b in record.perception.checkpoint_bindings],
            preprocessing_version=record.versions.preprocessing_version, calibration_sha256=record.perception.calibration_sha256,
            target_heads=xai.TARGETS, methods=xai.METHODS, sigma=xai.GAUSSIAN_SIGMA_PX, seed=0,
        )
        parity = None
        if result.get("status") == "complete" and result.get("no_op_member_raw") is not None:
            try:
                prediction = host.inference_client.predict(case_id=record.identity.case_id)["prediction"]
                parity = xai.no_op_parity(prediction, host.bundle.calibration, result["no_op_member_raw"])
            except Exception:  # noqa: BLE001 - a missing parity proof downgrades the receipt to failed
                parity = None
        receipt = xai.build_region_receipt(
            cache_key=cache_key, sweep_result=result, started_at=started_at, finished_at=self._clock(),
            runtime_seconds=runtime_seconds, peak_rss_bytes=None, no_op_parity_result=parity,
        )
        return receipt

    def _clinical_receipt(self, record: Any, result: Dict[str, Any], *, started_at: str, runtime_seconds: float) -> Dict[str, Any]:
        cache_key = xai.build_cache_key(
            dermoscopic_sha256=record.inputs.dermoscopic.sanitized_sha256,
            clinical_sha256=record.inputs.clinical.sanitized_sha256, case_id=record.identity.case_id,
            record_version=record.identity.record_version, record_sha256=record.record_sha256,
            checkpoint_shas=[b["sha256"] for b in record.perception.checkpoint_bindings],
            preprocessing_version=record.versions.preprocessing_version, calibration_sha256=record.perception.calibration_sha256,
            target_heads=xai.TARGETS, methods=(), sigma=None, seed=0,
        )
        return xai.seal_receipt({
            "receipt_id": xai.cache_key_hash(cache_key), "kind": "clinical", "cache_key": cache_key,
            "protocol_version": XAI_PROTOCOL_VERSION, "status": result.get("status", "failed"),
            "started_at": started_at, "finished_at": self._clock(), "runtime_seconds": runtime_seconds, **result,
        })


__all__ = ["JobCoordinator", "XAI_KINDS"]
