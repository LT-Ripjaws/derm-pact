"""Subprocess worker: loads SelectedPerception once, serves newline-delimited JSON commands on stdin -> stdout."""
from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from dermpact.eval.dermpact_phase3_calibration import ROOT, digest
from dermpact.reliability.inference import aggregate_fold_probabilities, calibrated_fold_probabilities
from dermpact.reliability.sentinels import sanitize_image
from dermpact.core import xai

FAKE_RUNTIME_ENV = "DERMPACT_XAI_FAKE_RUNTIME"


class WorkerError(Exception):
    """A structured, caller-facing failure. Never a raw traceback."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _resolve_allowed_path(raw_path: str) -> Path:
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else (ROOT / path)).resolve()
    if not resolved.is_relative_to(ROOT):
        raise WorkerError("path_outside_repository", "path must resolve under the repository root")
    if not resolved.is_file():
        raise WorkerError("file_not_found", "path does not exist")
    return resolved


def _checked_bytes(raw_path: Optional[str], expected_sha256: Optional[str]) -> Optional[bytes]:
    if raw_path is None:
        return None
    if not expected_sha256:
        raise WorkerError("missing_expected_hash", "expected sha256 required before opening any image")
    path = _resolve_allowed_path(raw_path)
    if digest(path) != expected_sha256:
        raise WorkerError("hash_mismatch", "file does not match the caller-supplied expected sha256")
    return path.read_bytes()


class _FakeEngine:
    """``--fake-runtime`` stand-in for ``SelectedPerception``."""

    def __init__(self):
        self._runtime = xai.FakePerceptionRuntime()
        self.calibrators = self._runtime.calibrators
        self.calibration_sha256 = "f" * 64

    def predict(self, dermoscopy_bytes, clinical_bytes=None, check_view=True):
        image, _, receipt = sanitize_image(dermoscopy_bytes)
        clinical, clinical_receipt = None, None
        if clinical_bytes is not None:
            clinical, _, clinical_receipt = sanitize_image(clinical_bytes)
        prefixes = self._runtime.prefix(image, clinical)
        raws, outputs, bindings = [], {}, []
        for fold, model, binding in self._runtime.fold_iter():
            out = model(*prefixes)
            raws.append({"diagnosis": np.asarray(out["diagnosis"])[0].tolist(),
                         "melanoma": float(np.asarray(out["melanoma"])[0]),
                         "criteria": np.asarray(out["criteria"])[0].tolist()})
            outputs[fold] = calibrated_fold_probabilities(out, fold, self.calibrators)
            bindings.append(binding)
        diag, binary = aggregate_fold_probabilities(outputs)
        return {"diagnosis_probabilities": diag[0].tolist(), "binary_probabilities": binary[0].tolist(),
                "criterion_state_labels": {}, "criterion_state_probabilities": {},
                "checklist_melanoma_probability": None,
                "alternative_binary_probabilities": None, "sentinel_embedding": [0.0] * 8,
                "raw_fold_outputs": raws, "raw_view_fold_outputs": [], "checkpoint_bindings": bindings,
                "calibration_sha256": self.calibration_sha256, "image_receipt": receipt,
                "clinical_receipt": clinical_receipt, "clinical_present": clinical is not None,
                "elapsed_seconds": 0.0, "inference_policy": "fake-runtime: not a real model",
                "ensemble_calibration_status": "fake", "coverage_status": "fake", "view_policy": "fake"}

    def xai_runtime(self):
        return self._runtime


def _build_engine(threads: int = 12):
    if os.environ.get(FAKE_RUNTIME_ENV) == "1":
        return _FakeEngine()
    from dermpact.reliability.selected_perception import SelectedPerception
    return SelectedPerception(threads)


def _perception_runtime(engine):
    if hasattr(engine, "xai_runtime"):
        return engine.xai_runtime()
    return xai.RealPerceptionRuntime(engine)


class Worker:
    """Dispatches one decoded request at a time."""

    def __init__(self, engine):
        self.engine = engine
        self._runtime = _perception_runtime(engine)
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    def cancel(self, request_id: str) -> None:
        with self._lock:
            self._cancelled.add(request_id)

    def _cancel_check(self, request_id: Optional[str]):
        if request_id is None:
            return None

        def check() -> bool:
            with self._lock:
                return request_id in self._cancelled
        return check

    def handle(self, request: dict, emit: Callable[[dict], None]) -> Any:
        cmd = request.get("cmd")
        request_id = request.get("id")

        if cmd == "ping":
            return {"pong": True}

        if cmd == "predict":
            dermoscopic = _checked_bytes(request.get("dermoscopic_path"), request.get("expected_dermoscopic_sha256"))
            if dermoscopic is None:
                raise WorkerError("missing_dermoscopic_path", "predict requires dermoscopic_path")
            clinical = _checked_bytes(request.get("clinical_path"), request.get("expected_clinical_sha256"))
            return self.engine.predict(dermoscopic, clinical, bool(request.get("check_view", True)))

        if cmd == "predict_diagnosis_only":
            dermoscopic = _checked_bytes(request.get("dermoscopic_path"), request.get("expected_dermoscopic_sha256"))
            if dermoscopic is None:
                raise WorkerError("missing_dermoscopic_path", "predict_diagnosis_only requires dermoscopic_path")
            from dermpact.core.final_inference import predict_diagnosis_only
            return predict_diagnosis_only(self.engine, dermoscopic, bool(request.get("check_view", False)))

        if cmd == "predict_final_full":
            dermoscopic = _checked_bytes(request.get("dermoscopic_path"), request.get("expected_dermoscopic_sha256"))
            if dermoscopic is None:
                raise WorkerError("missing_dermoscopic_path", "predict_final_full requires dermoscopic_path")
            clinical = _checked_bytes(request.get("clinical_path"), request.get("expected_clinical_sha256"))
            from dermpact.core.final_inference import predict_full
            return predict_full(self.engine, dermoscopic, clinical, bool(request.get("check_view", True)))

        if cmd in ("xai_region", "xai_clinical"):
            dermoscopic = _checked_bytes(request.get("dermoscopic_path"), request.get("expected_dermoscopic_sha256"))
            if dermoscopic is None:
                raise WorkerError("missing_dermoscopic_path", f"{cmd} requires dermoscopic_path")
            clinical_bytes = _checked_bytes(request.get("clinical_path"), request.get("expected_clinical_sha256"))
            image, _, _ = sanitize_image(dermoscopic)
            clinical_image = None
            if clinical_bytes is not None:
                clinical_image, _, _ = sanitize_image(clinical_bytes)

            def progress(payload: dict) -> None:
                emit({"id": request_id, **payload})

            cancel_check = self._cancel_check(request_id)
            if cmd == "xai_region":
                methods = tuple(request.get("methods") or xai.METHODS)
                return xai.run_region_sensitivity(self._runtime, image, clinical_image, methods=methods,
                                                   seed=int(request.get("seed", 0)),
                                                   cancel_check=cancel_check, progress_cb=progress)
            availability = request.get("clinical_availability")
            if not availability:
                raise WorkerError("missing_clinical_availability", "xai_clinical requires clinical_availability")
            return xai.run_clinical_contribution(self._runtime, image, clinical_image, availability,
                                                  cancel_check=cancel_check, progress_cb=progress)

        raise WorkerError("unknown_command", f"unrecognised command {cmd!r}")


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def run_stdio(engine=None) -> None:
    engine = engine if engine is not None else _build_engine()
    worker = Worker(engine)
    inbox: "queue.Queue[dict]" = queue.Queue()

    def reader() -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                _emit({"id": None, "ok": False, "error": {"code": "invalid_json", "detail": str(exc)}})
                continue
            if request.get("cmd") == "cancel":
                target = request.get("id")
                if target is not None:
                    worker.cancel(target)
                continue
            inbox.put(request)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    while True:
        request = inbox.get()
        request_id = request.get("id")
        cmd = request.get("cmd")
        try:
            result = worker.handle(request, _emit)
            _emit({"id": request_id, "ok": True, "result": result})
        except xai.Cancelled as exc:
            _emit({"id": request_id, "ok": False, "error": {"code": "cancelled", "detail": str(exc)}})
        except WorkerError as exc:
            _emit({"id": request_id, "ok": False, "error": {"code": exc.code, "detail": exc.detail}})
        except Exception as exc:  # noqa: BLE001 - structured to the caller; full trace only to stderr
            print(f"worker error handling {cmd}: {exc}\n{traceback.format_exc()}", file=sys.stderr)
            _emit({"id": request_id, "ok": False, "error": {"code": "internal_error", "detail": str(exc)}})
        if cmd == "shutdown":
            break


def self_test() -> bool:
    """ping + fake-runtime roundtrip, entirely in-process (no subprocess, no real model, no image outside a synthetic in-memory one)."""
    os.environ[FAKE_RUNTIME_ENV] = "1"
    engine = _FakeEngine()
    worker = Worker(engine)
    events: list = []

    pong = worker.handle({"cmd": "ping", "id": "1"}, events.append)
    assert pong == {"pong": True}, pong

    tiny = Image.new("RGB", (64, 64), color=(120, 60, 30))
    buf = io.BytesIO()
    tiny.save(buf, format="PNG")
    tmp_path = ROOT / "dermpact/core/_worker_self_test_tmp.png"
    tmp_path.write_bytes(buf.getvalue())
    try:
        sha = digest(tmp_path)
        predicted = worker.handle({"cmd": "predict", "id": "2", "dermoscopic_path": str(tmp_path),
                                    "expected_dermoscopic_sha256": sha, "check_view": False}, events.append)
        assert len(predicted["checkpoint_bindings"]) == 5, predicted

        region = worker.handle({"cmd": "xai_region", "id": "3", "dermoscopic_path": str(tmp_path),
                                 "expected_dermoscopic_sha256": sha}, events.append)
        assert region["status"] == "complete", region

        clinical = worker.handle({"cmd": "xai_clinical", "id": "4", "dermoscopic_path": str(tmp_path),
                                   "expected_dermoscopic_sha256": sha, "clinical_availability": "absent"},
                                  events.append)
        assert clinical["status"] == "not_applicable", clinical

        try:
            _checked_bytes("../../../etc/passwd", "0" * 64)
        except WorkerError as exc:
            assert exc.code == "path_outside_repository", exc.code
        else:
            raise AssertionError("path traversal was not rejected")

        try:
            _checked_bytes(str(tmp_path), "0" * 64)
        except WorkerError as exc:
            assert exc.code == "hash_mismatch", exc.code
        else:
            raise AssertionError("hash mismatch was not rejected")
    finally:
        tmp_path.unlink(missing_ok=True)
        os.environ.pop(FAKE_RUNTIME_ENV, None)
    print("WORKER_SELF_TEST_PASS")
    return True


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fake-runtime", action="store_true")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    if args.fake_runtime:
        os.environ[FAKE_RUNTIME_ENV] = "1"
    run_stdio(_build_engine(args.threads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
