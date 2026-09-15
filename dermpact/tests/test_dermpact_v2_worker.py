"""WP-X inference worker protocol: JSON framing, path allowlist, hash verification, structured errors and cancellation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from dermpact.eval.dermpact_phase3_calibration import ROOT, digest
from dermpact.core import xai
from dermpact.core.inference_worker import (
    FAKE_RUNTIME_ENV, Worker, WorkerError, _FakeEngine, _checked_bytes, self_test,
)
from dermpact.core.inference_client import WorkerClient, WorkerRequestError


class SelfTestTest(unittest.TestCase):
    def test_self_test_passes(self):
        self.assertTrue(self_test())


class InProcessWorkerTest(unittest.TestCase):
    """Exercises Worker.handle() directly (no subprocess) for the structured errors and cancellation semantics that would be slow/flaky to prove over a real pipe."""

    def setUp(self):
        self.engine = _FakeEngine()
        self.worker = Worker(self.engine)
        self.events = []

    def test_ping(self):
        self.assertEqual(self.worker.handle({"cmd": "ping", "id": "1"}, self.events.append), {"pong": True})

    def test_unknown_command_is_structured_error(self):
        with self.assertRaises(WorkerError) as ctx:
            self.worker.handle({"cmd": "nonsense", "id": "1"}, self.events.append)
        self.assertEqual(ctx.exception.code, "unknown_command")

    def test_predict_requires_dermoscopic_path(self):
        with self.assertRaises(WorkerError) as ctx:
            self.worker.handle({"cmd": "predict", "id": "1"}, self.events.append)
        self.assertEqual(ctx.exception.code, "missing_dermoscopic_path")

    def test_xai_clinical_requires_clinical_availability(self):
        tmp = ROOT / "dermpact/core/_test_tmp_image.png"
        _write_tiny_png(tmp)
        try:
            sha = digest(tmp)
            with self.assertRaises(WorkerError) as ctx:
                self.worker.handle({"cmd": "xai_clinical", "id": "1", "dermoscopic_path": str(tmp),
                                     "expected_dermoscopic_sha256": sha}, self.events.append)
            self.assertEqual(ctx.exception.code, "missing_clinical_availability")
        finally:
            tmp.unlink(missing_ok=True)

    def test_path_outside_repository_refused(self):
        with self.assertRaises(WorkerError) as ctx:
            _checked_bytes("../outside_repo.png", "0" * 64)
        self.assertEqual(ctx.exception.code, "path_outside_repository")

    def test_missing_expected_hash_refused(self):
        tmp = ROOT / "dermpact/core/_test_tmp_image2.png"
        _write_tiny_png(tmp)
        try:
            with self.assertRaises(WorkerError) as ctx:
                _checked_bytes(str(tmp), None)
            self.assertEqual(ctx.exception.code, "missing_expected_hash")
        finally:
            tmp.unlink(missing_ok=True)

    def test_hash_mismatch_refused(self):
        tmp = ROOT / "dermpact/core/_test_tmp_image3.png"
        _write_tiny_png(tmp)
        try:
            with self.assertRaises(WorkerError) as ctx:
                _checked_bytes(str(tmp), "0" * 64)
            self.assertEqual(ctx.exception.code, "hash_mismatch")
        finally:
            tmp.unlink(missing_ok=True)

    def test_cancel_before_dispatch_raises_cancelled(self):
        tmp = ROOT / "dermpact/core/_test_tmp_image4.png"
        _write_tiny_png(tmp)
        try:
            sha = digest(tmp)
            self.worker.cancel("req-1")
            with self.assertRaises(xai.Cancelled):
                self.worker.handle({"cmd": "xai_region", "id": "req-1", "dermoscopic_path": str(tmp),
                                     "expected_dermoscopic_sha256": sha}, self.events.append)
        finally:
            tmp.unlink(missing_ok=True)

    def test_no_uncancelled_request_unaffected(self):
        tmp = ROOT / "dermpact/core/_test_tmp_image5.png"
        _write_tiny_png(tmp)
        try:
            sha = digest(tmp)
            self.worker.cancel("some-other-request")
            result = self.worker.handle({"cmd": "xai_region", "id": "req-2", "dermoscopic_path": str(tmp),
                                          "expected_dermoscopic_sha256": sha}, self.events.append)
            self.assertEqual(result["status"], "complete")
        finally:
            tmp.unlink(missing_ok=True)


def _write_tiny_png(path: Path) -> None:
    from PIL import Image
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(path, format="PNG")


class SubprocessProtocolTest(unittest.TestCase):
    """Round trips through a real subprocess speaking the JSON protocol, with --fake-runtime so no real model is constructed."""

    @classmethod
    def setUpClass(cls):
        cls.client = WorkerClient(python_executable=sys.executable, extra_args=["--fake-runtime"],
                                   timeout_seconds=60)

    @classmethod
    def tearDownClass(cls):
        cls.client.shutdown()

    def test_ping_over_subprocess(self):
        self.assertEqual(self.client.ping(), {"pong": True})

    def test_predict_over_subprocess(self):
        tmp = ROOT / "dermpact/core/_test_tmp_subprocess.png"
        _write_tiny_png(tmp)
        try:
            sha = digest(tmp)
            result = self.client.predict(case_id="fixture", dermoscopic_path=tmp,
                                          expected_dermoscopic_sha256=sha, check_view=False)
            self.assertEqual(len(result["checkpoint_bindings"]), 5)
        finally:
            tmp.unlink(missing_ok=True)

    def test_predict_diagnosis_only_over_subprocess(self):
        tmp = ROOT / "dermpact/core/_test_tmp_diagnosis_only.png"
        _write_tiny_png(tmp)
        try:
            result = self.client.predict_diagnosis_only(
                dermoscopic_path=tmp,
                expected_dermoscopic_sha256=digest(tmp),
            )
            self.assertEqual(result["route"], "diagnosis_only")
            self.assertEqual(len(result["diagnosis_probabilities"]), 9)
            self.assertEqual(len(result["binary_probabilities"]), 1)
            self.assertEqual(len(result["checkpoint_bindings"]), 5)
        finally:
            tmp.unlink(missing_ok=True)

    def test_predict_final_full_over_subprocess(self):
        tmp = ROOT / "dermpact/core/_test_tmp_final_full.png"
        _write_tiny_png(tmp)
        try:
            result = self.client.predict_final_full(
                dermoscopic_path=tmp,
                expected_dermoscopic_sha256=digest(tmp),
                check_view=False,
            )
            self.assertEqual(result["route"], "full_reporting_adapter_fake")
            self.assertEqual(len(result["diagnosis_probabilities"]), 9)
            self.assertEqual(len(result["binary_probabilities"]), 8)
        finally:
            tmp.unlink(missing_ok=True)

    def test_path_outside_repository_rejected_over_subprocess(self):
        with self.assertRaises(WorkerRequestError) as ctx:
            self.client.predict(dermoscopic_path="C:/Windows/win.ini",
                                 expected_dermoscopic_sha256="0" * 64)
        self.assertEqual(ctx.exception.code, "path_outside_repository")

    def test_hash_mismatch_rejected_over_subprocess(self):
        tmp = ROOT / "dermpact/core/_test_tmp_subprocess2.png"
        _write_tiny_png(tmp)
        try:
            with self.assertRaises(WorkerRequestError) as ctx:
                self.client.predict(dermoscopic_path=tmp, expected_dermoscopic_sha256="0" * 64)
            self.assertEqual(ctx.exception.code, "hash_mismatch")
        finally:
            tmp.unlink(missing_ok=True)

    def test_xai_region_over_subprocess_with_progress(self):
        tmp = ROOT / "dermpact/core/_test_tmp_subprocess3.png"
        _write_tiny_png(tmp)
        try:
            sha = digest(tmp)
            progress_events = []
            result = self.client.xai_region(dermoscopic_path=tmp, expected_dermoscopic_sha256=sha,
                                             progress_cb=progress_events.append)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["members_measured"], 5)
            self.assertTrue(progress_events, "expected at least one progress event")
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
