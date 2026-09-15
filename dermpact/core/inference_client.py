"""Clients for the frozen selected model: a subprocess worker for real inference and on-demand XAI, and a cache reader for prebuilt inventory predictions."""
from __future__ import annotations

import itertools
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

from dermpact.eval.dermpact_phase3_calibration import ROOT, digest, write_json
from dermpact.core.versions import (
    PREPROCESSING_VERSION, SELECTION_MANIFEST, XAI_PROTOCOL_VERSION,
)
from dermpact.core import xai


class NotCached(Exception):
    """Raised by CachedInferenceClient when a case has no stored prediction."""


class WorkerTimeout(Exception):
    pass


class WorkerCrashed(Exception):
    pass


class WorkerRequestError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class InferenceClient(Protocol):
    """Shared surface: predict-by-case."""

    def predict(self, *, case_id: Optional[str] = None, **kwargs) -> dict: ...


class WorkerClient:
    """Spawns ``dermpact.core.inference_worker`` with the given Python interpreter (``venv_p2`` by default), talks newline-delimited JSON over stdin/stdout, supports a progress callback, cooperative cancel, and restarts the process on the next request after a crash."""

    def __init__(self, python_executable: Optional[str] = None, extra_args: Sequence = (),
                 timeout_seconds: float = 900.0, env: Optional[dict] = None):
        self.python_executable = python_executable or sys.executable
        self.extra_args = list(extra_args)
        self.timeout_seconds = timeout_seconds
        self.env = env
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._next_id = itertools.count(1)
        self._responses: dict[str, "queue.Queue[dict]"] = {}
        self._responses_lock = threading.Lock()
        self._stderr_tail: list[str] = []

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = [self.python_executable, "-m", "dermpact.core.inference_worker", *self.extra_args]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, bufsize=1, cwd=str(ROOT), env=self.env)
        self._responses = {}
        self._stderr_tail = []
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True).start()

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self._responses_lock:
                self._responses.setdefault(payload.get("id"), queue.Queue()).put(payload)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            self._stderr_tail.append(line.rstrip())
            del self._stderr_tail[:-200]

    def request(self, cmd: str, *, progress_cb: Optional[Callable[[dict], None]] = None, **fields) -> Any:
        with self._lock:
            self._ensure_started()
            request_id = str(next(self._next_id))
            with self._responses_lock:
                self._responses[request_id] = queue.Queue()
            payload = {"id": request_id, "cmd": cmd, **fields}
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise WorkerCrashed(f"could not send {cmd}: {exc}") from exc

            deadline = time.monotonic() + self.timeout_seconds
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise WorkerTimeout(f"{cmd} timed out after {self.timeout_seconds}s")
                    if self._proc.poll() is not None:
                        raise WorkerCrashed(f"worker exited with code {self._proc.returncode}: "
                                            + "; ".join(self._stderr_tail[-20:]))
                    try:
                        message = self._responses[request_id].get(timeout=min(remaining, 1.0))
                    except queue.Empty:
                        continue
                    if message.get("event") == "progress":
                        if progress_cb is not None:
                            progress_cb(message)
                        continue
                    if message.get("ok"):
                        return message["result"]
                    error = message.get("error", {})
                    raise WorkerRequestError(error.get("code", "unknown_error"), error.get("detail", ""))
            finally:
                with self._responses_lock:
                    self._responses.pop(request_id, None)

    def cancel(self, request_id: str) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.write(json.dumps({"cmd": "cancel", "id": request_id}) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def ping(self) -> dict:
        return self.request("ping")

    def predict(self, *, case_id: Optional[str] = None, dermoscopic_path, clinical_path=None,
                expected_dermoscopic_sha256: str, expected_clinical_sha256: Optional[str] = None,
                check_view: bool = True) -> dict:
        return self.request(
            "predict", dermoscopic_path=str(dermoscopic_path),
            clinical_path=str(clinical_path) if clinical_path else None,
            expected_dermoscopic_sha256=expected_dermoscopic_sha256,
            expected_clinical_sha256=expected_clinical_sha256, check_view=check_view)

    def predict_diagnosis_only(self, *, dermoscopic_path, expected_dermoscopic_sha256: str,
                               check_view: bool = False) -> dict:
        return self.request(
            "predict_diagnosis_only", dermoscopic_path=str(dermoscopic_path),
            expected_dermoscopic_sha256=expected_dermoscopic_sha256,
            check_view=check_view,
        )

    def predict_final_full(self, *, dermoscopic_path, clinical_path=None,
                           expected_dermoscopic_sha256: str,
                           expected_clinical_sha256: Optional[str] = None,
                           check_view: bool = True) -> dict:
        return self.request(
            "predict_final_full", dermoscopic_path=str(dermoscopic_path),
            clinical_path=str(clinical_path) if clinical_path else None,
            expected_dermoscopic_sha256=expected_dermoscopic_sha256,
            expected_clinical_sha256=expected_clinical_sha256,
            check_view=check_view,
        )

    def xai_region(self, *, dermoscopic_path, clinical_path=None, expected_dermoscopic_sha256: str,
                   expected_clinical_sha256: Optional[str] = None, methods=None, seed: int = 0,
                   progress_cb: Optional[Callable[[dict], None]] = None) -> dict:
        return self.request(
            "xai_region", progress_cb=progress_cb, dermoscopic_path=str(dermoscopic_path),
            clinical_path=str(clinical_path) if clinical_path else None,
            expected_dermoscopic_sha256=expected_dermoscopic_sha256,
            expected_clinical_sha256=expected_clinical_sha256,
            methods=list(methods) if methods else None, seed=seed)

    def xai_clinical(self, *, dermoscopic_path, clinical_path, expected_dermoscopic_sha256: str,
                      expected_clinical_sha256: Optional[str], clinical_availability: str,
                      progress_cb: Optional[Callable[[dict], None]] = None) -> dict:
        return self.request(
            "xai_clinical", progress_cb=progress_cb, dermoscopic_path=str(dermoscopic_path),
            clinical_path=str(clinical_path) if clinical_path else None,
            expected_dermoscopic_sha256=expected_dermoscopic_sha256,
            expected_clinical_sha256=expected_clinical_sha256, clinical_availability=clinical_availability)

    def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            self.request("shutdown")
        except Exception:
            pass
        finally:
            self.close()

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        self._proc = None


class CachedInferenceClient:
    """Reads predictions already built by ``build_inventory`` (see below)."""

    def __init__(self, inventory_dir):
        self.inventory_dir = Path(inventory_dir)
        self.predictions_dir = self.inventory_dir / "predictions"

    def predict(self, *, case_id: Optional[str] = None, **_ignored) -> dict:
        if not case_id:
            raise NotCached("case_id required for cached inference")
        path = self.predictions_dir / f"{case_id}.json"
        if not path.is_file():
            raise NotCached(f"no cached prediction for {case_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = dict(payload)
        result["source"] = "cached_inventory"
        result["computed_at"] = (payload.get("case_meta") or {}).get("computed_at")
        return result



def _load_reserved_case_ids() -> list[str]:
    records_path = ROOT / "research/experiments/dermpact_phase3/assessed_records_v3/records.json"
    if digest(records_path) != Path(str(records_path) + ".sha256").read_text().strip():
        raise AssertionError("assessed_records_v3 changed")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    ids = [r["raw_model_evidence"]["identity"]["case_id"] for r in records]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise AssertionError("expected exactly 50 reserved case ids")
    return ids


def _resolve_reserved_cases() -> dict:
    from dermpact.eval import dermpact_phase2_4b_inputs as case_inputs
    meta = ROOT / "friend-cuda/research/manifests"
    data = case_inputs.resolve_development_inputs(
        inventory_path=meta / "dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json",
        fold_manifest_path=meta / "dermpact_phase2_1/dermpact_phase2_1_development_folds_v1.json",
        clinical_photo_audit_path=meta / "dermpact_phase2_3/clinical_photo_features_v1/clinical_photo_feature_protocol.json",
        structured_fact_support_path=meta / "dermpact_phase2_3/structured_facts_v1/structured_fact_support.json",
        require_files=False)
    reserved_ids = set(_load_reserved_case_ids())
    cases = {c.group_id: c for c in data.cases if c.group_id in reserved_ids}
    if set(cases) != reserved_ids:
        raise AssertionError("could not resolve all 50 reserved cases from the development inventory")
    return cases


def _resolve_confirmatory_cases() -> dict:
    """Return 100 label-free development cases, balanced to 20 per fold."""
    from dermpact.eval import dermpact_phase2_4b_inputs as case_inputs
    meta = ROOT / "friend-cuda/research/manifests"
    data = case_inputs.resolve_development_inputs(
        inventory_path=meta / "dermpact_phase1_v1/dermpact_phase1_derm7pt_inventory_v1.json",
        fold_manifest_path=meta / "dermpact_phase2_1/dermpact_phase2_1_development_folds_v1.json",
        clinical_photo_audit_path=meta / "dermpact_phase2_3/clinical_photo_features_v1/clinical_photo_feature_protocol.json",
        structured_fact_support_path=meta / "dermpact_phase2_3/structured_facts_v1/structured_fact_support.json",
        require_files=False,
    )
    all_cases = {case.group_id: case for case in data.cases}
    selected_ids = set(_load_reserved_case_ids())
    per_fold = {fold: sum(all_cases[case_id].fold == fold for case_id in selected_ids) for fold in range(5)}
    for fold in range(5):
        needed = 20 - per_fold[fold]
        candidates = [
            case for case in data.cases
            if case.fold == fold and case.group_id not in selected_ids
        ]
        candidates.sort(key=lambda case: hashlib.sha256(
            f"dermpact-confirmatory100-v1:{case.group_id}".encode("utf-8")
        ).hexdigest())
        selected_ids.update(case.group_id for case in candidates[:needed])
    cases = {case_id: all_cases[case_id] for case_id in selected_ids}
    counts = {fold: sum(case.fold == fold for case in cases.values()) for fold in range(5)}
    if len(cases) != 100 or counts != {fold: 20 for fold in range(5)}:
        raise AssertionError(f"confirmatory case selection failed: n={len(cases)} folds={counts}")
    return cases


def _checklist_disagrees_with_direct_head(prediction: dict) -> Optional[bool]:
    """Demo-subset selection helper only: is the direct diagnosis head's top class 'melanoma' while the 7-point checklist does/doesn't independently reach the melanoma-suspicion threshold (or vice versa)?"""
    from dermpact.eval.dermpact_phase4_benchmark import CRITERIA, GLOBAL_CLASSES, INDETERMINATE_BAND, NEG_VALUE, POS_VALUE
    from dermpact.criteria.aggregate import score_criteria
    from dermpact.criteria.schema import validate_values

    binary = prediction["binary_probabilities"]
    values = {}
    for i, name in enumerate(CRITERIA):
        p = float(binary[1 + i])
        if abs(p - 0.5) < INDETERMINATE_BAND:
            values[name] = "indeterminate"
        else:
            values[name] = POS_VALUE[name] if p >= 0.5 else NEG_VALUE[name]
    try:
        scored = score_criteria(validate_values(values))
    except ValueError:
        return None
    if not scored.decidable:
        return None
    classifier_says_melanoma = int(np.argmax(prediction["diagnosis_probabilities"])) == GLOBAL_CLASSES.index("melanoma")
    return bool(scored.meets_threshold) != classifier_says_melanoma


def _peak_rss_bytes(pid: Optional[int]) -> Optional[int]:
    if pid is None:
        return None
    try:
        import psutil
        info = psutil.Process(pid).memory_info()
        return int(getattr(info, "peak_wset", info.rss))
    except Exception:
        return None


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def run_feasibility(out_dir: Path) -> dict:
    """Measures the worker end to end on derm7pt-case-22: predict, region sweep, clinical-photo comparison (if available), no-op parity, and checkpoint-hash re-verification."""
    out_dir = Path(out_dir)
    case = _resolve_reserved_cases()["derm7pt-case-22"]
    if digest(case.dermoscopic_path) != case.dermoscopic_sha256:
        raise AssertionError("reserved image changed")

    client = WorkerClient()
    latencies = {}
    peak_rss = {}
    t0 = time.perf_counter()
    prediction = client.predict(
        case_id=case.group_id, dermoscopic_path=case.dermoscopic_path,
        expected_dermoscopic_sha256=case.dermoscopic_sha256,
        clinical_path=case.clinical_path, expected_clinical_sha256=case.clinical_sha256, check_view=True)
    latencies["predict_seconds"] = time.perf_counter() - t0
    peak_rss["after_predict_bytes"] = _peak_rss_bytes(client.pid)

    t1 = time.perf_counter()
    region = client.xai_region(
        dermoscopic_path=case.dermoscopic_path, expected_dermoscopic_sha256=case.dermoscopic_sha256,
        clinical_path=case.clinical_path, expected_clinical_sha256=case.clinical_sha256)
    latencies["region_sweep_seconds"] = time.perf_counter() - t1
    peak_rss["after_region_sweep_bytes"] = _peak_rss_bytes(client.pid)

    clinical_result = None
    if case.clinical_path is not None:
        t2 = time.perf_counter()
        clinical_result = client.xai_clinical(
            dermoscopic_path=case.dermoscopic_path, expected_dermoscopic_sha256=case.dermoscopic_sha256,
            clinical_path=case.clinical_path, expected_clinical_sha256=case.clinical_sha256,
            clinical_availability="present")
        latencies["clinical_contribution_seconds"] = time.perf_counter() - t2
    peak_rss["final_bytes"] = _peak_rss_bytes(client.pid)
    client.shutdown()

    calibrators = json.loads((ROOT / "research/experiments/dermpact_phase3/calibration_v1/calibrators.json").read_text())
    parity = None
    if region["status"] == "complete":
        parity = xai.no_op_parity(prediction, calibrators, region["no_op_member_raw"])

    selection = json.loads((ROOT / SELECTION_MANIFEST).read_text())
    checkpoint_bindings = selection["artifacts"]["stage2"]["fold_checkpoints"]
    checkpoint_check = {
        "all_match_selection_manifest": all(
            digest(ROOT / b["path"]) == b["sha256"] for b in checkpoint_bindings),
        "n_checkpoints": len(checkpoint_bindings),
    }

    write_json(out_dir / "prediction.json", prediction)
    write_json(out_dir / "region_receipt.json", region)
    if clinical_result is not None:
        write_json(out_dir / "clinical_receipt.json", clinical_result)
    feasibility = {
        "case_id": case.group_id, "measured_at": _now_iso(),
        "n_members": 5, "members_measured": len(prediction["checkpoint_bindings"]),
        "all_five_checkpoints_loaded": len(prediction["checkpoint_bindings"]) == 5,
        "clinical_available": case.clinical_path is not None,
        "latencies": latencies, "peak_rss": peak_rss,
        "no_op_parity": parity, "checkpoint_integrity_after_run": checkpoint_check,
        "xai_protocol_version": XAI_PROTOCOL_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
        "scope": "one reserved development case; feasibility/latency measurement only",
    }
    write_json(out_dir / "feasibility.json", feasibility)
    return feasibility


def build_inventory(out_dir: Path) -> dict:
    """Runs the real five-member SelectedPerception.predict + reliability decision for all 50 reserved development cases, one worker process for the whole run (loaded once)."""
    from dermpact.reliability.bundle import ReliabilityBundle
    from dermpact.core.versions import RELIABILITY_BUNDLE

    out_dir = Path(out_dir)
    predictions_dir = out_dir / "predictions"
    cases = _resolve_reserved_cases()
    bundle = ReliabilityBundle(ROOT / RELIABILITY_BUNDLE)
    client = WorkerClient()
    manifest_rows = []
    try:
        for case_id in sorted(cases):
            case = cases[case_id]
            if digest(case.dermoscopic_path) != case.dermoscopic_sha256:
                raise AssertionError(f"reserved image changed: {case_id}")
            if case.clinical_path is not None and digest(case.clinical_path) != case.clinical_sha256:
                raise AssertionError(f"reserved clinical image changed: {case_id}")
            started = time.perf_counter()
            dermoscopic_payload = case.dermoscopic_path.read_bytes()
            clinical_payload = case.clinical_path.read_bytes() if case.clinical_path is not None else None
            prediction = client.predict(
                case_id=case_id, dermoscopic_path=case.dermoscopic_path,
                expected_dermoscopic_sha256=case.dermoscopic_sha256,
                clinical_path=case.clinical_path, expected_clinical_sha256=case.clinical_sha256, check_view=True)
            decision = bundle.assess_selected_prediction(dermoscopic_payload, prediction)
            runtime_seconds = time.perf_counter() - started
            payload = {
                "prediction": prediction, "decision": decision.to_dict(),
                "case_meta": {
                    "case_id": case_id, "fold": case.fold,
                    "dermoscopic_relative_path": str(Path(case.dermoscopic_path).resolve().relative_to(ROOT)),
                    "dermoscopic_sha256": case.dermoscopic_sha256,
                    "clinical_relative_path": (str(Path(case.clinical_path).resolve().relative_to(ROOT))
                                                if case.clinical_path is not None else None),
                    "clinical_sha256": case.clinical_sha256,
                    "clinical_availability": "present" if prediction["clinical_present"] else (
                        "duplicate_dermoscopy_excluded" if clinical_payload is not None else "not_supplied"),
                    "computed_at": _now_iso(), "runtime_seconds": runtime_seconds,
                },
            }
            write_json(predictions_dir / f"{case_id}.json", payload)
            manifest_rows.append({
                "case_id": case_id, "fold": case.fold, "disposition": decision.disposition,
                "iqa_state": decision.iqa_state, "ood_state": decision.ood_state,
                "view_stability_state": decision.view_stability_state,
                "selective_abstention": decision.selective_abstention,
                "clinical_availability": payload["case_meta"]["clinical_availability"],
                "runtime_seconds": runtime_seconds, "all_five_checkpoints_loaded": len(prediction["checkpoint_bindings"]) == 5,
                "checklist_direct_head_disagree": _checklist_disagrees_with_direct_head(prediction),
            })
            print(f"INVENTORY {len(manifest_rows)}/50 {case_id} disposition={decision.disposition} "
                  f"runtime={runtime_seconds:.1f}s", flush=True)
    finally:
        client.shutdown()

    demo_subset = _choose_demo_subset(manifest_rows)
    manifest = {
        "n_cases": len(manifest_rows), "rows": manifest_rows, "demo_subset": demo_subset,
        "xai_protocol_version": XAI_PROTOCOL_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
        "generated_at": _now_iso(),
    }
    write_json(out_dir / "inventory_manifest.json", manifest)
    return manifest


def build_confirmatory_inventory(out_dir: Path) -> dict:
    """Build a resumable 100-case development inventory for agent evaluation."""
    from dermpact.reliability.bundle import ReliabilityBundle
    from dermpact.core.versions import RELIABILITY_BUNDLE

    out_dir = Path(out_dir)
    predictions_dir = out_dir / "predictions"
    source_dir = ROOT / "research/experiments/dermpact_phase6/demo_inventory_v1/predictions"
    cases = _resolve_confirmatory_cases()
    bundle = ReliabilityBundle(ROOT / RELIABILITY_BUNDLE)
    client = None
    manifest_rows = []
    n_reused = 0
    n_computed = 0
    try:
        for position, case_id in enumerate(sorted(cases), start=1):
            case = cases[case_id]
            destination = predictions_dir / f"{case_id}.json"
            source = source_dir / f"{case_id}.json"
            if destination.is_file():
                payload = json.loads(destination.read_text(encoding="utf-8"))
                mode = "resumed"
            elif source.is_file():
                payload = json.loads(source.read_text(encoding="utf-8"))
                write_json(destination, payload)
                n_reused += 1
                mode = "reused"
            else:
                if digest(case.dermoscopic_path) != case.dermoscopic_sha256:
                    raise AssertionError(f"development image changed: {case_id}")
                if case.clinical_path is not None and digest(case.clinical_path) != case.clinical_sha256:
                    raise AssertionError(f"development clinical image changed: {case_id}")
                if client is None:
                    client = WorkerClient()
                started = time.perf_counter()
                dermoscopic_payload = case.dermoscopic_path.read_bytes()
                clinical_payload = case.clinical_path.read_bytes() if case.clinical_path is not None else None
                prediction = client.predict(
                    case_id=case_id, dermoscopic_path=case.dermoscopic_path,
                    expected_dermoscopic_sha256=case.dermoscopic_sha256,
                    clinical_path=case.clinical_path, expected_clinical_sha256=case.clinical_sha256,
                    check_view=True,
                )
                decision = bundle.assess_selected_prediction(dermoscopic_payload, prediction)
                runtime_seconds = time.perf_counter() - started
                payload = {
                    "prediction": prediction,
                    "decision": decision.to_dict(),
                    "case_meta": {
                        "case_id": case_id,
                        "fold": case.fold,
                        "dermoscopic_relative_path": str(Path(case.dermoscopic_path).resolve().relative_to(ROOT)),
                        "dermoscopic_sha256": case.dermoscopic_sha256,
                        "clinical_relative_path": (
                            str(Path(case.clinical_path).resolve().relative_to(ROOT))
                            if case.clinical_path is not None else None
                        ),
                        "clinical_sha256": case.clinical_sha256,
                        "clinical_availability": "present" if prediction["clinical_present"] else (
                            "duplicate_dermoscopy_excluded" if clinical_payload is not None else "not_supplied"
                        ),
                        "computed_at": _now_iso(),
                        "runtime_seconds": runtime_seconds,
                    },
                }
                write_json(destination, payload)
                n_computed += 1
                mode = "computed"
            if payload.get("case_meta", {}).get("case_id") != case_id:
                raise AssertionError(f"prediction case binding mismatch: {case_id}")
            prediction = payload["prediction"]
            decision = payload["decision"]
            if len(prediction.get("checkpoint_bindings") or []) != 5:
                raise AssertionError(f"not a five-member prediction: {case_id}")
            manifest_rows.append({
                "case_id": case_id,
                "fold": case.fold,
                "disposition": decision["disposition"],
                "iqa_state": decision["iqa_state"],
                "ood_state": decision["ood_state"],
                "view_stability_state": decision["view_stability_state"],
                "selective_abstention": decision["selective_abstention"],
                "clinical_availability": payload["case_meta"]["clinical_availability"],
                "runtime_seconds": payload["case_meta"].get("runtime_seconds"),
                "all_five_checkpoints_loaded": True,
                "checklist_direct_head_disagree": _checklist_disagrees_with_direct_head(prediction),
            })
            print(f"CONFIRMATORY_INVENTORY {position}/100 {case_id} mode={mode}", flush=True)
    finally:
        if client is not None:
            client.shutdown()

    manifest = {
        "n_cases": len(manifest_rows),
        "rows": manifest_rows,
        "demo_subset": _choose_demo_subset(manifest_rows),
        "selection": {
            "method": "retain reserved50; fixed SHA-256 rank within fold; 20 cases per fold",
            "uses_labels": False,
            "seed_string": "dermpact-confirmatory100-v1",
        },
        "reused_existing_predictions": n_reused,
        "new_predictions_computed": n_computed,
        "official_or_final_rows_loaded": 0,
        "xai_protocol_version": XAI_PROTOCOL_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "generated_at": _now_iso(),
    }
    if len(manifest_rows) != 100:
        raise AssertionError("confirmatory inventory is incomplete")
    write_json(out_dir / "inventory_manifest.json", manifest)
    return manifest


def _choose_demo_subset(rows: list[dict]) -> list[dict]:
    picks: list[dict] = []
    picked_ids: set[str] = set()

    def pick(predicate, reason):
        for row in rows:
            if row["case_id"] in picked_ids:
                continue
            if predicate(row):
                picks.append({"case_id": row["case_id"], "reason": reason})
                picked_ids.add(row["case_id"])
                return True
        return False

    pick(lambda r: r["case_id"] == "derm7pt-case-22", "Phase 3 smoke case; used for the WP-X feasibility receipt")
    pick(lambda r: r["disposition"] == "display", "covers the plain display disposition")
    pick(lambda r: r["selective_abstention"] == "deferred", "covers member-deferred selective abstention")
    pick(lambda r: r["ood_state"] == "flagged", "covers an OOD-flagged case")
    pick(lambda r: r["view_stability_state"] == "unstable", "covers a view-unstable case")
    pick(lambda r: r.get("checklist_direct_head_disagree") is True,
         "covers a case where the checklist and the direct diagnosis head disagree")
    pick(lambda r: r["clinical_availability"] == "present", "covers a case with a real clinical photo")
    pick(lambda r: r["disposition"] == "review", "covers the review disposition")
    if len(picks) < 8:
        for row in rows:
            if row["case_id"] not in picked_ids:
                picks.append({"case_id": row["case_id"], "reason": "fills out the ~8-case demo subset"})
                picked_ids.add(row["case_id"])
            if len(picks) >= 8:
                break
    return picks


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feasibility", action="store_true")
    parser.add_argument("--build-inventory", action="store_true")
    parser.add_argument("--build-confirmatory-inventory", action="store_true")
    args = parser.parse_args(argv)
    if args.feasibility:
        run_feasibility(ROOT / "research/experiments/dermpact_phase6/xai_feasibility_v1")
        return 0
    if args.build_inventory:
        build_inventory(ROOT / "research/experiments/dermpact_phase6/demo_inventory_v1")
        return 0
    if args.build_confirmatory_inventory:
        build_confirmatory_inventory(
            ROOT / "research/experiments/dermpact_phase7/confirmatory_inventory100_v1"
        )
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
