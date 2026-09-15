"""Measured explanations: region sensitivity and clinical-photo contribution."""
from __future__ import annotations

import hashlib
import io
import math
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from dermpact.eval.dermpact_phase2_5_window_cache import GRID_SIZE, TARGET_SIZE, WINDOW_SIZE
from dermpact.eval.dermpact_phase2_baselines import CRITERIA
from dermpact.reliability.inference import calibrated_fold_probabilities, load_selected_fold
from dermpact.core.versions import XAI_PROTOCOL_VERSION


N_REGIONS = GRID_SIZE * GRID_SIZE  # 16
METHODS = ("local_mean", "gaussian_blur")
GAUSSIAN_SIGMA_PX = 12.0
DIAGNOSIS_MELANOMA_INDEX = 0
BINARY_MELANOMA_INDEX = 0
TARGETS = ("melanoma_head", "diagnosis_head_melanoma") + tuple(f"criterion_{c}" for c in CRITERIA)
GRID_GEOMETRY_SIGNATURE = f"{GRID_SIZE}x{GRID_SIZE}@{WINDOW_SIZE}px_pad{TARGET_SIZE}"
OOD_QUALIFICATION = (
    "masked images may be out of distribution; sensitivity is a model "
    "measurement, not clinical causality"
)
CLINICAL_QUALIFICATION = (
    "missing-modality intervention; does not prove causal clinical necessity "
    "or isolate image quality from visual information"
)


class Cancelled(Exception):
    """Raised when a cooperative cancel flag is observed mid-sweep."""


class ReceiptMismatch(ValueError):
    """A receipt's cache key does not match the record it is being checked against."""



def _resize_and_pad_params(image_size: tuple[int, int]) -> dict[str, Any]:
    w, h = image_size
    if w < 1 or h < 1:
        raise ValueError("image size must be positive")
    scale = TARGET_SIZE / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    pad_top = (TARGET_SIZE - new_h) // 2
    pad_bottom = TARGET_SIZE - new_h - pad_top
    pad_left = (TARGET_SIZE - new_w) // 2
    pad_right = TARGET_SIZE - new_w - pad_left
    return {
        "scale": scale, "new_w": new_w, "new_h": new_h,
        "pad_top": pad_top, "pad_bottom": pad_bottom, "pad_left": pad_left, "pad_right": pad_right,
        "valid_top": pad_top, "valid_bottom": TARGET_SIZE - pad_bottom,
        "valid_left": pad_left, "valid_right": TARGET_SIZE - pad_right,
    }


def _map_padded_box_to_original(box, params, image_size):
    """box = (y0, y1, x0, x1) in padded-square pixels."""
    w, h = image_size
    y0, y1, x0, x1 = box
    iy0, iy1 = max(y0, params["valid_top"]), min(y1, params["valid_bottom"])
    ix0, ix1 = max(x0, params["valid_left"]), min(x1, params["valid_right"])
    if iy0 >= iy1 or ix0 >= ix1:
        return None
    scale = params["scale"]
    ox0 = max(0.0, (ix0 - params["pad_left"]) / scale)
    ox1 = min(float(w), (ix1 - params["pad_left"]) / scale)
    oy0 = max(0.0, (iy0 - params["pad_top"]) / scale)
    oy1 = min(float(h), (iy1 - params["pad_top"]) / scale)
    if ox1 <= ox0 or oy1 <= oy0:
        return None
    return (ox0, oy0, ox1, oy1)


def compute_region_geometry(image_size: tuple[int, int]) -> list[dict[str, Any]]:
    """16 regions tiling the padded square, in the same row-major order as dermpact.eval.dermpact_phase2_5_window_cache.window_positions() (index = row*GRID_SIZE + col)."""
    params = _resize_and_pad_params(image_size)
    regions = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y0, y1 = r * WINDOW_SIZE, (r + 1) * WINDOW_SIZE
            x0, x1 = c * WINDOW_SIZE, (c + 1) * WINDOW_SIZE
            original_box = _map_padded_box_to_original((y0, y1, x0, x1), params, image_size)
            area = (original_box[2] - original_box[0]) * (original_box[3] - original_box[1]) if original_box else 0.0
            regions.append({
                "index": r * GRID_SIZE + c, "row": r, "col": c,
                "padded_box": (y0, y1, x0, x1),
                "status": "padding_only" if original_box is None else "active",
                "original_box": original_box, "area_original_px": area,
            })
    return regions


def equal_area_control_geometry(image_size: tuple[int, int]) -> list[dict[str, Any]]:
    """One equal-area (WINDOW_SIZE x WINDOW_SIZE in padded-square pixels) control box per region, near the padded-square centre, offset from the centre per region so 16 distinct (but centrally located) sanity controls are produced rather than the same fixed region reused 16 times."""
    params = _resize_and_pad_params(image_size)
    grid_mid = (GRID_SIZE - 1) / 2.0
    step = WINDOW_SIZE // 4
    controls = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            dy = int(round((r - grid_mid) * step))
            dx = int(round((c - grid_mid) * step))
            cy0 = max(0, min(TARGET_SIZE - WINDOW_SIZE, TARGET_SIZE // 2 - WINDOW_SIZE // 2 + dy))
            cx0 = max(0, min(TARGET_SIZE - WINDOW_SIZE, TARGET_SIZE // 2 - WINDOW_SIZE // 2 + dx))
            padded_box = (cy0, cy0 + WINDOW_SIZE, cx0, cx0 + WINDOW_SIZE)
            original_box = _map_padded_box_to_original(padded_box, params, image_size)
            area = (original_box[2] - original_box[0]) * (original_box[3] - original_box[1]) if original_box else 0.0
            controls.append({
                "index": r * GRID_SIZE + c, "row": r, "col": c, "padded_box": padded_box,
                "original_box": original_box, "area_original_px": area,
                "status": "padding_only" if original_box is None else "active",
            })
    return controls



def _box_to_pixel_bounds(original_box, width, height):
    x0, y0, x1, y1 = original_box
    ix0 = max(0, int(math.floor(x0)))
    iy0 = max(0, int(math.floor(y0)))
    ix1 = min(width, int(math.ceil(x1)))
    iy1 = min(height, int(math.ceil(y1)))
    return ix0, iy0, ix1, iy1


def apply_local_mean(image: Image.Image, original_box) -> Image.Image:
    arr = np.array(image, dtype=np.float64)
    ix0, iy0, ix1, iy1 = _box_to_pixel_bounds(original_box, image.width, image.height)
    if ix1 <= ix0 or iy1 <= iy0:
        return image.copy()
    region = arr[iy0:iy1, ix0:ix1]
    mean = region.reshape(-1, region.shape[-1]).mean(axis=0)
    arr[iy0:iy1, ix0:ix1] = mean
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode=image.mode)


def apply_gaussian_blur(image: Image.Image, original_box, sigma: float = GAUSSIAN_SIGMA_PX) -> Image.Image:
    arr = np.array(image, dtype=np.float64)
    ix0, iy0, ix1, iy1 = _box_to_pixel_bounds(original_box, image.width, image.height)
    if ix1 <= ix0 or iy1 <= iy0:
        return image.copy()
    region = arr[iy0:iy1, ix0:ix1]
    blurred = np.stack([gaussian_filter(region[..., ch], sigma=sigma) for ch in range(region.shape[-1])], axis=-1)
    arr[iy0:iy1, ix0:ix1] = blurred
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode=image.mode)


def apply_method(image: Image.Image, original_box, method: str) -> Image.Image:
    if method == "local_mean":
        return apply_local_mean(image, original_box)
    if method == "gaussian_blur":
        return apply_gaussian_blur(image, original_box)
    raise ValueError(f"unknown perturbation method: {method!r}")



class PerceptionRuntime(Protocol):
    """What xai.py needs from a loaded model."""

    calibrators: Mapping[str, Any]

    def prefix(self, image: Image.Image, clinical: Optional[Image.Image]) -> tuple: ...

    def fold_iter(self) -> Iterable[tuple[int, Callable[..., Mapping[str, Any]], Mapping[str, Any]]]: ...


class RealPerceptionRuntime:
    """Adapts a live ``SelectedPerception`` instance to ``PerceptionRuntime``."""

    def __init__(self, engine):
        self.engine = engine
        self.calibrators = engine.calibrators

    def prefix(self, image, clinical):
        return self.engine._prefix(image, clinical)

    def fold_iter(self):
        for fold in range(5):
            model, _, binding = load_selected_fold(fold)
            try:
                yield fold, model, binding
            finally:
                del model


def _pil_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _identity_calibrators(n_folds: int = 5) -> dict:
    return {"folds": {str(f): {"temperature": {"value": 1.0, "status": "identity"},
                                "binary_temperatures": [{"value": 1.0, "status": "identity"} for _ in range(8)]}
                       for f in range(n_folds)}}


class FakePerceptionRuntime:
    """Deterministic stand-in for a real ``SelectedPerception``."""

    def __init__(self, n_folds: int = 5, fail_folds: Sequence[int] = (), calibrators: Optional[dict] = None):
        self.n_folds = n_folds
        self.fail_folds = set(fail_folds)
        self.calibrators = calibrators or _identity_calibrators(n_folds)
        self.prefix_calls: list[dict] = []

    def prefix(self, image, clinical):
        image_sha = hashlib.sha256(_pil_bytes(image)).hexdigest()
        clinical_sha = hashlib.sha256(_pil_bytes(clinical)).hexdigest() if clinical is not None else None
        self.prefix_calls.append({
            "image_sha256": image_sha, "clinical_sha256": clinical_sha,
            "streams_recomputed": ("global", "windows", "frozen_baseline"),
        })
        image_seed = int(image_sha[:8], 16)
        clinical_seed = int(clinical_sha[:8], 16) if clinical_sha else 0
        return (image_seed, clinical_seed, clinical is not None)

    def fold_iter(self):
        for fold in range(self.n_folds):
            if fold in self.fail_folds:
                def broken(*_args, _fold=fold, **_kw):
                    raise RuntimeError(f"forced failure on fold {_fold}")
                yield fold, broken, {"fold": fold, "path": f"fake/fold_{fold}.pt", "sha256": format(fold, "064x")}
                continue

            def fake_model(image_seed, clinical_seed, clinical_present, _fold=fold):
                rng = np.random.default_rng((image_seed * 1_000_003 + clinical_seed * 7919 + _fold) % (2 ** 32))
                diagnosis = rng.normal(size=9)[None, :]
                melanoma = rng.normal(size=1)
                criteria = rng.normal(size=7)[None, :]
                if clinical_present:
                    melanoma = melanoma + 0.05  # detectable, deterministic clinical-photo effect
                return {"diagnosis": diagnosis, "melanoma": melanoma, "criteria": criteria}
            yield fold, fake_model, {"fold": fold, "path": f"fake/fold_{fold}.pt", "sha256": format(fold, "064x")}



def _head_values(member_calibrated: Mapping[str, Any]) -> dict[str, float]:
    binary = member_calibrated["binary_probabilities"]
    diag = member_calibrated["diagnosis_probabilities"]
    values = {"melanoma_head": float(binary[BINARY_MELANOMA_INDEX]),
              "diagnosis_head_melanoma": float(diag[DIAGNOSIS_MELANOMA_INDEX])}
    for i, name in enumerate(CRITERIA):
        values[f"criterion_{name}"] = float(binary[1 + i])
    return values


def _calibrated_member(raw_output, fold, calibrators) -> dict[str, Any]:
    diag, binary = calibrated_fold_probabilities(raw_output, fold, calibrators)
    return {"diagnosis_probabilities": np.asarray(diag)[0].tolist(),
            "binary_probabilities": np.asarray(binary)[0].tolist()}



def run_region_sensitivity(runtime: PerceptionRuntime, image: Image.Image, clinical: Optional[Image.Image], *,
                            methods: Sequence[str] = METHODS, seed: int = 0,
                            cancel_check: Optional[Callable[[], bool]] = None,
                            progress_cb: Optional[Callable[[dict], None]] = None) -> dict:
    """One full region-sensitivity sweep."""
    w, h = image.size
    regions = compute_region_geometry((w, h))
    controls = equal_area_control_geometry((w, h))
    perturbations = [{"id": "no_op", "kind": "no_op", "region_index": None, "method": None, "image": image}]
    for region, control in zip(regions, controls):
        if region["status"] == "padding_only":
            continue
        for method in methods:
            perturbations.append({
                "id": f"region_{region['index']}_{method}", "kind": "primary",
                "region_index": region["index"], "method": method,
                "image": apply_method(image, region["original_box"], method),
            })
            if control["status"] == "active":
                perturbations.append({
                    "id": f"control_{region['index']}_{method}", "kind": "equal_area_control",
                    "region_index": region["index"], "method": method,
                    "image": apply_method(image, control["original_box"], method),
                })

    prefixes_by_id = {}
    for p in perturbations:
        if cancel_check is not None and cancel_check():
            raise Cancelled("cancelled during prefix computation")
        prefixes_by_id[p["id"]] = runtime.prefix(p["image"], clinical)

    member_raw: dict[int, dict[str, dict]] = {}
    failures: dict[str, str] = {}
    for fold, model, _binding in runtime.fold_iter():
        if cancel_check is not None and cancel_check():
            raise Cancelled(f"cancelled before fold {fold}")
        try:
            per_perturbation = {}
            for p in perturbations:
                out = model(*prefixes_by_id[p["id"]])
                per_perturbation[p["id"]] = _calibrated_member(out, fold, runtime.calibrators)
            member_raw[fold] = per_perturbation
            if progress_cb is not None:
                progress_cb({"event": "progress", "stage": "region_sensitivity", "fold": fold, "n_folds": 5})
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - persisted as a structured failure, not raised
            failures[str(fold)] = str(exc)
        finally:
            del model

    return _assemble_region_result(perturbations, regions, member_raw, failures, methods)


def _assemble_region_result(perturbations, regions, member_raw, failures, methods) -> dict:
    members_measured = len(member_raw)
    if failures or members_measured != 5:
        return {"status": "failed", "n_members": 5, "members_measured": members_measured,
                "failures": failures, "results": None, "heatmap": None,
                "padding_only_regions": [r["index"] for r in regions if r["status"] == "padding_only"],
                "no_op_member_raw": None}

    fold_ids = sorted(member_raw)
    baseline_values = {fold: _head_values(member_raw[fold]["no_op"]) for fold in fold_ids}
    entries: dict[str, dict] = {}
    for p in perturbations:
        if p["kind"] == "no_op":
            continue
        targets = {}
        for target in TARGETS:
            per_member_delta = []
            for fold in fold_ids:
                orig = baseline_values[fold][target]
                pert = _head_values(member_raw[fold][p["id"]])[target]
                per_member_delta.append(orig - pert)
            targets[target] = {
                "aggregated_delta": float(np.mean(per_member_delta)),
                "per_member_delta": [float(x) for x in per_member_delta],
            }
        entries[p["id"]] = {"region_index": p["region_index"], "method": p["method"], "kind": p["kind"], "targets": targets}

    ranked = []
    for region in regions:
        if region["status"] == "padding_only":
            continue
        idx = region["index"]
        primary_id = f"region_{idx}_local_mean"
        if primary_id not in entries:
            continue
        by_method = {m: entries[f"region_{idx}_{m}"] for m in methods if f"region_{idx}_{m}" in entries}
        control_by_method = {m: entries[f"control_{idx}_{m}"] for m in methods if f"control_{idx}_{m}" in entries}
        deltas_across_methods = [by_method[m]["targets"]["melanoma_head"]["aggregated_delta"] for m in methods if m in by_method]
        signs = {(1 if d > 0 else (-1 if d < 0 else 0)) for d in deltas_across_methods}
        ranked.append({
            "region_index": idx, "row": region["row"], "col": region["col"],
            "original_box_original_px": region["original_box"], "area_original_px": region["area_original_px"],
            "rank_key_local_mean_melanoma_head_abs_delta": abs(entries[primary_id]["targets"]["melanoma_head"]["aggregated_delta"]),
            "methods": by_method, "equal_area_controls": control_by_method,
            "sign_agreement_across_methods": len(signs) <= 1,
            "std_across_methods_melanoma_head": float(np.std(deltas_across_methods)) if len(deltas_across_methods) > 1 else 0.0,
        })
    ranked.sort(key=lambda r: r["rank_key_local_mean_melanoma_head_abs_delta"], reverse=True)

    heatmap = {}
    for method in methods:
        heatmap[method] = {}
        for target in ("melanoma_head", "diagnosis_head_melanoma"):
            grid = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
            for region in regions:
                pid = f"region_{region['index']}_{method}"
                if pid in entries:
                    grid[region["row"]][region["col"]] = entries[pid]["targets"][target]["aggregated_delta"]
            heatmap[method][target] = grid

    return {
        "status": "complete", "n_members": 5, "members_measured": members_measured, "failures": {},
        "results": ranked, "heatmap": heatmap,
        "padding_only_regions": [r["index"] for r in regions if r["status"] == "padding_only"],
        "no_op_member_raw": {str(fold): member_raw[fold]["no_op"] for fold in fold_ids},
    }


def no_op_parity(prediction: Mapping[str, Any], calibrators: Mapping[str, Any],
                  no_op_member_raw: Mapping[str, Mapping[str, Any]]) -> dict:
    """Compares the no-op perturbation's per-member calibrated output (computed through the occlusion code path) against the canonical ``SelectedPerception.predict`` per-member calibrated output (recomputed from its stored raw fold logits with the same calibrators)."""
    per_member = {}
    diffs = []
    for fold in range(5):
        raw = prediction["raw_fold_outputs"][fold]
        canonical = _calibrated_member(
            {"diagnosis": np.array([raw["diagnosis"]]), "melanoma": np.array([raw["melanoma"]]),
             "criteria": np.array([raw["criteria"]])}, fold, calibrators)
        observed = no_op_member_raw[str(fold)]
        d_diag = float(np.max(np.abs(np.asarray(canonical["diagnosis_probabilities"]) - np.asarray(observed["diagnosis_probabilities"]))))
        d_bin = float(np.max(np.abs(np.asarray(canonical["binary_probabilities"]) - np.asarray(observed["binary_probabilities"]))))
        per_member[fold] = {"max_abs_diff_diagnosis": d_diag, "max_abs_diff_binary": d_bin}
        diffs.extend([d_diag, d_bin])
    return {"max_abs_diff": max(diffs) if diffs else None, "per_member": per_member, "atol_target": 1e-6}



def run_clinical_contribution(runtime: PerceptionRuntime, image: Image.Image, clinical: Optional[Image.Image],
                               clinical_availability: str, *,
                               cancel_check: Optional[Callable[[], bool]] = None,
                               progress_cb: Optional[Callable[[dict], None]] = None) -> dict:
    """Compares the model with the clinical photo against its supported absent-photo path (clinical=None), all else unchanged."""
    if clinical_availability != "present" or clinical is None:
        return {"status": "not_applicable", "reason": f"clinical_availability={clinical_availability}"}

    prefixes_with = runtime.prefix(image, clinical)
    prefixes_without = runtime.prefix(image, None)
    member_raw = {}
    failures = {}
    for fold, model, _binding in runtime.fold_iter():
        if cancel_check is not None and cancel_check():
            raise Cancelled(f"cancelled before fold {fold}")
        try:
            with_photo = _calibrated_member(model(*prefixes_with), fold, runtime.calibrators)
            without_photo = _calibrated_member(model(*prefixes_without), fold, runtime.calibrators)
            member_raw[fold] = {"with_photo": with_photo, "without_photo": without_photo}
            if progress_cb is not None:
                progress_cb({"event": "progress", "stage": "clinical_contribution", "fold": fold, "n_folds": 5})
        except Exception as exc:  # noqa: BLE001
            failures[str(fold)] = str(exc)
        finally:
            del model

    if failures or len(member_raw) != 5:
        return {"status": "failed", "n_members": 5, "members_measured": len(member_raw), "failures": failures}

    targets = {}
    for target in TARGETS:
        per_member_delta = [
            _head_values(member_raw[fold]["with_photo"])[target] - _head_values(member_raw[fold]["without_photo"])[target]
            for fold in sorted(member_raw)
        ]
        targets[target] = {
            "aggregated_delta": float(np.mean(per_member_delta)),
            "per_member_delta": [float(x) for x in per_member_delta],
            "exact_zero": all(abs(x) < 1e-12 for x in per_member_delta),
        }
    return {"status": "complete", "n_members": 5, "members_measured": 5, "targets": targets,
            "qualification": CLINICAL_QUALIFICATION}



def _record_accessor(record_like: Mapping[str, Any]) -> dict[str, Any]:
    inputs = record_like["inputs"]
    perception = record_like["perception"]
    identity = record_like["identity"]
    clinical = inputs.get("clinical") or {}
    return {
        "dermoscopic_sanitized_sha256": inputs["dermoscopic"]["sanitized_sha256"],
        "clinical_sanitized_sha256": clinical.get("sanitized_sha256"),
        "record_sha256": record_like["record_sha256"],
        "checkpoint_shas": tuple(sorted(b["sha256"] for b in perception["checkpoint_bindings"])),
        "calibration_sha256": perception["calibration_sha256"],
        "case_id": identity["case_id"],
        "record_version": identity["record_version"],
    }


def build_cache_key(*, dermoscopic_sha256, clinical_sha256, case_id, record_version, record_sha256,
                     checkpoint_shas, preprocessing_version, calibration_sha256, target_heads,
                     methods, sigma, seed, grid_geometry_signature: str = GRID_GEOMETRY_SIGNATURE) -> dict:
    return {
        "dermoscopic_sanitized_sha256": dermoscopic_sha256,
        "clinical_sanitized_sha256": clinical_sha256,
        "case_id": case_id, "record_version": record_version, "record_sha256": record_sha256,
        "checkpoint_shas": sorted(checkpoint_shas),
        "preprocessing_version": preprocessing_version, "calibration_sha256": calibration_sha256,
        "target_heads": sorted(target_heads), "grid_geometry_signature": grid_geometry_signature,
        "methods": sorted(methods), "sigma": sigma, "seed": seed,
        "protocol_version": XAI_PROTOCOL_VERSION,
    }


def cache_key_hash(cache_key: Mapping[str, Any]) -> str:
    import json
    body = json.dumps(cache_key, sort_keys=True, allow_nan=False, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _receipt_content_sha256(receipt: Mapping[str, Any]) -> str:
    import json
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    body = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def seal_receipt(receipt: Mapping[str, Any]) -> dict:
    sealed = {k: v for k, v in dict(receipt).items() if k != "receipt_sha256"}
    sealed["receipt_sha256"] = _receipt_content_sha256(sealed)
    return sealed


def verify_receipt_content(receipt: Mapping[str, Any]) -> bool:
    if receipt.get("protocol_version") != XAI_PROTOCOL_VERSION:
        raise ReceiptMismatch("unknown XAI receipt protocol")
    if receipt.get("receipt_id") != cache_key_hash(receipt.get("cache_key") or {}):
        raise ReceiptMismatch("XAI receipt id does not match its cache key")
    if receipt.get("receipt_sha256") != _receipt_content_sha256(receipt):
        raise ReceiptMismatch("XAI receipt content hash mismatch")
    status = receipt.get("status")
    if status not in {"complete", "failed", "cancelled", "not_applicable"}:
        raise ReceiptMismatch(f"unknown XAI receipt status {status!r}")
    kind = receipt.get("kind")
    if kind not in {"region", "clinical"}:
        raise ReceiptMismatch(f"unknown XAI receipt kind {kind!r}")
    if status == "complete":
        if receipt.get("n_members") != 5 or receipt.get("members_measured") != 5 or receipt.get("failures"):
            raise ReceiptMismatch("complete XAI receipt requires all five successful members")
        if kind == "region":
            parity = receipt.get("no_op_parity")
            if not isinstance(parity, Mapping) or parity.get("max_abs_diff") is None:
                raise ReceiptMismatch("complete region receipt requires no-op parity")
            diff = float(parity["max_abs_diff"])
            target = float(parity.get("atol_target", 1e-6))
            if not np.isfinite(diff) or diff > target:
                raise ReceiptMismatch("region no-op parity exceeds its declared tolerance")
            if not receipt.get("results") or not receipt.get("heatmap"):
                raise ReceiptMismatch("complete region receipt lacks measured results/heatmap")
        elif not isinstance(receipt.get("targets"), Mapping) or not receipt.get("targets"):
            raise ReceiptMismatch("complete clinical receipt lacks measured targets")
    return True


def build_region_receipt(*, cache_key: Mapping[str, Any], sweep_result: Mapping[str, Any],
                          started_at: str, finished_at: Optional[str], runtime_seconds: Optional[float],
                          peak_rss_bytes: Optional[int], no_op_parity_result: Optional[dict],
                          status_override: Optional[str] = None) -> dict:
    status = status_override or sweep_result["status"]
    if status == "complete":
        parity_ok = (isinstance(no_op_parity_result, Mapping)
                     and no_op_parity_result.get("max_abs_diff") is not None
                     and np.isfinite(float(no_op_parity_result["max_abs_diff"]))
                     and float(no_op_parity_result["max_abs_diff"]) <= float(no_op_parity_result.get("atol_target", 1e-6)))
        if not parity_ok:
            status = "failed"
    receipt = {
        "receipt_id": cache_key_hash(cache_key), "cache_key": dict(cache_key),
        "protocol_version": XAI_PROTOCOL_VERSION, "kind": "region", "status": status,
        "started_at": started_at, "finished_at": finished_at, "runtime_seconds": runtime_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "n_members": sweep_result.get("n_members", 5), "members_measured": sweep_result.get("members_measured", 0),
        "failures": sweep_result.get("failures") or {}, "ood_qualification": OOD_QUALIFICATION,
        "results": sweep_result.get("results"), "heatmap": sweep_result.get("heatmap"),
        "padding_only_regions": sweep_result.get("padding_only_regions"),
        "no_op_parity": no_op_parity_result,
        "controls": {"description": "equal_area_control per region/method at padded-square centre, region-offset", "included": True},
    }
    return seal_receipt(receipt)


def verify_receipt(receipt: Mapping[str, Any], record_like: Mapping[str, Any]) -> bool:
    """Rejects a receipt bound to a different case, image sha, checkpoint set or calibration sha than the given record."""
    verify_receipt_content(receipt)
    expected = _record_accessor(record_like)
    key = receipt["cache_key"]
    mismatches = []
    if key.get("case_id") != expected["case_id"]:
        mismatches.append("case_id")
    if key.get("dermoscopic_sanitized_sha256") != expected["dermoscopic_sanitized_sha256"]:
        mismatches.append("dermoscopic_sanitized_sha256")
    if key.get("clinical_sanitized_sha256") != expected["clinical_sanitized_sha256"]:
        mismatches.append("clinical_sanitized_sha256")
    if tuple(sorted(key.get("checkpoint_shas") or [])) != expected["checkpoint_shas"]:
        mismatches.append("checkpoint_shas")
    if key.get("calibration_sha256") != expected["calibration_sha256"]:
        mismatches.append("calibration_sha256")
    if mismatches:
        raise ReceiptMismatch(f"receipt does not match record: {', '.join(mismatches)}")
    return True


def receipt_provenance(receipt: Mapping[str, Any], record_like: Mapping[str, Any]) -> dict[str, Any]:
    """Which record version the receipt was measured on versus the current one."""
    expected = _record_accessor(record_like)
    key = receipt["cache_key"]
    return {
        "measured_at_record_version": key.get("record_version"),
        "measured_at_record_sha256": key.get("record_sha256"),
        "current_record_version": expected["record_version"],
        "current_record_sha256": expected["record_sha256"],
        "record_unchanged_since_measurement": key.get("record_sha256") == expected["record_sha256"],
        "note": "attribution depends on image, frozen weights and calibration only; clinician amendments do not alter it",
    }
