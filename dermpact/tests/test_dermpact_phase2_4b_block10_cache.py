from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dermpact.eval import dermpact_phase2_4b_block10_cache as cache
from dermpact.eval import dermpact_phase2_4b_inputs as inputs


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fresh_rgb_callable():
    def _convert_to_rgb(image):
        return image

    return _convert_to_rgb


def _preprocess(*, resize: int = 256, red_mean: float = 0.485):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose(
        [
            transforms.Resize(
                resize,
                interpolation=InterpolationMode.BICUBIC,
                max_size=None,
                antialias=True,
            ),
            transforms.CenterCrop(224),
            _fresh_rgb_callable(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(red_mean, 0.456, 0.406),
                std=(0.228, 0.224, 0.225),
                inplace=False,
            ),
        ]
    )


def _write_with_sidecar(path: Path, value: bytes) -> str:
    path.write_bytes(value)
    digest = _sha(value)
    Path(str(path) + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _input_set(tmp_path: Path) -> inputs.Phase24BInputSet:
    facts = inputs.PrediagnosticFacts(
        age_years=None,
        sex="female",
        site="posterior_torso",
        elevation="flat",
        available=inputs.FactAvailability(False, True, True, True),
    )
    cases = []
    for index, value in enumerate((1, 2, None, 4)):
        derm = tmp_path / f"derm_{index}.jpg"
        derm_bytes = f"derm-{index}".encode()
        derm.write_bytes(derm_bytes)
        clinical = None if value is None else tmp_path / f"clinical_{index}.jpg"
        clinical_bytes = None if value is None else bytes([value])
        if clinical is not None and clinical_bytes is not None:
            clinical.write_bytes(clinical_bytes)
        cases.append(
            inputs.Phase24BCaseInput(
                case_index=index,
                case_num=index + 1,
                group_id=f"group-{index}",
                fold=index % 2,
                source_split="train",
                dermoscopic_path=derm,
                dermoscopic_relative_path=derm.name,
                dermoscopic_sha256=_sha(derm_bytes),
                clinical_path=clinical,
                clinical_relative_path=None if clinical is None else clinical.name,
                clinical_sha256=None if clinical_bytes is None else _sha(clinical_bytes),
                clinical_available=clinical is not None,
                clinical_unavailable_reason="dermoscopy_reused" if clinical is None else None,
                facts=facts,
            )
        )
    artifacts = []
    for name in ("inventory.json", "folds.json", "audit.json", "facts.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        artifacts.append(path)
    return inputs.Phase24BInputSet(
        cases=tuple(cases),
        inventory_path=artifacts[0],
        inventory_sha256=inputs.sha256_file(artifacts[0]),
        fold_manifest_path=artifacts[1],
        fold_manifest_sha256=inputs.sha256_file(artifacts[1]),
        clinical_photo_audit_path=artifacts[2],
        clinical_photo_audit_sha256=inputs.sha256_file(artifacts[2]),
        structured_fact_support_path=artifacts[3],
        structured_fact_support_sha256=inputs.sha256_file(artifacts[3]),
        images_root=tmp_path,
        hard_scope={
            "derm7pt_official_test_rows_loaded": 0,
            "ham_calibration_or_final_rows_loaded": 0,
            "bcn_validation_or_final_rows_loaded": 0,
            "ph2_rows_loaded": 0,
        },
    )


def _config() -> inputs.Block10CacheConfig:
    return inputs.Block10CacheConfig(
        model_id="pinned/model",
        model_revision="revision-1",
        checkpoint_sha256="a" * 64,
        preprocessing_sha256="b" * 64,
        hidden_size=2,
        tokens_per_image=3,
    )


def _encoder(calls: list[list[str]]) -> cache.PrefixEncoder:
    def values(paths):
        calls.append([Path(path).name for path in paths])
        rows = []
        for path in paths:
            value = float(Path(path).read_bytes()[0])
            rows.append(np.full((3, 2), value, dtype="float32"))
        return np.stack(rows)

    def direct(paths):
        return np.asarray(
            [
                [float(Path(path).read_bytes()[0]), float(Path(path).read_bytes()[0]) * 2]
                for path in paths
            ],
            dtype="float32",
        )

    def reconstruct(tokens):
        scalar = np.asarray(tokens, dtype="float32")[:, 0, 0]
        return np.stack((scalar, scalar * 2), axis=1).astype("float32")

    return cache.PrefixEncoder(
        config=_config(),
        provenance={"loader": "mock-pinned"},
        source_hashes={"cache_source": "c" * 64, "prefix_source": "d" * 64},
        preprocess_identity={"sha256": "b" * 64, "representation": "mock"},
        encode_prefix=values,
        direct_features=direct,
        reconstruct_features=reconstruct,
    )


def _derm_cache(tmp_path: Path, resolved: inputs.Phase24BInputSet):
    config = _config()
    token_path = tmp_path / "derm_tokens.npy"
    np.save(token_path, np.zeros((4, 3, 2), dtype="float32"), allow_pickle=False)
    token_sha = inputs.sha256_file(token_path)
    identity = inputs.canonical_hash(
        [
            {"id": case.group_id, "image_sha256": case.dermoscopic_sha256}
            for case in resolved.cases
        ]
    )
    manifest = {
        "record_identity_sha256": identity,
        "shape": [4, 3, 2],
        "dtype": "float32",
        "completed_visual_blocks": 10,
        "token_file_sha256": token_sha,
        "model": {
            "model_id": config.model_id,
            "checkpoint": {
                "snapshot_revision": config.model_revision,
                "combined_sha256": config.checkpoint_sha256,
            },
        },
    }
    manifest_path = tmp_path / "derm_manifest.json"
    _write_with_sidecar(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return token_path, manifest_path


def _complete_cache(tmp_path: Path):
    resolved = _input_set(tmp_path)
    token_path, manifest_path = _derm_cache(tmp_path, resolved)
    binding = cache.bind_existing_dermoscopic_cache(
        resolved, _config(), token_path=token_path, manifest_path=manifest_path
    )
    output = tmp_path / "output"
    encoder = _encoder([])
    cache.extract_clinical_prefix_cache(
        resolved,
        encoder,
        output_dir=output,
        derm_binding=binding,
        frozen_spec_sha256="9" * 64,
        batch_size=2,
        require_production_counts=False,
    )
    return resolved, binding, encoder, cache.CachePaths.under(output)


def _resign_manifest(paths: cache.CachePaths, manifest: dict) -> None:
    _write_with_sidecar(
        paths.manifest,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )


def test_existing_dermoscopic_cache_is_hash_bound_not_recomputed(tmp_path: Path) -> None:
    resolved = _input_set(tmp_path)
    token_path, manifest_path = _derm_cache(tmp_path, resolved)
    binding = cache.bind_existing_dermoscopic_cache(
        resolved,
        _config(),
        token_path=token_path,
        manifest_path=manifest_path,
    )
    assert binding["token_sha256"] == inputs.sha256_file(token_path)
    assert binding["manifest_sha256"] == inputs.sha256_file(manifest_path)
    assert "never recompute" in binding["reuse_policy"]

    with token_path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(AssertionError, match="token hash"):
        cache.bind_existing_dermoscopic_cache(
            resolved,
            _config(),
            token_path=token_path,
            manifest_path=manifest_path,
        )


def test_preprocess_identity_is_structural_and_process_address_independent() -> None:
    first_pipeline = _preprocess()
    second_pipeline = _preprocess()
    assert repr(first_pipeline) != repr(second_pipeline)
    first = cache._preprocess_identity(first_pipeline)
    second = cache._preprocess_identity(second_pipeline)
    assert first == second
    assert first["sha256"] == second["sha256"]
    serialized = json.dumps(first, sort_keys=True)
    assert "0x" not in serialized
    assert "torchvision.transforms.transforms.Resize" in serialized
    assert '"antialias": true' in serialized
    assert "BICUBIC" in serialized


def test_preprocess_identity_changes_with_model_affecting_parameters() -> None:
    baseline = cache._preprocess_identity(_preprocess())
    resized = cache._preprocess_identity(_preprocess(resize=257))
    renormalized = cache._preprocess_identity(_preprocess(red_mean=0.486))
    assert baseline["sha256"] != resized["sha256"]
    assert baseline["sha256"] != renormalized["sha256"]


def test_preprocess_identity_matches_across_fresh_python_processes() -> None:
    program = r'''
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dermpact.eval.dermpact_phase2_4b_block10_cache import _preprocess_identity

def make_rgb_callable():
    def _convert_to_rgb(image):
        return image
    return _convert_to_rgb

pipeline = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(224),
    make_rgb_callable(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
])
print(_preprocess_identity(pipeline)["sha256"])
'''
    first = subprocess.check_output(
        [sys.executable, "-B", "-c", program], text=True
    ).strip()
    second = subprocess.check_output(
        [sys.executable, "-B", "-c", program], text=True
    ).strip()
    assert first == second
    assert len(first) == 64


def test_interrupted_extraction_resumes_and_preserves_exact_zero_rows(
    tmp_path: Path,
) -> None:
    resolved = _input_set(tmp_path)
    token_path, manifest_path = _derm_cache(tmp_path, resolved)
    binding = cache.bind_existing_dermoscopic_cache(
        resolved, _config(), token_path=token_path, manifest_path=manifest_path
    )
    output = tmp_path / "output"
    calls: list[list[str]] = []
    interrupted = False

    def stop_once(_receipt):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated power loss")

    with pytest.raises(RuntimeError, match="power loss"):
        cache.extract_clinical_prefix_cache(
            resolved,
            _encoder(calls),
            output_dir=output,
            derm_binding=binding,
            frozen_spec_sha256="e" * 64,
            batch_size=2,
            require_production_counts=False,
            on_chunk_complete=stop_once,
        )
    paths = cache.CachePaths.under(output)
    assert paths.partial_tokens.is_file()
    assert paths.progress.is_file()
    assert len(list(paths.receipts_dir.glob("chunk_*.json"))) == 1
    assert calls == [["clinical_0.jpg", "clinical_1.jpg"]]

    result = cache.extract_clinical_prefix_cache(
        resolved,
        _encoder(calls),
        output_dir=output,
        derm_binding=binding,
        frozen_spec_sha256="e" * 64,
        batch_size=2,
        require_production_counts=False,
    )
    assert calls[1] == ["clinical_3.jpg"]
    values = np.load(paths.clinical_tokens, allow_pickle=False)
    assert values.shape == (4, 3, 2)
    assert values.dtype == np.float32
    assert np.all(values[2] == 0.0)
    assert np.all(values[0] == 1.0)
    assert np.all(values[1] == 2.0)
    assert np.all(values[3] == 4.0)
    assert np.array_equal(
        np.load(paths.clinical_available, allow_pickle=False),
        np.asarray([True, True, False, True]),
    )
    assert result["canonical_parity"]["status"] == "PASS"
    assert result["counts"]["unique_clinical_photos_encoded"] == 3
    assert result["counts"]["unavailable_zero_rows"] == 1
    assert result["hard_scope"]["derm7pt_official_test_rows_loaded"] == 0
    assert result["hard_scope"]["ham_calibration_or_final_rows_loaded"] == 0
    assert result["contract"]["encoder_config"]["checkpoint_sha256"] == "a" * 64
    assert result["contract"]["encoder_config"]["preprocessing_sha256"] == "b" * 64
    assert "clinical_path_identity_sha256" in result["contract"]
    assert "fold_identity_sha256" in result["contract"]
    for path in (
        paths.clinical_tokens,
        paths.clinical_available,
        paths.row_identity,
        paths.manifest,
        paths.progress,
    ):
        assert Path(str(path) + ".sha256").is_file()

    reused = cache.extract_clinical_prefix_cache(
        resolved,
        _encoder(calls),
        output_dir=output,
        derm_binding=binding,
        frozen_spec_sha256="e" * 64,
        batch_size=2,
        require_production_counts=False,
    )
    assert reused["status"] == "REUSED_VERIFIED"
    assert calls == [
        ["clinical_0.jpg", "clinical_1.jpg"],
        ["clinical_3.jpg"],
    ]


def test_resume_rejects_tampered_chunk_even_with_valid_receipt_sidecar(
    tmp_path: Path,
) -> None:
    resolved = _input_set(tmp_path)
    token_path, manifest_path = _derm_cache(tmp_path, resolved)
    binding = cache.bind_existing_dermoscopic_cache(
        resolved, _config(), token_path=token_path, manifest_path=manifest_path
    )
    output = tmp_path / "output"

    with pytest.raises(RuntimeError):
        cache.extract_clinical_prefix_cache(
            resolved,
            _encoder([]),
            output_dir=output,
            derm_binding=binding,
            frozen_spec_sha256="f" * 64,
            batch_size=1,
            require_production_counts=False,
            on_chunk_complete=lambda receipt: (_ for _ in ()).throw(RuntimeError("stop")),
        )
    paths = cache.CachePaths.under(output)
    partial = np.lib.format.open_memmap(paths.partial_tokens, mode="r+")
    partial[0, 0, 0] += 1.0
    partial.flush()
    del partial
    with pytest.raises(AssertionError, match="chunk hash"):
        cache.extract_clinical_prefix_cache(
            resolved,
            _encoder([]),
            output_dir=output,
            derm_binding=binding,
            frozen_spec_sha256="f" * 64,
            batch_size=1,
            require_production_counts=False,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "zero_row",
        "available_token",
        "availability",
        "row_order",
        "parity",
        "derm_binding",
        "hard_scope",
    ),
)
def test_completed_reuse_rejects_resigned_but_semantically_invalid_artifacts(
    tmp_path: Path, tamper: str
) -> None:
    resolved, binding, encoder, paths = _complete_cache(tmp_path)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    expected_error = ""
    if tamper == "zero_row":
        values = np.load(paths.clinical_tokens, allow_pickle=False)
        values[2, 0, 0] = 123.0
        with paths.clinical_tokens.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        digest = _write_with_sidecar(
            paths.clinical_tokens, paths.clinical_tokens.read_bytes()
        )
        manifest["artifacts"]["clinical_tokens"]["sha256"] = digest
        expected_error = "exact zeros"
    elif tamper == "available_token":
        values = np.load(paths.clinical_tokens, allow_pickle=False)
        values[0, 0, 0] = 123.0
        with paths.clinical_tokens.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        digest = _write_with_sidecar(
            paths.clinical_tokens, paths.clinical_tokens.read_bytes()
        )
        manifest["artifacts"]["clinical_tokens"]["sha256"] = digest
        expected_error = "parity revalidation"
    elif tamper == "availability":
        availability = np.load(paths.clinical_available, allow_pickle=False)
        availability[2] = True
        with paths.clinical_available.open("wb") as handle:
            np.save(handle, availability, allow_pickle=False)
        digest = _write_with_sidecar(
            paths.clinical_available, paths.clinical_available.read_bytes()
        )
        manifest["artifacts"]["clinical_available"]["sha256"] = digest
        expected_error = "availability mask content"
    elif tamper == "row_order":
        row_payload = json.loads(paths.row_identity.read_text(encoding="utf-8"))
        row_payload["rows"][0], row_payload["rows"][1] = (
            row_payload["rows"][1],
            row_payload["rows"][0],
        )
        digest = _write_with_sidecar(
            paths.row_identity,
            (json.dumps(row_payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        manifest["artifacts"]["row_identity"]["sha256"] = digest
        expected_error = "row-identity order"
    elif tamper == "parity":
        manifest["canonical_parity"]["maximum_abs_difference"] = 1.0
        expected_error = "exceeds tolerance"
    elif tamper == "derm_binding":
        manifest["artifacts"].pop("dermoscopic_tokens")
        expected_error = "dermoscopic binding"
    else:
        manifest["hard_scope"]["derm7pt_official_test_rows_loaded"] = 1
        expected_error = "hard-scope"
    _resign_manifest(paths, manifest)

    with pytest.raises(AssertionError, match=expected_error):
        cache.extract_clinical_prefix_cache(
            resolved,
            encoder,
            output_dir=paths.output_dir,
            derm_binding=binding,
            frozen_spec_sha256="9" * 64,
            batch_size=2,
            require_production_counts=False,
        )


def test_describe_mode_is_no_write(capsys) -> None:
    assert cache.main(["--describe"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["expected_shape"] == [616, 197, 768]
    assert payload["writes_performed"] == 0
