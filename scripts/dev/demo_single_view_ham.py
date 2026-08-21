"""DEMO: the single-view (classifier-only) decision report, end-to-end on real HAM10000 images.

HAM10000 carries diagnosis labels but NO 7-point criteria annotations, so the dual-view
DecisionContract cannot exist for it. This script proves the degraded path on real images:

    raw HAM JPG -> frozen DermLIP encode -> promoted-recipe HAM classifier (ADOPTED argmax arm)
    -> build_single_view_contract (hash-pinned guideline binding, mandatory checklist-unavailable
       disclosure, self-verification) -> tiered clinician report (default + on-request appendix).

THIS IS A DEMO ARTIFACT, NOT AN EVALUATION. Two hand-shaped criteria apply and no others:
one true melanoma and one true nevus, each chosen deterministically as the FIRST E3-test image
(manifest order) whose argmax prediction matches its label -- so the report exercises the
guideline-covered path (MEL) and the explicit-absence path (NEV) as designed, not by luck.
No test-set metric is computed or claimed here; the registered numbers live in
research/evidence/ham_v1/ham_evaluation.json and
research/evidence/derm7pt_v1/promoted_recipe_ham_bcn.json.

CLASSIFIER = the ADOPTED arm of the promoted-recipe transfer (finalize_paper_metrics.py:
"promoted recipe at argmax (metadata + train-CV C; NO operating point)" -- the Derm7pt-tuned
operating point collapses HAM melanoma recall and was never adopted). The fitting code path is
REUSED from scripts/dev/promoted_recipe_ham_bcn.py, not reimplemented: same E3 train split
(regenerated via build_ham_evaluation.make_split and asserted equal to the recorded partition),
same dermlip+metadata19 features, same pooled-correct-count C selection at seed 20260812, and the
resulting C is asserted equal to the recorded candidate_b selection before anything renders.

ENCODING PATH: the two query images are encoded FRESH from their raw JPGs with the frozen
DermLIP encoder (the honest raw-image entry), then compared against the cached E3 feature rows
(max |diff| recorded; batched CPU inference can differ from the cache in the last float32 bits).
If the encoder cannot load in this environment the script falls back to the cached rows and says
so in the artifact -- the contract/render half of the demo does not depend on which path ran.

E3 test-side images were already read post-hoc under the user-authorized re-read
(promoted_recipe_ham_bcn.json); this demo re-reads two of them and claims nothing statistical.

Cost: zero paid calls; all offline.

Run::
    venv_win/Scripts/python.exe scripts/dev/demo_single_view_ham.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from medguard_v2.evidence.guideline_binding import GuidelineCatalog  # noqa: E402
from medguard_v2.evidence.render import render_markdown  # noqa: E402
from medguard_v2.evidence.single_view import (  # noqa: E402
    build_single_view_contract,
    single_view_contract_to_dict,
)

OUT = REPO / "research/evidence/derm7pt_v1/single_view_ham_demo.json"

#: The canonical disclosure for HAM/BCN-class sources (see SingleViewContract's field docstring).
CHECKLIST_UNAVAILABLE_REASON = (
    "this imaging source carries no dermoscopic-criteria annotations; single-view decision"
)

#: (demo label, HAM raw label, expected 5-group prediction) for the two showcase cases.
PICKS: Tuple[Tuple[str, str, str], ...] = (
    ("melanoma (guideline-covered path)", "melanoma", "MEL"),
    ("nevus (explicit-absence path)", "nevus", "NEV"),
)


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fit_adopted_ham_classifier(pr) -> Dict[str, object]:
    """Reproduce the E3 five-group candidate_b (promoted recipe) fit, ARGMAX arm.

    ``pr`` is the imported promoted_recipe_ham_bcn module; every step below calls ITS functions
    (which in turn reuse build_ham_evaluation's split construction), so this is the recorded
    fitting code path, not a lookalike.
    """
    bhe = pr._load_ham_build_module()
    assert dict(bhe.HAM_TO_GROUP) == pr.HAM_TO_GROUP, "HAM 5-group mapping drifted"

    features, records = bhe.load_records()
    cluster_label = bhe.lesion_labels(records)
    train_clusters, test_clusters = bhe.make_split(cluster_label)

    baseline = json.loads(pr.HAM_E3_BASELINE.read_text(encoding="utf-8"))
    rec_split = baseline["split"]
    if train_clusters != rec_split["train_cluster_ids"] or test_clusters != rec_split["test_cluster_ids"]:
        raise SystemExit("E3 split regeneration does not match the recorded partition")

    train_set, test_set = set(train_clusters), set(test_clusters)
    train_idx = [i for i, r in enumerate(records) if str(r["cluster_id"]) in train_set]
    test_idx = [i for i, r in enumerate(records) if str(r["cluster_id"]) in test_set]

    meta_all = pr._ham_meta_matrix(records)
    xb_train = np.hstack([features[train_idx], meta_all[train_idx]])
    y5_train = [pr.HAM_TO_GROUP[str(records[i]["label"])] for i in train_idx]
    clusters_train = [str(records[i]["cluster_id"]) for i in train_idx]

    sel = pr._select_c_pooled_counts(xb_train, y5_train, clusters_train, pr.E3_SEED)

    # Wiring check: the C selection must reproduce the recorded candidate_b selection exactly
    # before this demo is allowed to claim it runs "the adopted classifier".
    recorded = json.loads(pr.OUT.read_text(encoding="utf-8"))
    rec_sel = recorded["ham_e3"]["five_group"]["candidate_b_promoted_recipe"]["c_selection"]
    if sel["selected_C"] != rec_sel["selected_C"] or sel["grid_cv_accuracy"] != rec_sel["grid_cv_accuracy"]:
        raise SystemExit(
            "WIRING DRIFT: E3 candidate_b C-selection did not reproduce "
            f"promoted_recipe_ham_bcn.json ({sel} vs {rec_sel}). Refusing to demo an "
            "untrustworthy classifier."
        )

    model = LogisticRegression(max_iter=5000, C=sel["selected_C"], random_state=pr.E3_SEED)
    model.fit(xb_train, y5_train)

    return {
        "model": model,
        "features": features,
        "meta_all": meta_all,
        "records": records,
        "test_idx": test_idx,
        "c_selection": sel,
        "n_train_images": len(train_idx),
        "n_test_images": len(test_idx),
        "config": (
            "dermlip + metadata19 (531-d), UNBALANCED LR C="
            f"{sel['selected_C']} (train-CV, pooled correct-count, seed {pr.E3_SEED}), "
            "fit on the E3 train split; ADOPTED ARM = ARGMAX (no operating point -- the "
            "Derm7pt-tuned SK/MISC downweights collapse HAM melanoma recall and were never "
            "adopted; see finalize_paper_metrics.py)"
        ),
    }


def _pick_cases(pr, fit: Dict[str, object]) -> List[Dict[str, object]]:
    """The two showcase cases, by the deterministic rule in the module docstring."""
    model: LogisticRegression = fit["model"]
    records = fit["records"]
    test_idx: List[int] = fit["test_idx"]

    xb_test = np.hstack(
        [fit["features"][test_idx], fit["meta_all"][test_idx]]
    )
    proba = model.predict_proba(xb_test)
    classes = [str(c) for c in model.classes_]
    argmax_pred = [classes[i] for i in proba.argmax(axis=1)]

    chosen: List[Dict[str, object]] = []
    for demo_label, raw_label, expected_group in PICKS:
        found: Optional[Dict[str, object]] = None
        for pos, global_i in enumerate(test_idx):
            rec = records[global_i]
            if str(rec["label"]) != raw_label or argmax_pred[pos] != expected_group:
                continue
            found = {
                "demo_label": demo_label,
                "record": rec,
                "global_index": int(global_i),
                "test_position": int(pos),
                "batch_argmax_prediction": argmax_pred[pos],
            }
            break
        if found is None:
            raise SystemExit(f"no E3-test image satisfies the pick rule for {demo_label!r}")
        chosen.append(found)
    return chosen


def _encode_queries(chosen: List[Dict[str, object]], fit: Dict[str, object]) -> Dict[str, object]:
    """Fresh DermLIP encode of the two raw JPGs, with the cached rows as a cross-check.

    Falls back to the cached rows (with an explicit note) if the encoder cannot load here.
    """
    paths = [Path(c["record"]["image_path"]) for c in chosen]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"raw HAM image missing on disk: {p}")

    cached = np.stack([fit["features"][c["global_index"]] for c in chosen])
    try:
        from medguard_v2.perception.encode import load_encoder

        started = time.perf_counter()
        encode, provenance = load_encoder("dermlip")
        fresh = np.asarray(encode(paths), dtype=np.float32)
        seconds = round(time.perf_counter() - started, 1)
        diffs = np.abs(fresh - cached).max(axis=1)
        return {
            "path_used": "fresh",
            "note": (
                "the two query images were encoded FRESH from their raw JPGs with the frozen "
                "DermLIP encoder (the raw-image entry path); the cached E3 feature rows agree "
                "to the reported max |diff| (float32 batched-CPU inference tolerance)"
            ),
            "encoder_provenance": provenance,
            "encode_seconds": seconds,
            "vectors": fresh,
            "max_abs_diff_vs_cached": [round(float(d), 8) for d in diffs],
        }
    except Exception as exc:  # encoder genuinely unavailable in this interpreter
        return {
            "path_used": "cached",
            "note": (
                "FALLBACK: the frozen DermLIP encoder could not load in this environment "
                f"({type(exc).__name__}: {exc}); the cached E3 feature rows were used instead. "
                "The contract/render half of the demo does not depend on which path ran."
            ),
            "encoder_provenance": None,
            "encode_seconds": None,
            "vectors": cached,
            "max_abs_diff_vs_cached": [0.0, 0.0],
        }


def main() -> None:
    started = time.perf_counter()
    pr = _load_script_module(
        "promoted_recipe_ham_bcn", REPO / "scripts/dev/promoted_recipe_ham_bcn.py"
    )

    print("=== fitting the adopted promoted-recipe HAM classifier (E3 train side) ===", flush=True)
    fit = _fit_adopted_ham_classifier(pr)
    print(f"  {fit['config']}", flush=True)

    chosen = _pick_cases(pr, fit)
    for c in chosen:
        print(f"  pick: {c['demo_label']} -> {c['record']['case_id']}", flush=True)

    print("=== encoding the two raw images ===", flush=True)
    enc = _encode_queries(chosen, fit)
    print(f"  path used: {enc['path_used']} (max |diff| vs cache: {enc['max_abs_diff_vs_cached']})",
          flush=True)

    model: LogisticRegression = fit["model"]
    classes = [str(c) for c in model.classes_]
    catalog = GuidelineCatalog.load()

    cases_out = []
    for i, c in enumerate(chosen):
        rec = c["record"]
        meta_vec = fit["meta_all"][c["global_index"]]
        x_query = np.concatenate([enc["vectors"][i], meta_vec]).reshape(1, -1)
        posterior = model.predict_proba(x_query)[0]
        diagnosis = classes[int(posterior.argmax())]  # ADOPTED ARM: plain argmax
        confidence = float(posterior.max())
        if diagnosis != c["batch_argmax_prediction"]:
            raise SystemExit(
                f"fresh-encode prediction {diagnosis} differs from the cached-batch prediction "
                f"{c['batch_argmax_prediction']} for {rec['case_id']} -- the demo would not be "
                "showing the recorded classifier's decision"
            )

        m = rec.get("metadata", {}) or {}
        contract = build_single_view_contract(
            case_id=str(rec["case_id"]),
            classifier_diagnosis=diagnosis,
            classifier_confidence=confidence,
            catalog=catalog,
            reason=CHECKLIST_UNAVAILABLE_REASON,
            receipt={
                "entry_point": "scripts/dev/demo_single_view_ham.py",
                "artifact_class": "DEMO -- not an evaluation",
                "classifier": fit["config"],
                "encoding_path": enc["path_used"],
                "encoder": enc["encoder_provenance"],
                "image_sha256": rec["image_sha256"],
                "metadata_used": {
                    "age": m.get("age"),
                    "sex": m.get("sex", "unknown"),
                    "localization": m.get("localization", "unknown"),
                },
            },
        )
        if not contract.safe_to_display:
            raise SystemExit(
                f"demo contract for {rec['case_id']} failed verification: "
                f"{contract.verification_failures()}"
            )

        render_default = render_markdown(contract)
        render_appendix = render_markdown(contract, include_appendix=True)
        print("\n" + "=" * 88, flush=True)
        print(f"DEMO CASE: {c['demo_label']} -- {rec['case_id']}", flush=True)
        print("=" * 88, flush=True)
        print(render_default, flush=True)

        cases_out.append({
            "demo_label": c["demo_label"],
            "case_id": rec["case_id"],
            "cluster_id": rec["cluster_id"],
            "true_label_demo_labeling_only": rec["label"],
            "predicted_group": diagnosis,
            "confidence": round(confidence, 4),
            "posterior": {cls: round(float(p), 4) for cls, p in zip(classes, posterior)},
            "max_abs_feature_diff_vs_cached": enc["max_abs_diff_vs_cached"][i],
            "contract": single_view_contract_to_dict(contract),
            "render_default": render_default,
            "render_with_appendix": render_appendix,
        })

    report = {
        "protocol": "single-view-ham-demo-v1",
        "artifact_class": (
            "DEMO -- illustrative end-to-end run of the single-view contract + tiered render on "
            "two real HAM10000 images. NOT an evaluation: no test-set metric is computed or "
            "claimed; the registered numbers live in ham_evaluation.json and "
            "promoted_recipe_ham_bcn.json. E3 test-side images were already read post-hoc under "
            "the user-authorized re-read; this demo re-reads two of them."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "classifier": {
            "config": fit["config"],
            "fitting_code_path": (
                "scripts/dev/promoted_recipe_ham_bcn.py functions (which reuse "
                "build_ham_evaluation.make_split), reproducing the ham_e3 five_group "
                "candidate_b_promoted_recipe fit; ARGMAX arm (adopted)"
            ),
            "c_selection": fit["c_selection"],
            "n_train_images": fit["n_train_images"],
            "n_test_images": fit["n_test_images"],
            "split_verification": (
                "E3 partition regenerated via build_ham_evaluation.make_split (seed 20260812) "
                "and asserted equal to the recorded train/test cluster_id lists; C selection "
                "asserted equal to the recorded candidate_b selection"
            ),
        },
        "case_selection_rule": (
            "for each of (melanoma, nevus): the FIRST E3-test image in manifest record order "
            "whose true label matches and whose argmax prediction agrees -- so the demo "
            "exercises the guideline-covered and explicit-absence paths as designed. True "
            "labels appear here for demo labeling only; no accuracy claim is made or implied."
        ),
        "encoding": {k: v for k, v in enc.items() if k != "vectors"},
        "checklist_unavailable_reason": CHECKLIST_UNAVAILABLE_REASON,
        "cases": cases_out,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    print(f"done in {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
