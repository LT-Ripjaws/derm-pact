"""Regression tests for medguard_v2/perception/attribution.py.

This module ("GradCAM on frozen DermLIP") had never been imported by
anything else in the codebase before this task -- confirmed by grep across
medguard_v2/ and scripts/ -- so nothing had ever exercised it end to end.
Three real defects were found as a result, none of them caught by the module
simply importing cleanly:

(a) Binary-class sign inversion. sklearn's binary LogisticRegression stores a
    single coef_ row (shape (1, n_features)) that points toward classes_[1].
    Both the sklearn-only feature-importance path and the GradCAM path used
    coef_[0] unconditionally, so every attribution sign was inverted whenever
    the explained class was classes_[0]. Both call sites now go through the
    shared _signed_weights / _signed_intercept helpers tested below.

(b) spatial_size inference. int(np.sqrt(n_tokens - 1)) alone assumes a
    leading CLS token; when the token count has no CLS token this either
    crashes on a non-square reshape or silently misgrids. _reshape_to_patch_grid
    (tested below with synthetic 197-token (14x14+CLS) and 196-token (14x14)
    arrays, as specified) checks both layouts explicitly and raises a clear
    error when neither fits.

(c) A CLS-pooling gradient dead end, found empirically while wiring
    scripts/dev/validate_attribution_faithfulness.py -- not one of the two
    originally-reviewed bugs, but a blocking one: DermLIP's visual tower
    reads only the CLS token after its LAST transformer block (pool_type=
    "tok"), so hooking that block's OUTPUT gives an exactly-zero gradient at
    every one of the 196 patch positions, for every image, every head, every
    time (verified: channel weights identically 0.0 across 5 real cases x 3
    real heads before the fix). That is a dead gradient path, not a
    faithfulness problem to go measure. _gradcam_target_layer (hooking the
    SECOND-to-last block instead, where the last block's own self-attention
    still lets the CLS token read every patch position) is tested in
    isolation below, against a lightweight fake model so it runs without the
    real DermLIP checkpoint.

Also confirmed empirically (not asserted by a test here, since it needs the
real checkpoint): gradcam_dermlip was unconditionally unpacking
medguard_v2.perception.encode.load_encoder's 2-tuple return into 3 names,
which raises ValueError on the very first call -- direct evidence the
function had literally never run. Fixed by adding _load_dermlip_model,
which reuses encode.py's checkpoint-resolution utilities (ENCODERS,
encoder_snapshot) instead of duplicating them.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression

from medguard_v2.perception.attribution import (
    _gradcam_target_layer,
    _reshape_to_patch_grid,
    _signed_intercept,
    _signed_weights,
    feature_importance_logreg,
)


# ---------------------------------------------------------------------------
# bug (a): binary-class sign inversion
# ---------------------------------------------------------------------------

class SignedWeightsBinaryHeadTest(unittest.TestCase):
    """sklearn's binary LogisticRegression.coef_ has shape (1, n_features) and
    points toward classes_[1]. Using it unconditionally to explain
    classes_[0] inverts every sign. Fit a tiny, well-separated binary head
    -- classes_ sorts alphabetically, so classes_[0]="neg", classes_[1]="pos"
    -- and check both explanations, i.e. "both ways" as asked."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        x_pos = rng.normal(loc=[4.0, 4.0], scale=0.2, size=(25, 2))
        x_neg = rng.normal(loc=[-4.0, -4.0], scale=0.2, size=(25, 2))
        cls.x = np.vstack([x_pos, x_neg]).astype(np.float64)
        cls.y = ["pos"] * 25 + ["neg"] * 25
        cls.head = LogisticRegression().fit(cls.x, cls.y)
        assert list(cls.head.classes_) == ["neg", "pos"], cls.head.classes_

    def test_signed_weights_negate_between_the_two_classes(self):
        w_pos = _signed_weights(self.head, class_idx=1)  # classes_[1] = "pos"
        w_neg = _signed_weights(self.head, class_idx=0)  # classes_[0] = "neg"
        np.testing.assert_allclose(w_pos, self.head.coef_[0])
        np.testing.assert_allclose(w_neg, -self.head.coef_[0])

    def test_signed_intercept_negates_between_the_two_classes(self):
        b_pos = _signed_intercept(self.head, class_idx=1)
        b_neg = _signed_intercept(self.head, class_idx=0)
        self.assertAlmostEqual(b_pos, float(self.head.intercept_[0]))
        self.assertAlmostEqual(b_neg, -float(self.head.intercept_[0]))

    def test_feature_importance_logreg_both_ways_on_a_positive_example(self):
        # A clearly "pos"-like point: explaining "pos" should show positive
        # contributions (evidence FOR the predicted class); explaining "neg" on
        # the SAME point should show the exact mirror (negative). Under the
        # original bug (coef_[0] used unconditionally) the two explanations
        # would be IDENTICAL instead of mirrored, since "pos" happens to be
        # classes_[1] and the bug only manifests explaining classes_[0].
        example = np.array([4.0, 4.0])

        result_pos = feature_importance_logreg(self.head, example, "crit", "pos")
        result_neg = feature_importance_logreg(self.head, example, "crit", "neg")

        contrib_pos = {f["feature_idx"]: f["contribution"] for f in result_pos["top_features"]}
        contrib_neg = {f["feature_idx"]: f["contribution"] for f in result_neg["top_features"]}
        self.assertEqual(set(contrib_pos), set(contrib_neg))
        for idx in contrib_pos:
            self.assertGreater(contrib_pos[idx], 0.0, "evidence for the predicted class should be positive")
            self.assertAlmostEqual(contrib_pos[idx], -contrib_neg[idx], places=6)

        weight_pos = {f["feature_idx"]: f["weight"] for f in result_pos["top_features"]}
        weight_neg = {f["feature_idx"]: f["weight"] for f in result_neg["top_features"]}
        for idx in weight_pos:
            self.assertAlmostEqual(weight_pos[idx], -weight_neg[idx], places=6)

        self.assertAlmostEqual(result_pos["intercept"], -result_neg["intercept"], places=6)

    def test_unknown_predicted_value_returns_an_error_not_a_crash(self):
        result = feature_importance_logreg(self.head, np.array([0.0, 0.0]), "crit", "not-a-class")
        self.assertIn("error", result)


class SignedWeightsMulticlassHeadTest(unittest.TestCase):
    """The multiclass (softmax) path must be untouched by the binary-sign fix:
    coef_ already has one row per class there, so no sign flip should ever
    apply, regardless of which class index is explained."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        blobs = {
            "a": rng.normal(loc=[5, 0], scale=0.2, size=(20, 2)),
            "b": rng.normal(loc=[-5, 5], scale=0.2, size=(20, 2)),
            "c": rng.normal(loc=[-5, -5], scale=0.2, size=(20, 2)),
        }
        cls.x = np.vstack(list(blobs.values())).astype(np.float64)
        cls.y = sum(([label] * len(pts) for label, pts in blobs.items()), [])
        cls.head = LogisticRegression().fit(cls.x, cls.y)
        assert cls.head.coef_.shape[0] == 3, cls.head.coef_.shape

    def test_signed_weights_matches_coef_row_directly_for_every_class(self):
        for idx in range(len(self.head.classes_)):
            np.testing.assert_array_equal(_signed_weights(self.head, idx), self.head.coef_[idx])

    def test_signed_intercept_matches_intercept_row_directly_for_every_class(self):
        for idx in range(len(self.head.classes_)):
            self.assertEqual(_signed_intercept(self.head, idx), float(self.head.intercept_[idx]))


# ---------------------------------------------------------------------------
# bug (b): spatial_size inference
# ---------------------------------------------------------------------------

class ReshapeToPatchGridTest(unittest.TestCase):
    """Synthetic (1, n_tokens, d) arrays -- no torch, no model, no image
    required, per the task's explicit ask for a 197-token (14x14+CLS) and a
    196-token (14x14) unit test. Plain numpy arrays are used deliberately:
    _reshape_to_patch_grid's docstring claims it works for "torch tensors and
    numpy arrays, whichever the caller uses", so testing it with numpy alone
    (gradcam_dermlip itself calls it with torch tensors) also locks in that
    it has no hidden torch-only dependency."""

    def test_197_tokens_strips_a_leading_cls_token_and_grids_14x14(self):
        # Distinct values per token so a wrong slice/reshape is detectable, not
        # just shape-compatible by accident.
        act = np.arange(197 * 4, dtype=np.float64).reshape(1, 197, 4)
        grad = -act.copy()

        out_act, out_grad, spatial_size = _reshape_to_patch_grid(act, grad)

        self.assertEqual(spatial_size, 14)
        self.assertEqual(out_act.shape, (1, 14, 14, 4))
        self.assertEqual(out_grad.shape, (1, 14, 14, 4))
        # The CLS token (original index 0) must have been dropped: the first
        # patch in the grid is original token 1, not token 0.
        np.testing.assert_array_equal(out_act[0, 0, 0], act[0, 1])
        np.testing.assert_array_equal(out_grad[0, 0, 0], grad[0, 1])
        # And the last grid cell is the last token (index 196).
        np.testing.assert_array_equal(out_act[0, 13, 13], act[0, 196])

    def test_196_tokens_grids_14x14_without_dropping_anything(self):
        act = np.arange(196 * 4, dtype=np.float64).reshape(1, 196, 4)
        grad = -act.copy()

        out_act, out_grad, spatial_size = _reshape_to_patch_grid(act, grad)

        self.assertEqual(spatial_size, 14)
        self.assertEqual(out_act.shape, (1, 14, 14, 4))
        # No CLS token to drop: the first grid cell is original token 0.
        np.testing.assert_array_equal(out_act[0, 0, 0], act[0, 0])
        np.testing.assert_array_equal(out_grad[0, 0, 0], grad[0, 0])

    def test_token_count_fitting_neither_layout_raises_a_clear_error(self):
        # 200 fits neither 14*14=196 nor 14*14+1=197 (nor any other k*k or
        # k*k+1): with_cls = isqrt(199) = 14, 14*14+1 = 197 != 200;
        # without_cls = isqrt(200) = 14, 14*14 = 196 != 200. This is exactly
        # the "silently misgrids" failure mode the fix must not have: the old
        # unconditional int(sqrt(n_tokens-1)) code would have reshaped 200
        # tokens into a nonsense 14x14 grid, silently dropping 5 tokens
        # instead of raising.
        act = np.zeros((1, 200, 4))
        grad = np.zeros((1, 200, 4))
        with self.assertRaises(RuntimeError) as ctx:
            _reshape_to_patch_grid(act, grad)
        self.assertIn("200", str(ctx.exception))

    def test_a_larger_perfect_square_without_cls_still_resolves_correctly(self):
        # 225 = 15*15 exactly (and 224 = sqrt(...)~14.97 is not a perfect
        # square either), so this must resolve via the without-CLS branch, at
        # a grid size the with-CLS-token test above does not cover.
        act = np.arange(225 * 2, dtype=np.float64).reshape(1, 225, 2)
        grad = -act.copy()
        out_act, out_grad, spatial_size = _reshape_to_patch_grid(act, grad)
        self.assertEqual(spatial_size, 15)
        np.testing.assert_array_equal(out_act[0, 0, 0], act[0, 0])


# ---------------------------------------------------------------------------
# bug (c): CLS-pooling gradient dead end (found while building the
# faithfulness validation harness, not one of the two originally reviewed)
# ---------------------------------------------------------------------------

def _fake_resblocks_model(n_blocks: int):
    """Minimal stand-in for an open_clip-style model: just enough attribute
    structure (model.visual.transformer.resblocks) for _gradcam_target_layer
    to walk, with sentinel objects so `is` identity checks are meaningful."""
    blocks = [object() for _ in range(n_blocks)]
    model = SimpleNamespace(visual=SimpleNamespace(transformer=SimpleNamespace(resblocks=blocks)))
    return model, blocks


def _fake_trunk_model(n_blocks: int):
    """Minimal stand-in for a timm-style model: model.visual.trunk.blocks."""
    blocks = [object() for _ in range(n_blocks)]
    model = SimpleNamespace(visual=SimpleNamespace(trunk=SimpleNamespace(blocks=blocks)))
    return model, blocks


class GradcamTargetLayerTest(unittest.TestCase):
    def test_picks_the_second_to_last_block_not_the_last(self):
        model, blocks = _fake_resblocks_model(12)
        target = _gradcam_target_layer(model)
        self.assertIs(target, blocks[-2])
        self.assertIsNot(target, blocks[-1])

    def test_falls_back_to_the_only_block_when_just_one_exists(self):
        model, blocks = _fake_resblocks_model(1)
        target = _gradcam_target_layer(model)
        self.assertIs(target, blocks[0])

    def test_trunk_style_model_also_uses_second_to_last(self):
        model, blocks = _fake_trunk_model(12)
        target = _gradcam_target_layer(model)
        self.assertIs(target, blocks[-2])

    def test_a_model_with_neither_layout_raises_a_clear_error(self):
        model = SimpleNamespace(visual=SimpleNamespace())
        with self.assertRaises(RuntimeError):
            _gradcam_target_layer(model)


if __name__ == "__main__":
    unittest.main()
