"""Visual attribution maps for criterion heads via GradCAM on frozen DermLIP.

Produces per-criterion heatmaps showing which image regions drove each
criterion head's prediction.  The maps are suggestive, not causal: they
show what the frozen encoder attended to, filtered by what the linear
head learned matters for each criterion.

Two methods:
  1. **GradCAM**: gradient of the head's predicted-class logit w.r.t. the
     second-to-last transformer block's spatial activations → weighted
     average → ReLU.  Second-to-last, not last: DermLIP's visual tower reads
     only the CLS token after its final block, so the last block's own
     patch-token outputs have zero gradient to the pooled embedding by
     construction, regardless of image content -- one block earlier is where
     spatial gradient actually exists (see the comment in
     :func:`gradcam_dermlip`).
  2. **Feature attribution**: SHAP-style permutation importance over the
     DermLIP feature vector (512-d for the ``dermlip`` encoder), visualised
     as a bar chart (not spatial).

GradCAM requires the full DermLIP model and torch; feature attribution
works on pre-extracted features with sklearn only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AttributionMap:
    """A spatial attribution map for one criterion on one image."""
    criterion: str
    predicted_value: str
    heatmap: np.ndarray
    image_size: Tuple[int, int]
    method: str
    #: The un-resized, native ``(spatial_size, spatial_size)`` patch-grid
    #: saliency -- ``heatmap`` before it is upsampled to ``image_size`` by PIL.
    #: ``None`` unless the caller asked for it via ``gradcam_dermlip``'s
    #: ``return_patch_grid=True``.  Most callers only want the overlay-ready
    #: ``heatmap``; faithfulness evaluation needs to mask patches in the same
    #: grid the encoder actually tokenises, and going through ``heatmap``
    #: would mean downsampling an upsampled map -- a lossy, avoidable round
    #: trip through PIL's bilinear resize.
    patch_heatmap: Optional[np.ndarray] = None


# --------------------------------------------------------------------------
# shared helpers (used by both the sklearn-only path and the GradCAM path)
# --------------------------------------------------------------------------

def _signed_weights(head, class_idx: int) -> np.ndarray:
    """Per-feature logit weights for explaining ``head.classes_[class_idx]``.

    sklearn's binary ``LogisticRegression`` stores a single coefficient row in
    ``coef_`` (shape ``(1, n_features)``): ``decision_function(x) = x @
    coef_[0] + intercept_[0]`` is the logit for ``classes_[1]``, and because
    the two class probabilities are ``sigmoid(s)`` and ``sigmoid(-s)`` for the
    same score ``s``, the logit for ``classes_[0]`` is its exact negation.
    Using ``coef_[0]`` unconditionally -- regardless of which class is being
    explained -- silently inverts every attribution sign whenever the
    explained class is ``classes_[0]``.  Multiclass (softmax) heads already
    store one coefficient row per class in ``coef_``, so no sign flip applies.
    """
    if head.coef_.shape[0] == 1:
        return head.coef_[0] if class_idx == 1 else -head.coef_[0]
    return head.coef_[class_idx]


def _signed_intercept(head, class_idx: int) -> float:
    """Intercept counterpart to :func:`_signed_weights`; see its docstring."""
    if head.intercept_.shape[0] == 1:
        return float(head.intercept_[0] if class_idx == 1 else -head.intercept_[0])
    return float(head.intercept_[class_idx])


def _reshape_to_patch_grid(act, grad) -> Tuple[Any, Any, int]:
    """Reshape ``(1, n_tokens, d)`` activation/gradient tensors to a square
    spatial grid ``(1, size, size, d)``, dropping a leading CLS token if present.

    ViT patch-token sequences are square in one of two layouts: exactly
    ``size * size`` patch tokens, or ``size * size + 1`` tokens when a
    leading CLS token is prepended.  ``int(np.sqrt(n_tokens - 1))`` alone
    assumes the CLS-token layout unconditionally: when the encoder has no
    CLS token this either crashes on a non-square reshape or -- worse --
    silently misgrids if ``n_tokens - 1`` happens to also be a perfect
    square.  Both layouts are checked explicitly here, and anything that
    fits neither raises instead of guessing.

    Takes and returns plain tensor-like objects (works for both torch
    tensors and numpy arrays, whichever the caller uses), so it has no
    dependency on torch, a model, or an image and can be unit-tested with
    synthetic tensors alone.
    """
    n_tokens = act.shape[1]
    with_cls = int(np.sqrt(n_tokens - 1)) if n_tokens > 1 else -1
    without_cls = int(np.sqrt(n_tokens))
    if n_tokens > 1 and with_cls * with_cls + 1 == n_tokens:
        spatial_size = with_cls
        act = act[:, 1:, :]
        grad = grad[:, 1:, :]
    elif without_cls * without_cls == n_tokens:
        spatial_size = without_cls
    else:
        raise RuntimeError(
            f"cannot infer a square patch grid (with or without a leading CLS "
            f"token) from {n_tokens} tokens"
        )
    act = act.reshape(1, spatial_size, spatial_size, -1)
    grad = grad.reshape(1, spatial_size, spatial_size, -1)
    return act, grad, spatial_size


def _load_dermlip_model(encoder_key: str):
    """Load the raw model + eval preprocessing transform, for GradCAM.

    :func:`medguard_v2.perception.encode.load_encoder` deliberately returns
    an opaque ``(encode_fn, provenance)`` pair for the batch
    feature-extraction pipeline: ``encode_fn`` wraps every forward pass in
    ``torch.inference_mode()``, which permanently disables autograd on its
    outputs, and it never exposes the ``model``/``preprocess`` objects it
    closes over.  GradCAM needs both -- gradients enabled, and hooks
    registered directly on one of the model's transformer blocks -- so it
    cannot go through that closure.

    This reuses the same checkpoint-resolution utilities ``encode.py`` uses
    (``ENCODERS`` for the model id, ``encoder_snapshot`` for offline-safe
    cache resolution and the same clear failure mode when a checkpoint is
    not cached) rather than re-deriving them, and only re-does the single
    step ``encode.py`` does not expose: constructing the live model object.
    """
    import open_clip

    from medguard_v2.perception.encode import ENCODERS, PerceptionEncodeError, encoder_snapshot

    if encoder_key not in ENCODERS:
        raise PerceptionEncodeError(
            f"unknown encoder {encoder_key!r}; choose from {sorted(ENCODERS)}"
        )
    spec = ENCODERS[encoder_key]
    if spec.loader != "open_clip":
        raise PerceptionEncodeError(f"unsupported loader {spec.loader!r} for {encoder_key!r}")

    # Side effect only: verifies the checkpoint is cached offline and raises the
    # same PerceptionEncodeError encode.load_encoder would, instead of an opaque
    # failure from deep inside huggingface_hub/open_clip.
    encoder_snapshot(encoder_key)

    model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{spec.model_id}")
    model.eval()
    return model, preprocess


def _gradcam_target_layer(model):
    """The transformer block GradCAM should hook: SECOND-to-last, not last.

    DermLIP (open_clip ``VisionTransformer``, ``pool_type="tok"``) reads only
    the CLS token (position 0) after the final block: its ``_pool`` selects
    ``x[:, 0]``, and everything downstream -- including ``encode_image``'s
    returned embedding -- is a function of that one position. The last
    block's OUTPUT at the other 196 patch positions is therefore never read
    by anything downstream, so ``d(logit)/d(last_block_output)`` at a patch
    position is exactly zero for every image, every head, every time --
    confirmed empirically (channel weights identically ``0.0``, not just
    small, across 5 real cases x 3 real heads before this fix). That is not
    a faithfulness problem to measure; it is a dead gradient path, because
    GradCAM's premise -- every spatial position of the target layer
    influences the target logit -- does not hold at the very last layer of a
    CLS-pooled ViT. One block earlier it does hold: the last block's own
    self-attention lets the CLS token read from every patch position in ITS
    input (= the second-to-last block's output), so that is where spatial
    gradient actually exists. The same architectural argument applies to a
    timm-style ``trunk`` (also CLS-pooled by default); untested against a
    real timm checkpoint since only the open_clip ``resblocks`` branch is
    exercised by DermLIP, but the reasoning is identical.
    """
    visual = model.visual
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        blocks = visual.transformer.resblocks
    elif hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        blocks = visual.trunk.blocks
    else:
        raise RuntimeError("Cannot locate a transformer block in DermLIP")
    return blocks[-2] if len(blocks) > 1 else blocks[-1]


def feature_importance_logreg(
    head,
    feature_vector: np.ndarray,
    criterion_name: str,
    predicted_value: str,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Coefficient-based feature importance for a logistic regression head.

    Returns the top-k most important feature indices and their weights
    for the predicted class.  Fast, no model loading required.
    """
    classes = list(head.classes_)
    if predicted_value not in classes:
        return {"criterion": criterion_name, "error": f"{predicted_value} not in classes"}

    class_idx = classes.index(predicted_value)
    weights = _signed_weights(head, class_idx)

    contributions = weights * feature_vector.ravel()
    top_indices = np.argsort(np.abs(contributions))[::-1][:top_k]

    return {
        "criterion": criterion_name,
        "predicted_value": predicted_value,
        "method": "logreg_coefficients",
        "top_features": [
            {
                "feature_idx": int(idx),
                "weight": round(float(weights[idx]), 6),
                "activation": round(float(feature_vector.ravel()[idx]), 6),
                "contribution": round(float(contributions[idx]), 6),
            }
            for idx in top_indices
        ],
        "intercept": round(_signed_intercept(head, class_idx), 6),
    }


def gradcam_dermlip(
    image_path: str,
    head,
    criterion_name: str,
    encoder_key: str = "dermlip",
    *,
    return_patch_grid: bool = False,
) -> AttributionMap:
    """Compute GradCAM heatmap for a criterion head on a dermoscopy image.

    Requires torch and the DermLIP model.  The gradient flows from the
    criterion head's logit through the frozen encoder's second-to-last
    transformer block to produce spatial attribution (see
    :func:`_gradcam_target_layer` for why not the last one).

    ``return_patch_grid=True`` additionally populates
    ``AttributionMap.patch_heatmap`` with the native, un-resized patch-grid
    saliency (see that field's docstring); ``False`` (the default) preserves
    the original return shape exactly.
    """
    import torch
    from PIL import Image as PILImage

    model, preprocess = _load_dermlip_model(encoder_key)

    img = PILImage.open(image_path).convert("RGB")
    orig_size = img.size
    tensor = preprocess(img).unsqueeze(0)

    activations = {}
    gradients = {}

    def _save_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    def _save_gradient(name):
        def hook(module, input, grad_output):
            gradients[name] = grad_output[0]
        return hook

    target_layer = _gradcam_target_layer(model)

    fwd_handle = target_layer.register_forward_hook(_save_activation("target"))
    bwd_handle = target_layer.register_full_backward_hook(_save_gradient("target"))

    try:
        with torch.enable_grad():
            tensor.requires_grad_(True)
            features = model.encode_image(tensor)
            # Every head in this codebase (diagnosis and all 7 criteria) is fit
            # exclusively on L2-normalised DermLIP features -- encode.py enforces
            # this on every cached feature it writes. Raw model.encode_image output
            # is NOT normalised (its norm is ~5-15, not 1), so predicting on it
            # directly feeds the head a vector far outside the scale it was fit on.
            # Confirmed empirically to flip predictions on borderline cases (e.g.
            # a properly-normalised proba of 0.55/0.45 became a saturated 1.0/0.0
            # on the raw vector for one Derm7pt case). Normalise identically to
            # encode.py before doing anything the head sees, so GradCAM explains
            # the same prediction the deployed pipeline actually makes -- and so
            # gradients flow through the normalisation too, which is correct: the
            # normalisation is part of what the head's logit is actually a
            # function of.
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            features_np = features.detach().cpu().numpy()

            pred_class = head.predict(features_np)[0]
            class_idx = list(head.classes_).index(pred_class)
            weights_vec = torch.tensor(_signed_weights(head, class_idx), dtype=torch.float32)

            logit = (features * weights_vec.unsqueeze(0)).sum()
            logit.backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    act = activations["target"]
    grad = gradients["target"]

    if act.dim() == 3:
        act, grad, _spatial_size = _reshape_to_patch_grid(act, grad)

    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = (weights * act).sum(dim=-1).squeeze()
    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    patch_heatmap = cam.copy() if return_patch_grid else None

    from PIL import Image as PILImage
    cam_resized = np.array(
        PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
            orig_size, PILImage.BILINEAR
        )
    ) / 255.0

    return AttributionMap(
        criterion=criterion_name,
        predicted_value=pred_class,
        heatmap=cam_resized,
        image_size=orig_size,
        method="gradcam_dermlip",
        patch_heatmap=patch_heatmap,
    )


def save_attribution_overlay(
    image_path: str,
    attr_map: AttributionMap,
    output_path: str,
    colormap: str = "jet",
    alpha: float = 0.4,
) -> str:
    """Save a GradCAM heatmap overlaid on the original image."""
    import matplotlib
    from PIL import Image as PILImage

    img = PILImage.open(image_path).convert("RGB")
    img_arr = np.array(img).astype(np.float32) / 255.0

    cmap = matplotlib.colormaps[colormap]
    heatmap_colored = cmap(attr_map.heatmap)[:, :, :3]

    overlay = (1 - alpha) * img_arr + alpha * heatmap_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    PILImage.fromarray(overlay).save(output_path)
    return output_path
