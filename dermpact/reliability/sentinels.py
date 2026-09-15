"""Auditable image sanitization and separate IQA/OOD recovery signals."""
from dataclasses import dataclass
from io import BytesIO
import hashlib
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import laplace


def sanitize_image(payload: bytes, max_bytes=40_000_000, max_pixels=25_000_000):
    if not payload or len(payload) > max_bytes:
        raise ValueError('image is empty or exceeds byte limit')
    try:
        with Image.open(BytesIO(payload)) as source:
            if source.width * source.height > max_pixels or min(source.size) < 1:
                raise ValueError('image exceeds pixel limit')
            oriented = ImageOps.exif_transpose(source).convert('RGB')
            # Reconstruct pixels rather than copying a metadata-bearing image.
            clean = Image.fromarray(np.array(oriented, copy=True))
    except (OSError, SyntaxError) as exc:
        raise ValueError('undecodable image') from exc
    sink = BytesIO()
    clean.save(sink, format='PNG')
    encoded = sink.getvalue()
    return clean, encoded, {'status': 'stripped', 'input_sha256': hashlib.sha256(payload).hexdigest(),
                            'sanitized_sha256': hashlib.sha256(encoded).hexdigest(),
                            'orientation': 'EXIF orientation applied before stripping'}


def quality_features(image):
    arr = np.asarray(image.convert('RGB').resize((224, 224)), dtype=float) / 255.
    gray = arr @ np.array([.299, .587, .114])
    return np.array([np.log1p(1 / (float(np.var(laplace(gray))) + 1e-8)),
                     np.mean(gray < .03), np.mean(gray > .97),
                     -float(gray.std())], dtype=float)


@dataclass
class QuantileQuality:
    reference: np.ndarray

    def __post_init__(self):
        self.reference = np.asarray(self.reference, dtype=float)
        if self.reference.ndim != 2 or self.reference.shape[1] != 4 or not len(self.reference) or not np.isfinite(self.reference).all():
            raise ValueError('quality reference must be finite [N,4]')

    def score(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
            raise ValueError('quality query must be finite [N,4]')
        ranks = np.stack([(self.reference[:, j][None, :] < x[:, j, None]).mean(1) for j in range(4)], axis=1)
        return ranks.max(1)


def normalize_embeddings(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('embeddings must be nonempty finite matrix')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError('zero embedding cannot be scored')
    return x / norm


def cosine_ood(reference, query):
    bank, x = normalize_embeddings(reference), normalize_embeddings(query)
    if bank.shape[1] != x.shape[1]:
        raise ValueError('embedding dimension mismatch')
    return 1 - np.max(x @ bank.T, axis=1)


def diagonal_distance(reference, query):
    bank, x = normalize_embeddings(reference), normalize_embeddings(query)
    if bank.shape[1] != x.shape[1]:
        raise ValueError('embedding dimension mismatch')
    return np.mean((x - bank.mean(0)) ** 2 / np.maximum(bank.var(0), 1e-5), axis=1)


def recovery(iqa_failed=False, ood_flagged=False, deferred=False, unstable=False):
    warnings = []
    for flag, code, reason, action in [
        (iqa_failed, 'iqa_fail', 'Image falls outside calibrated quality reference', 'reacquire_image'),
        (ood_flagged, 'ood_flag', 'Image is unfamiliar to frozen reference bank; advisory only', 'display_unfamiliarity_warning'),
        (deferred, 'selective_deferral', 'Predictive uncertainty exceeds operating threshold', 'clinician_review'),
        (unstable, 'view_unstable', 'Predictions vary across assessed views', 'reacquire_or_review'),
    ]:
        if flag:
            warnings.append({'code': code, 'reason': reason, 'recovery_action': action,
                             'remaining_usable_output': 'input provenance and supplied facts' if iqa_failed else 'model evidence with review warning'})
    return {'disposition': 'reacquire' if iqa_failed else ('refer' if deferred or unstable else 'display'), 'warnings': warnings}
