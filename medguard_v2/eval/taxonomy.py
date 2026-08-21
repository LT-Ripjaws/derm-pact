"""Keep the two diagnostic taxonomies in this repo from being mixed.

Two incompatible label sets are in play:

* **HAM10000, 7 classes** -- ``melanoma, nevus, bcc, akiec, bkl, df, vasc``.
  Ported verbatim from v1; it is what :data:`medguard_v2.eval.metrics.DERM_CLASS_ORDER`
  holds and what ``classification_summary`` falls back to when ``labels`` is omitted.
* **Derm7pt, 5 groups** -- ``BCC, NEV, MEL, SK, MISC``, defined by
  :data:`medguard_v2.data.derm7pt.DERM7PT_CLASS_ORDER`.

``classification_summary(y_true, y_pred, labels=None)`` takes the taxonomy as an
optional argument and defaults to the HAM10000 order.  Feeding it Derm7pt data
therefore fails in one of three ways, and only the first is loud:

1. *Loud but cryptic.* A class-diverse Derm7pt cohort raises
   ``ValueError: unknown diagnostic label truth='MEL', pred='MEL'`` -- which
   names a label, never the taxonomy, and reads like bad data rather than a
   mis-declared label set.
2. *Silent, via the one overlapping token.* ``bcc`` / ``BCC`` is a member of
   **both** taxonomies.  A cohort containing only BCC cases scores
   ``balanced_accuracy = 0.143`` under the HAM10000 labels and ``0.200`` under
   the Derm7pt ones, with no error either way.
3. *Silent, via arity.* Macro-averaged metrics divide by ``len(labels)``, so a
   label sequence with extra or missing classes silently rescales
   ``balanced_accuracy`` / ``macro_f1`` (verified: 0.800 -> 0.667 by appending a
   single never-observed class).  This is why :func:`resolve_labels` returns
   **the** canonical order rather than leaving each caller to assemble one.

``metrics.py`` is a byte-identical port of v1 and must not be touched, so the
guard lives beside it instead of inside it.  Intended use::

    labels = assert_labels_match(y_true, "derm7pt")
    summary = classification_summary(y_true, y_pred, labels)

Management decisions (``clinical follow up`` / ``excision`` /
``no further examination``) are an ordinal *action* ladder, not a diagnosis
taxonomy, and are deliberately not registered here: a caller checking them
against either registered taxonomy should get an error, which is what happens.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from ..data.derm7pt import DERM7PT_CLASS_ORDER
from .metrics import DERM_CLASS_ORDER

__all__ = [
    "TaxonomyError",
    "TAXONOMIES",
    "HAM10000_CLASS_ORDER",
    "DERM7PT_CLASS_ORDER",
    "resolve_labels",
    "assert_labels_match",
]


#: HAM10000 7-class order, taken from ``metrics.DERM_CLASS_ORDER`` so that
#: ``resolve_labels("ham10000")`` is identical to ``classification_summary``'s own
#: default.  A guard that silently reordered the confusion-matrix columns relative
#: to v1's artefacts would be worse than no guard at all.
HAM10000_CLASS_ORDER = DERM_CLASS_ORDER

#: The registered taxonomies, by canonical name.
TAXONOMIES: Mapping[str, tuple[str, ...]] = {
    "ham10000": HAM10000_CLASS_ORDER,
    "derm7pt": DERM7PT_CLASS_ORDER,
}


class TaxonomyError(ValueError):
    """Raised when a label set and the named taxonomy disagree.

    Subclasses :class:`ValueError` so callers already guarding
    ``classification_summary``'s own ``ValueError`` keep working.
    """


def _known_names() -> str:
    return ", ".join(repr(name) for name in sorted(TAXONOMIES))


def _fold(value: str) -> str:
    """Case/whitespace folding, matching how ``metrics`` compares labels."""
    return str(value).strip().lower()


def resolve_labels(taxonomy: str) -> tuple[str, ...]:
    """Return the canonical label order for ``taxonomy``.

    Raises :class:`TaxonomyError` for an unregistered name rather than falling
    back to a default -- a default is exactly how the wrong taxonomy gets used.
    """
    key = _fold(taxonomy)
    try:
        return TAXONOMIES[key]
    except KeyError:
        raise TaxonomyError(
            f"unknown taxonomy {taxonomy!r}; registered taxonomies: {_known_names()}"
        ) from None


def _taxonomies_accepting(values: Iterable[str]) -> list[str]:
    """Names of every registered taxonomy that contains all of ``values``."""
    folded = {_fold(value) for value in values}
    return sorted(
        name
        for name, labels in TAXONOMIES.items()
        if folded <= {_fold(label) for label in labels}
    )


def assert_labels_match(observed_labels: Iterable[str], taxonomy: str) -> tuple[str, ...]:
    """Assert every observed class value belongs to ``taxonomy``; return its order.

    ``observed_labels`` may be raw ``y_true`` / ``y_pred`` values or an explicit
    ``labels`` sequence -- both are "class values that claim to be this
    taxonomy".  A strict subset is fine: a split need not contain every class.

    Returns the canonical order so the checked call reads as one expression::

        classification_summary(y_true, y_pred, assert_labels_match(y_true, "derm7pt"))

    Raises :class:`TaxonomyError` when any value is outside the named taxonomy
    (naming the taxonomy that *would* have accepted them, when there is one), and
    :class:`TypeError` when handed a bare string, which would otherwise be
    iterated one character at a time and produce a nonsense diagnosis.
    """
    if isinstance(observed_labels, (str, bytes)):
        raise TypeError(
            "observed_labels must be a sequence of class values, not a single "
            f"string ({observed_labels!r}); wrap it in a list if that was intended"
        )
    expected = resolve_labels(taxonomy)
    observed = [str(value) for value in observed_labels]
    if not observed:
        raise TaxonomyError(
            f"no observed class values to check against taxonomy {taxonomy!r}; "
            "an empty cohort cannot be validated"
        )

    allowed = {_fold(label) for label in expected}
    unknown = sorted({value for value in observed if _fold(value) not in allowed})
    if not unknown:
        return expected

    alternatives = [name for name in _taxonomies_accepting(observed)
                    if name != _fold(taxonomy)]
    if alternatives:
        hint = (
            "those values are "
            + " / ".join(repr(name) for name in alternatives)
            + f" labels -- pass taxonomy={alternatives[0]!r} instead"
        )
    else:
        hint = f"no registered taxonomy contains every observed value ({_known_names()})"
    raise TaxonomyError(
        f"observed class values do not belong to taxonomy {_fold(taxonomy)!r}: "
        f"{unknown} not in {list(expected)}; {hint}. Mixing taxonomies changes the "
        "per-class table and the macro averages without necessarily raising."
    )
