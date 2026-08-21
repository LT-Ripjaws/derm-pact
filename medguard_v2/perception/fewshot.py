"""Extensions to the G1 criterion assessor: visual exemplars, model profiles, parallel calls.

G1 concluded that "a VLM cannot read the seven dermoscopic criteria".  That conclusion rests on
exactly one configuration -- ``qwen/qwen3-vl-32b-instruct``, zero-shot, text definitions only,
one dermoscopic image, one pass.  This module supplies the three knobs G1c turns, without editing
anything G1 depends on:

**F1, the model.**  Not every gateway model accepts the same decoding parameters.  Anthropic
models reject ``seed``; OpenAI's GPT-5 family rejects any ``temperature`` other than 1.0; Gemini
spends most of its completion budget on reasoning tokens and returns an empty string if
``max_tokens`` is set to G1's 900.  :data:`MODEL_PROFILES` records what each served model accepts,
so a configuration can be described declaratively instead of by trial and error, and so the
decoding regime actually used ends up in the report (and in the cache key) rather than being
silently different per model.

**F2, visual exemplars.**  Text definitions tell the model what "atypical network" *means*; they do
not tell it what it *looks like in this imaging setup*.  The obvious fix -- three reference images
per value -- is unaffordable: ~490 input tokens per image times 28 values times 3 would be
~41,000 tokens per case, roughly 22x the entire zero-shot prompt.  :func:`build_contact_sheets`
instead renders **one composite contact sheet per criterion**, each panel a single TRAIN exemplar
captioned with the value it illustrates.  All 28 permitted values are covered for
:data:`~medguard_v2.criteria.schema.CRITERION_NAMES` at a per-case cost of ~2.6k image tokens --
see :func:`sheet_token_cost`.

Exemplars are chosen as the **medoid** of the train cases carrying that value in the frozen DermLIP
feature space that G1b already cached: the case closest to its value's centroid is the least
idiosyncratic example available, the choice is deterministic, it costs nothing, and it never looks
at valid or test.  ``select_exemplars`` raises if handed a case from any split but ``train``.

**F3, the clinical view.**  Inherited from :class:`CriterionAssessor`, which already honours
``has_distinct_views`` so the nine Derm7pt cases whose "clinical" image is a copy of the
dermoscopic file are not sent the same picture twice.

Two invariants this module is built to preserve
-----------------------------------------------
*The zero-shot path stays byte-identical.*  With no exemplars configured,
:meth:`FewShotAssessor.build_request` delegates straight to the parent, so the request identity --
and therefore the cache key -- matches what ``g1_criterion_readability`` produced.  G1's paid
answers replay for free, and the "baseline" arm of G1c is literally G1's own numbers restricted to
the subset rather than a re-run that might drift.

*The criterion definitions are untouched.*  The few-shot prompt is the zero-shot prompt with its
first paragraph (the image inventory) swapped and a reference-sheet block inserted.  Every
criterion framing, every value definition and every rule is the same text.  F2 therefore varies
"has seen an example" and nothing else.  :func:`build_fewshot_user_prompt` asserts the structure it
edits and raises if the base prompt ever drifts, rather than silently producing a prompt that
describes images that are not there.
"""
from __future__ import annotations

import hashlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from medguard_v2.criteria.schema import CRITERIA_BY_NAME, CRITERION_NAMES
# ``_ordered_vocabulary`` and ``_file_sha256`` are imported rather than reimplemented on purpose:
# the sheet's panel order must be the *same* order the prompt lists the permitted values in, and
# the sheet's digest must be computed exactly as the assessor computes an image digest. A local
# copy of either would drift silently.
from medguard_v2.perception.criterion_assessor import (
    AssessorRequest,
    CriterionAssessment,
    CriterionAssessor,
    _file_sha256,
    _ordered_vocabulary,
    build_user_prompt,
)

__all__ = [
    "FEWSHOT_DESIGN_VERSION",
    "EXEMPLAR_SELECTION_VERSION",
    "PANEL_PX",
    "CAPTION_PX",
    "SMART_RESIZE_FACTOR",
    "ModelProfile",
    "MODEL_PROFILES",
    "profile_for",
    "grid_shape",
    "sheet_token_cost",
    "fewshot_image_token_cost",
    "select_exemplars",
    "exemplar_gaps",
    "build_contact_sheets",
    "build_fewshot_user_prompt",
    "ExemplarSheet",
    "FewShotAssessor",
]

#: Bump when the sheet geometry, captioning or prompt block changes.  Not itself part of the cache
#: key -- the rendered sheet bytes and the full prompt text are, and both change with it -- but it
#: is written into the report so two runs can be told apart by name.
FEWSHOT_DESIGN_VERSION = "g1c-contact-sheet-v1"

#: Bump when the exemplar *selection rule* changes.  Same reasoning.
EXEMPLAR_SELECTION_VERSION = "dermlip-medoid-train-only-v1"

#: Panel edge and caption strip, both multiples of the Qwen-VL 28px visual-token grid so the
#: token cost of a sheet is exact rather than an estimate that rounding moves around.
PANEL_PX = 252
CAPTION_PX = 28
SMART_RESIZE_FACTOR = 28

#: Widest grid we will lay a sheet out in.  A single row of eight panels is 2016px wide and 280px
#: tall; providers that cap the long edge (Anthropic at 1568px) would shrink such a strip hard,
#: costing exactly the fine structure the exemplar exists to show.
MAX_SHEET_COLUMNS = 4

#: Longest permitted edge ratio for a sheet.  Minimising cells alone picks a single column for any
#: prime panel count (five values would become a 252x1400 ribbon), which is the cheapest layout and
#: the worst one to survive a provider's long-edge cap.
MAX_SHEET_ASPECT = 4.0


# ---------------------------------------------------------------------------
# F1 -- what each served model will actually accept
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelProfile:
    """Decoding parameters a gateway model accepts, and what it costs.

    ``temperature`` and ``seed`` are not preferences: sending ``seed`` to an Anthropic model or
    ``temperature=0`` to a GPT-5 model is a 400 from the gateway, and both are recorded in the
    assessor's cache key, so they have to be decided before the first call rather than patched
    afterwards.

    ``max_tokens`` is a truncation guard, not a decoding knob -- but it stops being one when a
    reasoning model spends the whole budget thinking.  Measured on one probe call per model:
    Gemini 3.1 Pro emitted 896 reasoning tokens and *no JSON* at G1's 900, and parsed cleanly at
    3000.  The per-model value here is the one that leaves room for an answer.
    """

    model: str
    temperature: float = 0.0
    seed: Optional[int] = 20260810
    max_tokens: int = 900
    price_in_per_mtok: Optional[float] = None
    price_out_per_mtok: Optional[float] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "price_in_per_mtok": self.price_in_per_mtok,
            "price_out_per_mtok": self.price_out_per_mtok,
            "note": self.note,
        }

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        if self.price_in_per_mtok is None or self.price_out_per_mtok is None:
            return None
        return (
            prompt_tokens / 1e6 * self.price_in_per_mtok
            + completion_tokens / 1e6 * self.price_out_per_mtok
        )


#: Prices are the gateway's own ``/models`` rate card read on 2026-08-10, in USD per 1e6 tokens.
#: They are recorded so a spend figure can be reproduced, not because they are contracted.
MODEL_PROFILES: Mapping[str, ModelProfile] = {
    "qwen/qwen3-vl-32b-instruct": ModelProfile(
        "qwen/qwen3-vl-32b-instruct", 0.0, 20260810, 900, 0.104, 0.416,
        "G1 baseline; these are exactly G1's decoding parameters, so its cache replays",
    ),
    "qwen/qwen3-vl-235b-a22b-instruct": ModelProfile(
        "qwen/qwen3-vl-235b-a22b-instruct", 0.0, 20260810, 900, 0.26, 1.04,
        "same family, ~7x the parameters: separates capacity from vendor",
    ),
    "google/gemini-3.1-pro-preview": ModelProfile(
        "google/gemini-3.1-pro-preview", 0.0, 20260810, 3000, 2.0, 12.0,
        "reasoning model; returns no JSON at max_tokens=900 (896 reasoning tokens, truncated)",
    ),
    "google/gemini-3.5-flash": ModelProfile(
        "google/gemini-3.5-flash", 0.0, 20260810, 3000, 1.5, 9.0,
        "reasoning model; same truncation hazard as Gemini Pro",
    ),
    "anthropic/claude-opus-5": ModelProfile(
        "anthropic/claude-opus-5", 0.0, None, 900, 5.0, 25.0,
        "does not accept 'seed'; determinism rests on temperature=0 alone",
    ),
    "anthropic/claude-sonnet-5": ModelProfile(
        "anthropic/claude-sonnet-5", 1.0, None, 900, 2.0, 10.0,
        "accepts neither 'seed' nor 'temperature'",
    ),
    "openai/gpt-5.6-terra": ModelProfile(
        "openai/gpt-5.6-terra", 1.0, 20260810, 4000, 2.0, 12.0,
        "GPT-5 family accepts only temperature=1.0; reasoning tokens billed as completion",
    ),
    "openai/gpt-5.4": ModelProfile(
        "openai/gpt-5.4", 1.0, 20260810, 4000, 2.5, 15.0,
        "GPT-5 family accepts only temperature=1.0",
    ),
    "x-ai/grok-4.5": ModelProfile(
        "x-ai/grok-4.5", 0.0, 20260810, 3000, 2.0, 6.0,
        "reasoning model; accepts temperature and seed",
    ),
}


def profile_for(model: str) -> ModelProfile:
    """Profile for ``model``, or a conservative default for one we have not probed.

    The default deliberately keeps G1's temperature and seed: an unprobed model that rejects them
    fails loudly on its first call, which is the right outcome.  Silently dropping parameters
    would make two configurations differ in a way the report could not see.
    """
    known = MODEL_PROFILES.get(model)
    if known is not None:
        return known
    return ModelProfile(model, 0.0, 20260810, 900, None, None, "unprobed model; default profile")


# ---------------------------------------------------------------------------
# F2 -- contact-sheet geometry and its exact token cost
# ---------------------------------------------------------------------------
def grid_shape(n_panels: int) -> Tuple[int, int]:
    """(columns, rows) for ``n_panels``: fewest grid *cells* subject to an aspect cap.

    Empty cells are not free -- they occupy pixels and therefore visual tokens -- so the layout
    minimises ``cols * ceil(n/cols)`` rather than picking a fixed width.  But minimising cells
    alone degenerates to one column whenever ``n_panels`` is prime, so layouts whose edge ratio
    exceeds :data:`MAX_SHEET_ASPECT` are excluded first.  Ties go to the wider grid.
    """
    if n_panels < 1:
        raise ValueError("a contact sheet needs at least one panel")
    cell_h = PANEL_PX + CAPTION_PX
    best: Optional[Tuple[int, int, int]] = None  # (cells, -cols, cols) sort key carrier
    fallback: Optional[Tuple[int, int, int]] = None
    for cols in range(1, MAX_SHEET_COLUMNS + 1):
        rows = math.ceil(n_panels / cols)
        width, height = cols * PANEL_PX, rows * cell_h
        candidate = (cols * rows, -cols, cols)
        if fallback is None or candidate < fallback:
            fallback = candidate
        if max(width, height) / min(width, height) > MAX_SHEET_ASPECT:
            continue
        if best is None or candidate < best:
            best = candidate
    chosen = best if best is not None else fallback
    assert chosen is not None
    cols = chosen[2]
    return cols, math.ceil(n_panels / cols)


def sheet_token_cost(n_panels: int) -> int:
    """Visual tokens one sheet costs under the Qwen-VL 28x28 grid, +2 delimiters.

    Exact rather than estimated: every dimension is a multiple of :data:`SMART_RESIZE_FACTOR`, so
    ``smart_resize`` is the identity and the count is ``pixels / 784``.
    """
    cols, rows = grid_shape(n_panels)
    width = cols * PANEL_PX
    height = rows * (PANEL_PX + CAPTION_PX)
    return (width * height) // (SMART_RESIZE_FACTOR * SMART_RESIZE_FACTOR) + 2


def fewshot_image_token_cost(criteria: Sequence[str] = CRITERION_NAMES) -> Dict[str, Any]:
    """Per-case image-token bill for a given set of reference sheets."""
    per_sheet = {
        name: {
            "n_values": len(_ordered_vocabulary(name)),
            "grid": list(grid_shape(len(_ordered_vocabulary(name)))),
            "image_tokens": sheet_token_cost(len(_ordered_vocabulary(name))),
        }
        for name in criteria
    }
    return {
        "panel_px": PANEL_PX,
        "caption_px": CAPTION_PX,
        "per_sheet": per_sheet,
        "n_sheets": len(per_sheet),
        "total_image_tokens": sum(v["image_tokens"] for v in per_sheet.values()),
    }


# ---------------------------------------------------------------------------
# F2 -- exemplar selection, train only
# ---------------------------------------------------------------------------
def select_exemplars(
    train_cases: Sequence[Any],
    features: Mapping[str, Any],
    *,
    criteria: Sequence[str] = CRITERION_NAMES,
) -> Dict[str, Dict[str, int]]:
    """Pick one TRAIN exemplar per permitted value: the medoid in DermLIP feature space.

    "Medoid" here is the case whose feature vector is closest to the mean of all train cases
    carrying that value -- the least idiosyncratic available example, chosen without looking at
    any label the gate will be scored against and without a single API call.

    Raises if any supplied case is not from ``train``.  Drawing an exemplar from valid would leak
    the evaluation cohort into the prompt; drawing one from test would be worse.  This is cheap to
    get wrong by accident and impossible to detect afterwards from the numbers, so it is a hard
    error rather than a warning.
    """
    import numpy as np

    foreign = sorted({str(case.split) for case in train_cases} - {"train"})
    if foreign:
        raise ValueError(f"exemplars may only come from the train split; got splits {foreign}")

    chosen: Dict[str, Dict[str, int]] = {}
    for name in criteria:
        by_value: Dict[str, List[Any]] = {}
        for case in train_cases:
            value = str(case.criteria[name])
            if str(case.case_num) in features:
                by_value.setdefault(value, []).append(case)

        per_value: Dict[str, int] = {}
        for value in _ordered_vocabulary(name):
            members = by_value.get(value, [])
            if not members:
                # A vocabulary value with no train example cannot be illustrated.  Derm7pt has
                # one: ``pigmentation = "localized regular"`` occurs in neither train nor valid.
                # It is left off the sheet and reported by :func:`exemplar_gaps` -- never filled
                # from valid, and never quietly dropped from the *permitted values* the prompt
                # still offers, because that would change the label space F2 is meant to hold
                # fixed.
                continue
            matrix = np.stack([np.asarray(features[str(c.case_num)], dtype=float) for c in members])
            centroid = matrix.mean(axis=0)
            distances = np.linalg.norm(matrix - centroid, axis=1)
            order = sorted(
                range(len(members)), key=lambda i: (float(distances[i]), members[i].case_num)
            )
            per_value[value] = int(members[order[0]].case_num)
        chosen[name] = per_value
    return chosen


def exemplar_gaps(
    exemplars: Mapping[str, Mapping[str, int]], *, criteria: Sequence[str] = CRITERION_NAMES
) -> Dict[str, List[str]]:
    """Permitted values that no reference sheet can illustrate, because train has no example.

    Reported rather than papered over: a value the model is offered but never shown is a known
    weakness of the F2 arm, and it must not be mistaken for a rendering bug later.
    """
    return {
        name: sorted(set(CRITERIA_BY_NAME[name].vocabulary) - set(exemplars.get(name, {})))
        for name in criteria
        if set(CRITERIA_BY_NAME[name].vocabulary) - set(exemplars.get(name, {}))
    }


# ---------------------------------------------------------------------------
# F2 -- rendering
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExemplarSheet:
    criterion: str
    path: Path
    sha256: str
    values: Tuple[str, ...]
    exemplar_case_nums: Tuple[int, ...]
    columns: int
    rows: int
    image_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "filename": self.path.name,
            "sha256": self.sha256,
            "values_in_reading_order": list(self.values),
            "exemplar_case_nums": list(self.exemplar_case_nums),
            "grid_columns": self.columns,
            "grid_rows": self.rows,
            "image_tokens_qwen_grid": self.image_tokens,
        }


def _panel(image_path: Path, size: int):
    """Aspect-preserving resize to short side ``size``, then a centre crop to ``size``x``size``.

    Derm7pt dermoscopic images are 768x512 with the lesion roughly centred, so a centre crop keeps
    the lesion and discards the surrounding skin -- the opposite trade to a squash, which would
    distort exactly the shapes (round globules, mesh holes) the panel exists to show.
    """
    from PIL import Image

    with Image.open(image_path) as raw:
        image = raw.convert("RGB")
        width, height = image.size
        scale = size / min(width, height)
        new = (max(size, round(width * scale)), max(size, round(height * scale)))
        image = image.resize(new, Image.LANCZOS)
        left = (new[0] - size) // 2
        top = (new[1] - size) // 2
        return image.crop((left, top, left + size, top + size))


def build_contact_sheets(
    exemplars: Mapping[str, Mapping[str, int]],
    cases_by_num: Mapping[int, Any],
    out_dir: Path,
    *,
    criteria: Sequence[str] = CRITERION_NAMES,
) -> List[ExemplarSheet]:
    """Render one captioned contact sheet per criterion.  Deterministic, idempotent.

    The sheets are derived Derm7pt images.  Derm7pt is CC BY-NC-ND, so they must not be
    redistributed: ``out_dir`` gets a directory-local ``.gitignore`` and the report records only
    each sheet's SHA-256, which makes the derivation auditable without shipping the pixels.
    """
    from PIL import Image, ImageDraw, ImageFont

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gitignore = out_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "# Derived Derm7pt images (CC BY-NC-ND): regenerable, never redistributed.\n*\n",
            encoding="utf-8",
        )

    try:
        font = ImageFont.load_default(size=17)
    except TypeError:  # Pillow < 9.2 has no sized default font
        font = ImageFont.load_default()

    sheets: List[ExemplarSheet] = []
    for name in criteria:
        per_value = exemplars.get(name, {})
        values = tuple(v for v in _ordered_vocabulary(name) if v in per_value)
        if not values:
            raise ValueError(f"{name}: no exemplar available for any value")
        cols, rows = grid_shape(len(values))
        cell_h = PANEL_PX + CAPTION_PX
        sheet = Image.new("RGB", (cols * PANEL_PX, rows * cell_h), (32, 32, 32))
        draw = ImageDraw.Draw(sheet)

        for index, value in enumerate(values):
            case = cases_by_num[per_value[value]]
            x = (index % cols) * PANEL_PX
            y = (index // cols) * cell_h
            sheet.paste(_panel(Path(case.derm_path), PANEL_PX), (x, y))
            draw.rectangle([x, y + PANEL_PX, x + PANEL_PX - 1, y + cell_h - 1], fill=(255, 255, 255))
            draw.text((x + 5, y + PANEL_PX + 5), value, fill=(0, 0, 0), font=font)
            draw.rectangle([x, y, x + PANEL_PX - 1, y + cell_h - 1], outline=(0, 0, 0), width=1)

        path = out_dir / f"{name}.png"
        sheet.save(path, format="PNG", optimize=True)
        sheets.append(
            ExemplarSheet(
                criterion=name,
                path=path,
                sha256=_file_sha256(path),
                values=values,
                exemplar_case_nums=tuple(int(per_value[v]) for v in values),
                columns=cols,
                rows=rows,
                image_tokens=sheet_token_cost(len(values)),
            )
        )
    return sheets


# ---------------------------------------------------------------------------
# F2 -- the prompt
# ---------------------------------------------------------------------------
_SHEET_BLOCK_HEADER = "REFERENCE SHEETS"


def build_fewshot_user_prompt(sheets: Sequence[ExemplarSheet], *, has_clinical: bool = False) -> str:
    """G1's user prompt with a new image inventory and a reference-sheet block.

    Everything downstream of the inventory -- criterion framings, value definitions, the five
    rules, the JSON template -- is :func:`build_user_prompt`'s text verbatim.  That is what makes
    F2 a clean contrast: the model is given pictures of the values, not a different description of
    them.

    Raises if the base prompt's first paragraph is no longer the image inventory.  Editing the
    inventory by position is fragile, and a silent failure here would send seven images the prompt
    never mentions.
    """
    if not sheets:
        raise ValueError("build_fewshot_user_prompt requires at least one reference sheet")

    base = build_user_prompt(has_clinical=has_clinical)
    head, separator, tail = base.partition("\n\n")
    if not separator or not head.startswith("You are given "):
        raise RuntimeError(
            "criterion_assessor.build_user_prompt no longer starts with an image-inventory "
            "paragraph; the few-shot prompt builder must be updated to match it"
        )

    lesion_images = 2 if has_clinical else 1
    total = lesion_images + len(sheets)
    if has_clinical:
        inventory = (
            f"You are given {total} images. IMAGE 1 is the dermoscopic image of the lesion you must "
            "assess and IMAGE 2 is a clinical (macro) photograph of that same lesion. Judge the "
            "criteria from IMAGE 1; use IMAGE 2 only for context. The remaining images are "
            "reference sheets, described below."
        )
    else:
        inventory = (
            f"You are given {total} images. IMAGE 1 is the dermoscopic image of the lesion you must "
            "assess. The remaining images are reference sheets, described below."
        )

    lines = [
        _SHEET_BLOCK_HEADER,
        "Each reference sheet shows one example of every permitted value of a single criterion,",
        "taken from a separate set of training lesions. Panels read left to right, then top to",
        "bottom, and every panel is captioned with the value it illustrates.",
        "",
    ]
    for offset, sheet in enumerate(sheets):
        index = lesion_images + 1 + offset
        lines.append(
            f'  IMAGE {index} -- reference sheet for "{sheet.criterion}" '
            f"({sheet.columns} columns x {sheet.rows} rows): "
            + ", ".join(f'"{value}"' for value in sheet.values)
        )
    lines += [
        "",
        "Use the sheets to calibrate what each value looks like under this imaging setup. They are",
        "illustrations, not answers: the reference lesions are unrelated to the lesion in IMAGE 1,",
        "and you must never report a criterion value because you saw it on a sheet. Assess IMAGE 1",
        "and nothing else.",
    ]
    unillustrated = sorted(
        {
            value
            for sheet in sheets
            for value in _ordered_vocabulary(sheet.criterion)
            if value not in sheet.values
        }
    )
    if unillustrated:
        # Without this the sheet reads as the label space, and a value with no panel silently
        # stops being answerable -- which would be F2 changing the vocabulary, not the exemplars.
        lines += [
            "",
            "A few permitted values have no reference example available and therefore no panel. "
            "They are still permitted answers; judge them from their written definition below.",
        ]
    sheet_block = "\n".join(lines)

    return f"{inventory}\n\n{sheet_block}\n\n{tail}"


# ---------------------------------------------------------------------------
# The assessor
# ---------------------------------------------------------------------------
class FewShotAssessor(CriterionAssessor):
    """:class:`CriterionAssessor` plus reference sheets and a thread pool.

    With ``sheets=None`` every request is built by the parent, unchanged, so the cache key matches
    what G1 wrote and G1's paid answers replay for nothing.  That is the whole reason this is a
    subclass rather than a fork: the "no few-shot" arm of G1c has to be *the same request*, not a
    reconstruction of it.

    Parallelism is per-thread-*assessor*, not per-thread-client.  ``CriterionAssessor._call`` reads
    ``client.last_call_metadata`` after the call returns, which is a single mutable attribute; two
    threads sharing one client would attribute each other's token counts. Each worker therefore
    gets its own assessor and its own client, and they share only the cache, whose writes are
    already locked and whose reads are pure.
    """

    def __init__(
        self,
        *args: Any,
        sheets: Optional[Sequence[ExemplarSheet]] = None,
        workers: int = 1,
        client_factory: Optional[Callable[[], Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.sheets: Tuple[ExemplarSheet, ...] = tuple(sheets or ())
        self.workers = max(1, int(workers))
        self.client_factory = client_factory

    # -- request construction ------------------------------------------------------------
    def build_request(self, case) -> AssessorRequest:
        if not self.sheets:
            return super().build_request(case)

        base = super().build_request(case)
        sheet_paths = tuple(sheet.path for sheet in self.sheets)
        sheet_roles = tuple(f"reference:{sheet.criterion}" for sheet in self.sheets)
        sheet_hashes = tuple(sheet.sha256 for sheet in self.sheets)

        overlap = set(map(str, base.image_paths)) & set(map(str, sheet_paths))
        if overlap:
            # LLMClient.vision rejects duplicate paths outright; catching it here names the cause.
            raise RuntimeError(f"reference sheet collides with a case image: {sorted(overlap)}")

        return replace(
            base,
            image_paths=base.image_paths + sheet_paths,
            image_roles=base.image_roles + sheet_roles,
            image_sha256=base.image_sha256 + sheet_hashes,
            user_prompt=build_fewshot_user_prompt(self.sheets, has_clinical=base.has_clinical),
        )

    # -- parallel execution --------------------------------------------------------------
    def _clone_for_thread(self) -> "FewShotAssessor":
        clone = FewShotAssessor(
            client=self.client_factory() if self.client_factory else self.client,
            model=self.decoding.model,
            temperature=self.decoding.temperature,
            max_tokens=self.decoding.max_tokens,
            seed=self.decoding.seed,
            include_clinical=self.include_clinical,
            cache=self.cache,
            allow_remote=self.allow_remote,
            refresh=self.refresh,
            max_repair_attempts=self.max_repair_attempts,
            sheets=self.sheets,
            workers=1,
        )
        return clone

    def assess_cases(self, cases: Sequence[Any], *, progress=None) -> List[CriterionAssessment]:
        if self.workers <= 1 or self.client_factory is None:
            return super().assess_cases(cases, progress=progress)

        local = threading.local()

        def _run(case) -> CriterionAssessment:
            worker = getattr(local, "assessor", None)
            if worker is None:
                worker = self._clone_for_thread()
                local.assessor = worker
            return worker.assess(case)

        results: List[Optional[CriterionAssessment]] = [None] * len(cases)
        completed = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_run, case): index for index, case in enumerate(cases)}
            for future in as_completed(futures):
                # Results are stored by submission index, never by completion order: the caller
                # zips them against ``cases`` to attach gold labels, so a reordering here would
                # score every case against somebody else's annotation.
                index = futures[future]
                assessment = future.result()
                results[index] = assessment
                completed += 1
                if progress is not None:
                    progress(completed, len(cases), assessment)
        missing = [index for index, value in enumerate(results) if value is None]
        if missing:
            raise RuntimeError(f"parallel assessment lost {len(missing)} case(s): {missing[:10]}")
        return [value for value in results if value is not None]


# ---------------------------------------------------------------------------
# Convenience: what a config's prompt costs before anything is sent
# ---------------------------------------------------------------------------
def prompt_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def default_client_factory(model: str, max_retries: Optional[int] = None) -> Callable[[], Any]:
    """A factory that builds one v1 gateway client per call.  No credential is read here."""

    def _factory() -> Any:
        from medguard.pipeline.llm import LLMClient  # noqa: PLC0415 -- lazy, keeps imports light

        kwargs: Dict[str, Any] = {"model": model}
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        return LLMClient(**kwargs)

    return _factory
