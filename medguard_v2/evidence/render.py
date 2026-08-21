"""Render a decision contract for a clinician, in three explicit tiers.

Written against one test: **a dermatologist with the image in front of them should be able to
audit this in about ten seconds and disagree with any single line** -- and, added by the tier
redesign, **a routine case should read as routine in about three**. Boring cases earn a boring
report; warning furniture on every case teaches readers to ignore warnings.

The three tiers (marked in the code below):

- **TIER 1 -- the 3-second glance.** Headline (assessment + one-clause reason + trust chip),
  the verified case brief when one exists, and the Next step. A routine dual-view case reaches
  its "### Next step" line within eight non-blank lines and carries no warning furniture.
- **TIER 2 -- the 10-second audit.** The conflict warning when it fires (immediately after
  Tier 1: a disagreement between the two views is the most decision-relevant thing the system
  can say), the two-view table, the criteria table with points so the score can be recomputed
  by eye, the hinge / robustness / resolution surface, and ONE guideline sentence quoted
  verbatim with its citation. Citations render as "id · version · ✓" -- the ✓ asserts the
  content hash verified (rendering only happens after verification passes); the hex prefix
  itself is Tier-3 material.
- **TIER 3 -- on demand only** (``include_appendix=True``): similar confirmed cases, the
  saliency caveat pointer, the remaining guideline recommendations, the record hashes, and the
  full per-criterion LOCO surface. The default view is byte-independent of everything here.

Nothing at any tier is generated prose. Every sentence is either a fixed template or a verbatim
quote, so there is no surface for a language model to invent a recommendation. Guideline text is
quoted verbatim with its citation, or its absence is stated in the same place a recommendation
would have appeared -- a silent gap reads as "nothing to say"; an explicit one reads as "no
guidance exists", and those are different clinical facts.

Both contract types render here: the dual-view :class:`DecisionContract` and the classifier-only
:class:`~medguard_v2.evidence.single_view.SingleViewContract` (imaging sources with no 7-point
annotations). The single-view report renders the checklist's table slot as the explicit
unavailable-disclosure and its Next step carries the clinical-correlation sentence -- one view
must never silently read like two.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

from medguard_v2.criteria.aggregate import (
    leave_one_criterion_out,
    recommend_management,
    score_criteria,
    verdict_hinges,
)
from medguard_v2.criteria.schema import CRITERIA_BY_NAME, MELANOMA_SUSPICION_THRESHOLD
from medguard_v2.evidence.contract import DecisionContract, SimilarCasesContext
from medguard_v2.evidence.guideline_binding import (
    GROUP_FULL_NAME as _FULL_NAME,
)
from medguard_v2.evidence.guideline_binding import BindingResult, Recommendation
from medguard_v2.evidence.single_view import SingleViewContract, single_view_action

if TYPE_CHECKING:
    # Only for the type hint on render_markdown's optional `brief` parameter. summary.py imports
    # render_headline from THIS module, so a runtime import here would be circular; render.py
    # only ever touches `brief` by duck-typed attribute access (.verified, .fallback_used,
    # .text), so no runtime import is needed at all.
    from medguard_v2.evidence.summary import CaseBrief

RenderableContract = Union[DecisionContract, SingleViewContract]

#: pigment_network's cheapest NON-scoring reading is "typical" -- the network is there, just not
#: atypical -- which is the more clinically natural alternate hypothesis than "there is no
#: network at all". Every other criterion's cheapest non-scoring reading is simply "absent":
#: checked against schema.py, "absent" is in every criterion's vocabulary and in none of their
#: scoring_values, so it always scores zero.
_CHEAPEST_NON_SCORING_VALUE = {"pigment_network": "typical"}


def _non_scoring_reading(name: str) -> str:
    return _CHEAPEST_NON_SCORING_VALUE.get(name, "absent")


def _scoring_reading(name: str) -> str:
    """A deterministic representative value that earns this criterion's points.

    Every value in a criterion's ``scoring_values`` earns the same ``spec.points`` (schema.py
    defines no per-value weighting), so any one will do; sorted for reproducibility.
    """
    return sorted(CRITERIA_BY_NAME[name].scoring_values)[0]


def render_headline(contract: RenderableContract) -> str:
    """R1: the one-line verdict, first thing after the case header, before everything else.

    Derived entirely from existing contract fields: no new judgement is made here. Dual-view:
    the checklist-implied action/reason plus the agreement-state trust chip. Single-view: the
    binding-derived action (see ``single_view_action`` -- follow the cited record, or state the
    absence) plus the ``single-view`` chip, so one view can never read like two at a glance.
    """
    if isinstance(contract, SingleViewContract):
        action, reason = single_view_action(contract)
        return f"**Assessment: {action}** — {reason} · single-view"

    action = contract.criteria_implied_action
    reason = contract.criteria_implied_reason
    if not contract.score.decidable:
        tag = "checklist unresolved"
    elif contract.conflict.present:
        tag = "VIEWS DISAGREE"
    else:
        tag = "views agree"
    return f"**Assessment: {action}** — {reason} · {tag}"


#: MG52 (corpus human-factors finding): a displayed raw confidence NUMBER measurably drives
#: clinician overreliance on the classifier, independent of the number's actual accuracy. Binning
#: it into a coarse word and making THAT the headline -- with the number kept in parentheses for
#: anyone auditing the line -- keeps the number available without making it the thing a reader
#: anchors their trust on.
def _confidence_band(confidence: float) -> str:
    if confidence >= 0.75:
        word = "high"
    elif confidence >= 0.5:
        word = "moderate"
    else:
        word = "low"
    return f"{word} model agreement ({confidence:.2f})"


def _hinge_line(contract: DecisionContract) -> str:
    """R2: the LOCO surface, rendered. Either names the single criterion this verdict hinges on
    and what reading it the other way would do to the score and action, or -- when no criterion
    alone can move it -- states that robustness explicitly, because it is itself a clinically
    meaningful fact about how much to trust the verdict."""
    # Reads .score_min throughout below rather than .score: when some OTHER criterion is also
    # indeterminate in this contract (rare -- the hinge criterion itself cannot be, see
    # verdict_hinges' docstring), substituting only the hinge criterion still leaves a range, and
    # .score_min is its honest conservative lower bound rather than a value .score would raise on.
    margins = contract.criterion_margins
    hinges = verdict_hinges(contract.criteria, margins)
    if not hinges:
        return "_No single criterion changes this verdict._"

    if margins is not None:
        # N2: when the promoted perception path supplies per-criterion margins, verdict_hinges has
        # already ordered the hinges by ascending model margin, so hinges[0] is the one the reader
        # is least certain about -- the most valuable one to re-examine. Surface that, not the
        # highest-point one.
        top = hinges[0]
    else:
        # Fallback (no margins available -- the pre-promotion path): hinges is in canonical
        # criterion order and the two directions never mix within one case (see verdict_hinges'
        # docstring), so ranking by points_contributed with max()'s left-to-right tie-break
        # deterministically picks the highest-point hinge in canonical order.
        top = max(hinges, key=lambda h: h["points_contributed"])
    others = len(hinges) - 1

    name = str(top["name"])
    label = name.replace("_", " ")
    value = str(top["value"])
    points = int(top["points_contributed"])

    if top["flipped_verdict_direction"] == "erodes":
        alt = _non_scoring_reading(name)
        alt_score = score_criteria({**contract.criteria, name: alt})
        alt_action, _ = recommend_management(alt_score)
        body = (
            f"This verdict hinges on {label} ({value}, +{points}): read as {alt}, the score "
            f"falls to {alt_score.score_min} and {alt_action} applies"
        )
    else:
        alt = _scoring_reading(name)
        alt_score = score_criteria({**contract.criteria, name: alt})
        alt_action, _ = recommend_management(alt_score)
        body = (
            f"This verdict hinges on {label} (currently {value}): read as {alt}, the score "
            f"would rise to {alt_score.score_min} and {alt_action} would apply"
        )
    if others:
        # "criterion/criteria" matches this file's existing convention for an uninflected count
        # (see the unresolved-criteria line below) rather than a hand-rolled plural rule.
        body += f" — and {others} other criterion/criteria"
    return f"_{body}._"


def _resolution_protocol_lines(contract: DecisionContract) -> List[str]:
    """R3: replaces the old dead-end 'undetermined ... re-imaging or specialist review is
    needed' sentence. Names the single most decisive unresolved criterion -- the one with the
    most points at stake, so canonical order breaks a major/major or minor/minor tie -- and both
    of its resolutions' concrete outcomes, computed the same deterministic way the score itself
    is. When more than one criterion is unresolved, only the most decisive is expanded; the rest
    are named so their existence is not lost, not silently dropped.
    """
    unresolved = contract.unresolved_criteria
    per_max_points = {row.name: row.max_points for row in contract.score.per_criterion}
    resolve_name = max(unresolved, key=lambda n: per_max_points[n])

    scoring_value = _scoring_reading(resolve_name)
    non_scoring_value = "absent"  # R3 always contrasts against "absent" -- a boundary check on
    # what's unresolved, not R2's clinically-likeliest-alternate reading (_non_scoring_reading).
    score_scoring = score_criteria({**contract.criteria, resolve_name: scoring_value})
    action_scoring, _ = recommend_management(score_scoring)
    score_absent = score_criteria({**contract.criteria, resolve_name: non_scoring_value})
    action_absent, _ = recommend_management(score_absent)

    label = resolve_name.replace("_", " ")
    lines = [
        f"_Resolve {label}: {scoring_value} → score {score_scoring.score_min}, "
        f"{action_scoring}; {non_scoring_value} → score {score_absent.score_min}, "
        f"{action_absent}._"
    ]
    remaining = [c for c in unresolved if c != resolve_name]
    if remaining:
        others = ", ".join(c.replace("_", " ") for c in remaining)
        lines.append(f"_{len(remaining)} other unresolved criterion/criteria remain ({others})._")
    return lines


#: The single caveat the saliency pointer is allowed to carry. GradCAM over the frozen DermLIP
#: backbone is QUANTITATIVELY faithful (deletion/insertion beats a random masking order at p<1e-5,
#: research/evidence/derm7pt_v1/attribution_faithfulness_v1.json) but VISUALLY diffuse and prone to
#: attend to image artifacts, so it is offered only as a caveated on-demand pointer, never as primary
#: localization. render never computes saliency (that needs the model); this is the pointer only.
_SALIENCY_CAVEAT = (
    "A GradCAM saliency map over the frozen DermLIP backbone is available on request but is offered "
    "only as a caveated pointer: **faithful by deletion/insertion but spatially diffuse; not a "
    "substitute for clinician localization.** It beats a random masking order at p<1e-5 "
    "(deletion and insertion, `research/evidence/derm7pt_v1/attribution_faithfulness_v1.json`) yet "
    "attends to artifacts, so it is not surfaced as primary evidence and no region is drawn here."
)

#: The sentence every single-view Next step must carry. One view means no independent
#: cross-check ever ran, and the reader must weight their own examination accordingly.
_SINGLE_VIEW_CORRELATION = (
    "Single-view decision: no independent checklist cross-check was possible — clinical "
    "correlation carries more weight than on a dual-view case."
)


def _citation_line(rec: Recommendation) -> str:
    """Tier 1-2 citation: id and version stay visible (provenance auditable at a glance); the ✓
    asserts the content hash verified -- true by construction, since rendering only happens after
    ``verification_failures()`` came back empty. The hex prefix itself is Tier-3 material."""
    return f"— {rec.citation_id} · {rec.source_version} · ✓"


def _citation_line_hashed(rec: Recommendation) -> str:
    """Tier-3 citation: the content-hash prefix, for a reader auditing the record itself."""
    return f"— {rec.citation_id} · {rec.source_version} · `{rec.content_sha256[:16]}`"


def _withheld_lines(case_id: str, failures: List[str]) -> str:
    lines = [
        f"## Case {case_id} — OUTPUT WITHHELD",
        "",
        "This decision failed its own verification and must not be acted on:",
        "",
    ]
    lines += [f"- {f}" for f in failures]
    return "\n".join(lines)


def _guidance_lines(guideline: BindingResult, diagnosis: str) -> List[str]:
    """Tier 2 guideline evidence: ONE sentence quoted verbatim (the first bound recommendation),
    or the explicit absence. Remaining recommendations are named, not shown -- they move behind
    Tier 3 so a routine reader is not handed three block quotes to skim past."""
    lines = ["### Authoritative guidance", ""]
    if guideline.covered:
        rec = guideline.recommendations[0]
        lines += [f"> {rec.text.strip()}", "", _citation_line(rec), ""]
        n_more = len(guideline.recommendations) - 1
        if n_more:
            lines += [f"_+{n_more} more recommendation(s) on request._", ""]
    else:
        lines += [
            f"**No recommendation available for {_FULL_NAME.get(diagnosis, diagnosis)}.** "
            f"{guideline.absence_reason}",
            "",
            "_No management advice is offered here because none can be cited. This is a stated "
            "limit of the guideline corpus, not an assessment that no action is required._",
            "",
        ]
    return lines


def _next_step_lines_dual(contract: DecisionContract) -> List[str]:
    """R5: every case gets a Next step, and the tier redesign moves it INTO the 3-second glance
    -- assessment and action are what a reader needs before deciding whether to audit. Branch
    logic unchanged (escalate -> specialist review; else the checklist-implied action with its
    citation or the absence pointer); only the undecidable wording is corrected -- an unresolved
    checklist has not DISAGREED with anything, and saying so would misstate the case."""
    lines = ["### Next step", ""]
    if contract.conflict.escalate:
        if contract.conflict.present:
            lines += [
                "Specialist review. The two assessments disagree, and this system does not "
                "adjudicate between them.",
                "",
            ]
        else:
            lines += [
                "Specialist review. The checklist has not reached a verdict — resolve the "
                "unresolved criteria (protocol below).",
                "",
            ]
    elif contract.guideline.covered:
        first_citation = contract.guideline.recommendations[0].citation_id
        lines += [
            f"{contract.criteria_implied_action.capitalize()} — see {first_citation} "
            "(quoted below).",
            "",
        ]
    else:
        lines += [
            f"{contract.criteria_implied_action.capitalize()} — no citable guidance (see below).",
            "",
        ]
    return lines


# ---------------------------------------------------------------------------------------------
# TIER 3 building blocks (rendered ONLY under include_appendix=True; the default view is
# byte-independent of everything below)
# ---------------------------------------------------------------------------------------------


def _similar_cases_lines(sc: SimilarCasesContext) -> List[str]:
    """R-XAI: the ON-DEMAND, faithful-by-construction case-based explanation block.

    Rendered only behind the on-request appendix (never in the default 10-second view -- MG52:
    a displayed explanation measurably raises clinician cognitive load, so this is progressive
    disclosure). Shows the query's nearest confirmed TRAIN cases in the model's own frozen feature
    space and the nearest different-label case as a boundary counterfactual.
    """
    lines: List[str] = [
        "### Similar confirmed cases (on request)",
        "",
        "_On request only — not shown in the 10-second view (MG52: a displayed explanation raises "
        "cognitive load, so this is progressive disclosure)._",
        "",
        f"Nearest confirmed **training** cases to this lesion in the model's own frozen DermLIP "
        f"feature space ({sc.feature_space or 'frozen image embedding'}). Faithful **by "
        f"construction**: this is the identity of the case in the representation the reader itself "
        f"reads, not a post-hoc rationalisation. Train-only and leave-one-out, so the case never "
        f"retrieves itself. Confirmed diagnoses, not system predictions.",
        "",
        "| train case | confirmed diagnosis | cosine similarity | same as committed |",
        "|---|---|---:|:--:|",
    ]
    for nb in sc.neighbors:
        full = _FULL_NAME.get(nb.diagnosis, nb.diagnosis)
        same = "yes" if nb.same_label else "—"
        lines.append(f"| {nb.case_id} | {full} ({nb.diagnosis}) | {nb.similarity:.2f} | {same} |")
    lines.append("")

    committed_full = _FULL_NAME.get(sc.reference_diagnosis, sc.reference_diagnosis)
    if sc.nearest_different_label is not None:
        nd = sc.nearest_different_label
        nd_full = _FULL_NAME.get(nd.diagnosis, nd.diagnosis)
        lines += [
            f"**Nearest different-label case (decision boundary):** the committed diagnosis is "
            f"{committed_full}; the nearest confirmed **{nd_full}** is train case {nd.case_id} at "
            f"cosine {nd.similarity:.2f}. The closer that case sits, the nearer this lesion is to a "
            f"class boundary.",
            "",
        ]
    else:
        lines += [
            "_No different-label case is present in the bank (single-label bank), so no boundary "
            "counterfactual is shown._",
            "",
        ]

    lines += [
        f"— retrieval bank `{sc.bank_name}` · `{sc.bank_hash[:16]}` · k={sc.k}",
        "",
    ]
    return lines


def _remaining_recommendation_lines(
    guideline: BindingResult, contingent: Optional[BindingResult]
) -> List[str]:
    """Tier 3: the recommendations Tier 2 named but did not show, quoted verbatim with hashes."""
    committed_rest = list(guideline.recommendations[1:]) if guideline.covered else []
    contingent_rest = (
        list(contingent.recommendations[1:])
        if contingent is not None and contingent.covered
        else []
    )
    if not committed_rest and not contingent_rest:
        return []
    lines: List[str] = ["#### Remaining recommendations (on request)", ""]
    for rec in committed_rest:
        lines += [f"> {rec.text.strip()}", "", _citation_line_hashed(rec), ""]
    if contingent_rest:
        lines += ["_Contingent (applies only if the checklist's reading is right):_", ""]
        for rec in contingent_rest:
            lines += [f"> {rec.text.strip()}", "", _citation_line_hashed(rec), ""]
    return lines


def _record_hash_lines(
    guideline: BindingResult, contingent: Optional[BindingResult]
) -> List[str]:
    """Tier 3: every bound record's content-hash prefix, so the ✓ marks above are auditable."""
    recs: List[Recommendation] = []
    if guideline.covered:
        recs += list(guideline.recommendations)
    if contingent is not None and contingent.covered:
        recs += list(contingent.recommendations)
    if not recs:
        return []
    lines = ["#### Record hashes (the ✓ marks above assert these verified)", ""]
    for rec in recs:
        lines.append(f"- {rec.citation_id} · {rec.source_version} · `{rec.content_sha256[:16]}`")
    lines.append("")
    return lines


def _loco_lines(contract: DecisionContract) -> List[str]:
    """Tier 3 (new with the tier redesign): the FULL per-criterion LOCO surface. Tier 2 shows
    only the single most valuable hinge; a reader auditing the whole checklist gets every
    criterion's removal consequence here, computed by the same deterministic aggregation the
    score itself uses."""
    lines = ["#### Per-criterion influence (leave-one-criterion-out)", ""]
    if not contract.score.decidable:
        lines += [
            "_The verdict is undecidable, so no criterion has a settled verdict to hinge on; "
            "see the resolution protocol above._",
            "",
        ]
        return lines
    loco = leave_one_criterion_out(contract.criteria)
    lines += [
        "| criterion | observed | points | if unassessed → range | verdict changes |",
        "|---|---|---:|---|:--:|",
    ]
    for row in contract.score.per_criterion:
        entry = loco[row.name]
        lo, hi = entry["score_range"]
        changed = "YES" if entry["verdict_changed"] else "—"
        observed = row.value if row.determinate else "_could not assess_"
        lines.append(
            f"| {row.name.replace('_', ' ')} | {observed} | {row.points} | "
            f"[{lo},{hi}] | {changed} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------------------------
# the two clinician-facing views
# ---------------------------------------------------------------------------------------------


def render_markdown(
    contract: RenderableContract,
    brief: Optional["CaseBrief"] = None,
    *,
    include_appendix: bool = False,
) -> str:
    """The clinician-facing view, for either contract type.

    ``brief`` is optional: a verified :class:`~medguard_v2.evidence.summary.CaseBrief` from
    ``summary.generate_case_brief``, rendered directly under the headline when it is both
    verified and not a fallback. A fallback brief renders nothing extra -- its own ``text`` IS
    the deterministic headline, already shown, and repeating it would add nothing. Single-view
    contracts ignore it entirely: ``generate_case_brief`` is dual-view-only (its checks read
    ``contract.score``), so no verified single-view brief can exist, and rendering an unverified
    one would break the never-unverified-prose rule.

    ``include_appendix`` (default ``False``) is the progressive-disclosure switch for TIER 3.
    Default OFF keeps the glance/audit view byte-independent of every Tier-3 element -- the
    similar-cases block, the saliency pointer, the remaining recommendations, the record hashes
    and the LOCO surface only appear when a caller (the agent, a UI "explain" affordance)
    explicitly asks. This is the MG52 rationale in one flag: never on-by-default, always
    available.
    """
    if isinstance(contract, SingleViewContract):
        return _render_single_view(contract, include_appendix=include_appendix)
    return _render_dual_view(contract, brief, include_appendix=include_appendix)


def _render_dual_view(
    contract: DecisionContract,
    brief: Optional["CaseBrief"],
    *,
    include_appendix: bool,
) -> str:
    failures = contract.verification_failures()
    if failures:
        return _withheld_lines(contract.case_id, failures)

    # ══ TIER 1 — the 3-second glance: assessment, verified brief, next step. A routine case
    # (no conflict, decidable, views agree) ends its useful read here, within eight non-blank
    # lines and with no warning furniture. ═══════════════════════════════════════════════════
    out: List[str] = [f"## Case {contract.case_id}", "", render_headline(contract), ""]

    if brief is not None and brief.verified and not brief.fallback_used:
        out += [f"_AI summary (verified against the decision record):_ {brief.text}", ""]

    out += _next_step_lines_dual(contract)

    # ══ TIER 2 — the 10-second audit. ══════════════════════════════════════════════════════
    # The conflict warning comes first in the audit tier, immediately after Tier 1, when it
    # fires. A disagreement between the two views is the most decision-relevant thing the
    # system can say, and burying it below the tables would be actively harmful.
    if contract.conflict.present:
        out += [
            "> ### ⚠ The two assessments disagree",
            "> ",
            f"> {contract.conflict.description}",
            "> ",
            "> **This is the finding to act on.** The image classifier and the dermoscopic "
            "checklist fail on different lesions, so a disagreement is not noise — it marks a "
            "case where one view is seeing something the other is not.",
            "",
        ]

    # What each view says, side by side. Never merged into one number.
    score = contract.score
    score_text = (
        str(score.score_min)
        if score.determinate
        else f"{score.score_min}–{score.score_max} (incomplete)"
    )
    out += [
        "| assessment | result |",
        "|---|---|",
        f"| Image pattern model | **{_FULL_NAME.get(contract.classifier_diagnosis, contract.classifier_diagnosis)}** "
        f"({_confidence_band(contract.classifier_confidence)}) |",
        f"| 7-point checklist | **score {score_text}** "
        f"(threshold for excision: {MELANOMA_SUSPICION_THRESHOLD}) |",
        f"| Checklist implies | **{contract.criteria_implied_action}** — {contract.criteria_implied_reason} |",
        "",
    ]
    # PHASE 2 INSERTION POINT: once perception work resumes, the classifier's conformal
    # prediction set gets added as one more row here (e.g. "| Conformal set (90%) | {NEV, BCC} |"
    # directly after the Image pattern model row above). Perception is out of bounds for this
    # change; this comment only marks where that row goes.

    # The checklist itself, so the score can be recomputed by eye.
    out += [
        "### Dermoscopic criteria",
        "",
        "| criterion | observed | points |",
        "|---|---|---:|",
    ]
    for row in score.per_criterion:
        spec = CRITERIA_BY_NAME[row.name]
        label = row.name.replace("_", " ")
        if not row.determinate:
            observed = "_could not assess_"
            points = f"0–{row.max_points}"
        else:
            observed = row.value
            points = str(row.points)
        out.append(f"| {label} ({spec.category}, max {spec.points}) | {observed} | {points} |")
    out += ["", f"**Total: {score_text}**", ""]

    if not score.decidable:
        # R3: replaces the old dead-end "re-imaging or specialist review is needed" sentence
        # with a concrete resolution protocol -- what to check, and what each answer implies.
        out += _resolution_protocol_lines(contract)
        out += [""]
    else:
        if contract.unresolved_criteria:
            unresolved = ", ".join(c.replace("_", " ") for c in contract.unresolved_criteria)
            out += [
                f"_{len(contract.unresolved_criteria)} criterion/criteria could not be assessed "
                f"({unresolved}), but the verdict holds regardless of how they would resolve._",
                "",
            ]
        # R2: the LOCO surface, on every decidable case regardless of whether anything is
        # unresolved -- a fully-assessed case can still hinge on a single criterion's reading.
        out += [_hinge_line(contract), ""]

    # Guideline evidence: ONE verbatim sentence (or the explicit absence). See _guidance_lines.
    out += _guidance_lines(contract.guideline, contract.classifier_diagnosis)

    # When the checklist is raising a condition the classifier did not commit to, the clinician
    # is deciding about THAT condition. Quoting only "no guidance for naevus" would answer a
    # question nobody asked.
    if contract.contingent_guideline is not None and contract.contingent_guideline.covered:
        out += [
            "#### If the checklist is right",
            "",
            "_The following applies to melanoma, the condition the checklist is raising. It is "
            "contingent on that reading, not on the committed diagnosis above._",
            "",
        ]
        rec = contract.contingent_guideline.recommendations[0]
        out += [f"> {rec.text.strip()}", "", _citation_line(rec), ""]
        n_more = len(contract.contingent_guideline.recommendations) - 1
        if n_more:
            out += [f"_+{n_more} more contingent recommendation(s) on request._", ""]

    # ══ TIER 3 — on demand only. Guarded on the caller asking; each block additionally guards
    # on its own data existing. The default view above is byte-independent of all of it. ═════
    if include_appendix:
        if contract.similar_cases is not None:
            out += _similar_cases_lines(contract.similar_cases)
        out += ["#### Saliency (caveated appendix)", "", _SALIENCY_CAVEAT, ""]
        out += _remaining_recommendation_lines(contract.guideline, contract.contingent_guideline)
        out += _record_hash_lines(contract.guideline, contract.contingent_guideline)
        out += _loco_lines(contract)

    out += [
        "---",
        "_Decision support only. The diagnosis is produced by a frozen image model and a "
        "published checklist; it has not been validated against clinician judgement._",
    ]
    return "\n".join(out)


def _render_single_view(contract: SingleViewContract, *, include_appendix: bool) -> str:
    failures = contract.verification_failures()
    if failures:
        return _withheld_lines(contract.case_id, failures)

    covered = contract.guideline.covered

    # ══ TIER 1 — the 3-second glance. Same shape as a routine dual-view case: headline (with
    # the single-view trust chip) and the next step, compact and free of warning furniture --
    # the degraded mode is disclosed, not dramatised. ═══════════════════════════════════════
    out: List[str] = [f"## Case {contract.case_id}", "", render_headline(contract), ""]

    out += ["### Next step", ""]
    if covered:
        citation = contract.guideline.recommendations[0].citation_id
        out += [f"Follow {citation} (quoted below). {_SINGLE_VIEW_CORRELATION}", ""]
    else:
        out += [f"No citable management — see the stated absence below. {_SINGLE_VIEW_CORRELATION}", ""]

    # ══ TIER 2 — the 10-second audit. The two-view table renders the checklist row as the
    # explicit unavailable-disclosure -- the absence occupies the exact slot the cross-check
    # would have, so its absence can never be read as "checked, nothing found". ═════════════
    out += [
        "| assessment | result |",
        "|---|---|",
        f"| Image pattern model | **{_FULL_NAME.get(contract.classifier_diagnosis, contract.classifier_diagnosis)}** "
        f"({_confidence_band(contract.classifier_confidence)}) |",
        f"| 7-point checklist | _unavailable — {contract.checklist_unavailable_reason}_ |",
        "",
    ]

    out += _guidance_lines(contract.guideline, contract.classifier_diagnosis)

    # ══ TIER 3 — on demand only. No saliency pointer (its faithfulness evidence is
    # Derm7pt-specific) and no LOCO surface (there are no criteria to leave out). ═══════════
    if include_appendix:
        if contract.similar_cases is not None:
            out += _similar_cases_lines(contract.similar_cases)
        out += _remaining_recommendation_lines(contract.guideline, None)
        out += _record_hash_lines(contract.guideline, None)

    out += [
        "---",
        "_Decision support only. This is a SINGLE-VIEW decision from a frozen image model; no "
        "dermoscopic-criteria cross-check exists for this imaging source, and the diagnosis has "
        "not been validated against clinician judgement._",
    ]
    return "\n".join(out)
