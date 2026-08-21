import math

from medguard.eval.h2h_b4_adapter import (
    RAW_COSINE_SEMANTICS,
    DenseTextCandidate,
    DynaRAGGMMATKSComponentAdapter,
    EvidenceModality,
    ImageExemplarCandidate,
    MergeConfig,
    OrderedViewCandidates,
)
from medguard.pipeline.adaptive_selector import AdaptiveSelection


class _SpySelector:
    def __init__(self, result):
        self.result = result
        self.inputs = []

    def select(self, scores):
        self.inputs.append(list(scores))
        return self.result


def _view(view_id, candidates):
    return OrderedViewCandidates(
        view_id=view_id,
        candidates=tuple(
            ImageExemplarCandidate(identifier, score)
            for identifier, score in candidates
        ),
        score_semantics=RAW_COSINE_SEMANTICS,
    )


def _dense(count=5):
    return tuple(
        DenseTextCandidate(f"text-{index}", 1.0 - index / 10.0)
        for index in range(count)
    )


def test_raw_cosines_reach_selector_and_all_ordered_views_remain_auditable():
    spy = _SpySelector(AdaptiveSelection([0, 1], [0.7, 0.8], 2, 4.0))
    result = DynaRAGGMMATKSComponentAdapter(selector=spy).run(
        (
            _view("view-left", (("ex-b", 0.90), ("ex-a", 0.20))),
            _view("view-right", (("ex-a", 0.80), ("ex-b", 0.40))),
            _view("view-close", (("ex-a", 0.60),)),
        ),
        _dense(),
    )
    # Stable ID order gives ex-a then ex-b; values are raw max cosines, not ranks/RRF.
    assert spy.inputs == [[0.80, 0.90]]
    assert result.audit.selector_input_ids == ("ex-a", "ex-b")
    assert result.audit.selector_input_raw_cosines == (0.80, 0.90)
    assert result.audit.selector_input_per_view_cosines == (
        (0.20, 0.80, 0.60),
        (0.90, 0.40, None),
    )
    assert result.audit.ordered_view_ids == (
        "view-left", "view-right", "view-close"
    )
    assert [item.exemplar_id for item in result.selected_exemplars] == [
        "ex-b", "ex-a"
    ]


def test_rank_or_rrf_semantics_are_rejected_before_selection():
    try:
        OrderedViewCandidates(
            "view-a",
            (ImageExemplarCandidate("ex-a", 0.03),),
            score_semantics="rrf_score",
        )
    except ValueError as exc:
        assert "RRF inputs are forbidden" in str(exc)
    else:
        raise AssertionError("RRF score semantics reached the GMM adapter")


def test_constant_distribution_uses_bounded_stable_raw_fallback():
    candidates = tuple((f"ex-{index}", 0.5) for index in range(8))
    spy = _SpySelector(
        AdaptiveSelection(
            list(range(8)), [1.0] * 8, 1, 0.0,
            "constant_score_distribution",
        )
    )
    result = DynaRAGGMMATKSComponentAdapter(selector=spy).run(
        (_view("view-a", candidates),), _dense()
    )
    assert result.audit.fallback_applied
    assert result.audit.selector_fallback_reason == "constant_score_distribution"
    assert [item.exemplar_id for item in result.selected_exemplars] == [
        "ex-0", "ex-1", "ex-2", "ex-3", "ex-4"
    ]
    assert len(result.merged_evidence) == 5


def test_failed_fit_falls_back_to_one_highest_raw_cosine():
    spy = _SpySelector(
        AdaptiveSelection([], [0.0, 0.0, 0.0], 0, math.inf, "all_gmm_fits_failed")
    )
    result = DynaRAGGMMATKSComponentAdapter(selector=spy).run(
        (_view("view-a", (("low", 0.1), ("best", 0.9), ("mid", 0.5))),),
        _dense(),
    )
    assert result.audit.fallback_applied
    assert [item.exemplar_id for item in result.selected_exemplars] == ["best"]


def test_balanced_image_first_merge_has_one_global_top_five_ceiling():
    spy = _SpySelector(
        AdaptiveSelection([0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6], 2, 1.0)
    )
    result = DynaRAGGMMATKSComponentAdapter(selector=spy).run(
        (_view("view-a", tuple((f"ex-{i}", 0.9 - i / 10) for i in range(4))),),
        _dense(5),
    )
    assert [(item.modality, item.evidence_id) for item in result.merged_evidence] == [
        (EvidenceModality.IMAGE_EXEMPLAR, "ex-0"),
        (EvidenceModality.TEXT_DENSE, "text-0"),
        (EvidenceModality.IMAGE_EXEMPLAR, "ex-1"),
        (EvidenceModality.TEXT_DENSE, "text-1"),
        (EvidenceModality.IMAGE_EXEMPLAR, "ex-2"),
    ]
    assert result.audit.merge_rule == (
        "balanced_round_robin_image_exemplar_first_under_shared_top_5"
    )


def test_default_selector_configuration_matches_the_b4_card():
    selector = DynaRAGGMMATKSComponentAdapter().selector
    assert selector.gamma == 0.5
    assert selector.max_components == 4
    assert selector.min_selected == 1
    assert selector.max_selected == 5
    assert selector.covariance_floor == 1e-6
    assert selector.n_init == 5
    assert selector.random_state == 0


def test_real_gmm_path_selects_the_high_similarity_mode_deterministically():
    adapter = DynaRAGGMMATKSComponentAdapter()
    views = (
        _view(
            "view-a",
            (
                ("hi-1", 0.95), ("hi-2", 0.90), ("hi-3", 0.85),
                ("lo-1", 0.20), ("lo-2", 0.15), ("lo-3", 0.10),
            ),
        ),
    )
    first = adapter.run(views, _dense())
    second = adapter.run(views, _dense())
    assert first == second
    assert {item.exemplar_id for item in first.selected_exemplars} == {
        "hi-1", "hi-2", "hi-3"
    }
    assert first.audit.selector_components >= 2
    assert not first.audit.fallback_applied


def test_empty_image_candidates_still_preserve_views_and_fill_from_text():
    result = DynaRAGGMMATKSComponentAdapter().run(
        (
            OrderedViewCandidates("v1", (), RAW_COSINE_SEMANTICS),
            OrderedViewCandidates("v2", (), RAW_COSINE_SEMANTICS),
        ),
        _dense(6),
    )
    assert result.audit.ordered_view_ids == ("v1", "v2")
    assert result.audit.selector_fallback_reason == "empty_candidate_set"
    assert len(result.merged_evidence) == 5
    assert all(
        item.modality is EvidenceModality.TEXT_DENSE
        for item in result.merged_evidence
    )


def test_zero_ordered_views_fails_closed_instead_of_becoming_text_only():
    try:
        DynaRAGGMMATKSComponentAdapter().run((), _dense())
    except ValueError as exc:
        assert "at least one ordered lesion view" in str(exc)
    else:
        raise AssertionError("zero-view B4 case silently became text-only")


def test_duplicate_view_and_nonfinite_scores_fail_closed():
    adapter = DynaRAGGMMATKSComponentAdapter()
    try:
        adapter.run((_view("v", ()), _view("v", ())), ())
    except ValueError as exc:
        assert "view identifiers must be unique" in str(exc)
    else:
        raise AssertionError("duplicate view identifiers were accepted")
    try:
        ImageExemplarCandidate("ex", float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite cosine was accepted")
    try:
        adapter.run((_view("v", (("ex", 0.5),)),), (_dense(1)[0], _dense(1)[0]))
    except ValueError as exc:
        assert "dense evidence identifiers must be unique" in str(exc)
    else:
        raise AssertionError("duplicate dense evidence consumed the shared ceiling")


def test_non_enum_merge_configuration_fails_at_construction():
    try:
        MergeConfig(first_modality="image_exemplar")
    except ValueError as exc:
        assert "EvidenceModality enum" in str(exc)
    else:
        raise AssertionError("string merge mode was accepted")


def test_merge_start_modality_is_explicitly_configurable():
    spy = _SpySelector(AdaptiveSelection([0], [1.0], 1, 0.0))
    result = DynaRAGGMMATKSComponentAdapter(
        selector=spy,
        merge_config=MergeConfig(
            final_ceiling=2, first_modality=EvidenceModality.TEXT_DENSE
        ),
    ).run((_view("v", (("ex", 0.8),)),), _dense(1))
    assert [item.modality for item in result.merged_evidence] == [
        EvidenceModality.TEXT_DENSE,
        EvidenceModality.IMAGE_EXEMPLAR,
    ]
