from medguard.config import DIAGNOSIS_LABELS, MALIGNANT_LABELS
from medguard.pipeline.image_retriever import (
    summarize_neighbors, ExemplarAtlas, make_encoder, OpenClipEncoder,
)


NB = [("1", "nevus", 0.9), ("2", "nevus", 0.8), ("3", "melanoma", 0.85)]


def test_make_encoder_routes_to_pluggable_perception_heads():
    # lazy imports mean no heavy dependency is loaded by the factory
    assert isinstance(make_encoder("hf-hub:redlessone/DermLIP_ViT-B-16"), OpenClipEncoder)
    try:
        make_encoder("unsupported")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported encoder was silently accepted")


def test_vote_picks_similarity_majority_without_prior():
    r = summarize_neighbors(NB, DIAGNOSIS_LABELS, MALIGNANT_LABELS)
    assert r.exemplar_pred == "nevus"                  # 1.7 vote vs 0.85
    assert abs(r.exemplar_confidence - (1.7 / 2.55)) < 1e-6
    assert abs(r.exemplar_malignant_score - (0.85 / 2.55)) < 1e-6
    assert "nevus" in r.exemplar_context


def test_inverse_prevalence_correction_lifts_rare_malignant():
    # a nevus-heavy atlas: without correction nevus wins; with alpha=1 the rare
    # melanoma's vote is upweighted enough to flip the committed label
    prior = {"nevus": 0.66, "melanoma": 0.11}
    r = summarize_neighbors(NB, DIAGNOSIS_LABELS, MALIGNANT_LABELS,
                            prior=prior, prior_alpha=1.0)
    assert r.exemplar_pred == "melanoma"
    assert r.exemplar_malignant_score > 0.5            # malignant mass now dominates


def test_probe_outputs_override_the_coarse_vote():
    r = summarize_neighbors(NB, DIAGNOSIS_LABELS, MALIGNANT_LABELS,
                            diag_probs={"melanoma": 0.9, "nevus": 0.1},
                            malignant_prob=0.77)
    assert r.exemplar_pred == "melanoma" and abs(r.exemplar_confidence - 0.9) < 1e-9
    assert abs(r.exemplar_malignant_score - 0.77) < 1e-9


def test_empty_neighbors_is_safe():
    r = summarize_neighbors([], DIAGNOSIS_LABELS, MALIGNANT_LABELS)
    assert r.exemplar_confidence == 0.0
    assert r.exemplar_context == "(no neighbours)"


class _FakeEncoder:
    def encode_image(self, image_path):
        import numpy as np
        return np.ones(4, dtype="float32")


class _FakeIndex:
    def search(self, vec, k):
        import numpy as np
        # two nearest are indices 0,2 with high sim
        return np.array([[0.9, 0.85]]), np.array([[0, 2]])


def test_atlas_probe_wires_neighbors_to_summary():
    meta = [{"id": "a", "label": "nevus"}, {"id": "b", "label": "bcc"},
            {"id": "c", "label": "melanoma"}]
    atlas = ExemplarAtlas(index=_FakeIndex(), meta=meta, encoder=_FakeEncoder(), top_k=2)
    r = atlas.probe("img.jpg")
    labels = [lab for _id, lab, _s in r.neighbors]
    assert labels == ["nevus", "melanoma"]             # indices 0 and 2
    assert r.exemplar_pred in ("nevus", "melanoma")


def test_atlas_generator_records_keep_train_case_and_source_lineage():
    meta = [
        {"id": "exemplar:a", "case_id": "a", "cluster_id": "lesion-a",
         "label": "nevus", "split": "train", "source_version": "dermlip:frozen"},
        {"id": "exemplar:b", "case_id": "b", "cluster_id": "lesion-b",
         "label": "bcc", "split": "train", "source_version": "dermlip:frozen"},
        {"id": "exemplar:c", "case_id": "c", "cluster_id": "lesion-c",
         "label": "melanoma", "split": "train", "source_version": "dermlip:frozen"},
    ]
    atlas = ExemplarAtlas(index=_FakeIndex(), meta=meta, encoder=_FakeEncoder(), top_k=2)
    rows = atlas.search_records_by_image("img.jpg", 2)
    assert [row["case_id"] for row in rows] == ["a", "c"]
    assert {row["authority_class"] for row in rows} == {"image_exemplar"}
    assert all(row["split"] == "train" for row in rows)
    assert rows[0]["document_id"] == "exemplar:a"
    assert rows[0]["citation_id"] == "HAM10000:a"
    assert rows[0]["provenance_uri"] == "dataset://HAM10000/a"


def test_multiview_atlas_fuses_every_view_deterministically():
    meta = [
        {"id": "exemplar:a", "case_id": "a", "label": "nevus", "split": "train"},
        {"id": "exemplar:b", "case_id": "b", "label": "bcc", "split": "train"},
        {"id": "exemplar:c", "case_id": "c", "label": "melanoma", "split": "train"},
    ]
    atlas = ExemplarAtlas(index=_FakeIndex(), meta=meta, encoder=_FakeEncoder(), top_k=2)
    rows = atlas.search_records_by_images(["view-a.jpg", "view-b.jpg"], 2)
    assert [row["document_id"] for row in rows] == ["exemplar:a", "exemplar:c"]
    assert all(row["query_view_count"] == 2 for row in rows)
    assert all(len(row["per_view_ranks"]) == 2 for row in rows)
