"""Contract tests for the frozen, hash-versioned kNN memory bank.

The bank underpins the P2 retrieval-augmented perception arm. Three properties have to hold or the
"auditable like the guideline corpus" claim is empty:

* the content hash is deterministic and reacts to features, ids, AND labels (a relabelled bank is a
  different bank);
* a similarity-weighted vote is a proper distribution (non-negative, sums to 1) and picks the
  neighbourhood's label;
* leave-one-out exclusion actually removes a case from its own retrieval, which is what keeps a
  train-on-train query out-of-sample.

These pin the invariants the ensemble relies on, independent of any dataset.
"""
from __future__ import annotations

import unittest

import numpy as np

from medguard_v2.perception.knn_memory import FrozenKNNMemory, build_memory


def _toy() -> FrozenKNNMemory:
    # Three well-separated directions on the unit circle; two entries per direction.
    feats = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0], [-0.9, -0.1]]
    )
    ids = ["a", "b", "c", "d", "e", "f"]
    labels = {"y": ["X", "X", "Y", "Y", "Z", "Z"]}
    return FrozenKNNMemory(feats, ids, labels, name="toy")


class HashTest(unittest.TestCase):
    def test_hash_is_deterministic(self):
        self.assertEqual(_toy().bank_hash, _toy().bank_hash)

    def test_hash_reacts_to_labels(self):
        base = _toy()
        feats = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0], [-0.9, -0.1]]
        )
        relabelled = FrozenKNNMemory(
            feats, ["a", "b", "c", "d", "e", "f"],
            {"y": ["X", "X", "Y", "Y", "Z", "X"]}, name="toy",  # last label flipped
        )
        self.assertNotEqual(base.bank_hash, relabelled.bank_hash)

    def test_hash_reacts_to_features(self):
        base = _toy()
        feats = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0], [-0.9, -0.2]]
        )
        moved = FrozenKNNMemory(
            feats, ["a", "b", "c", "d", "e", "f"],
            {"y": ["X", "X", "Y", "Y", "Z", "Z"]}, name="toy",
        )
        self.assertNotEqual(base.bank_hash, moved.bank_hash)


class VoteTest(unittest.TestCase):
    def test_distribution_is_proper(self):
        bank = _toy()
        q = np.array([[1.0, 0.05], [0.0, 1.0], [-1.0, 0.05]])
        dist = bank.vote(q, k=2, label_name="y", label_order=["X", "Y", "Z"])
        self.assertTrue(np.all(dist >= 0.0))
        np.testing.assert_allclose(dist.sum(axis=1), 1.0, atol=1e-9)

    def test_vote_follows_neighbourhood(self):
        bank = _toy()
        q = np.array([[1.0, 0.05], [0.0, 1.0], [-1.0, 0.05]])
        dist = bank.vote(q, k=2, label_name="y", label_order=["X", "Y", "Z"])
        picked = [["X", "Y", "Z"][int(i)] for i in dist.argmax(axis=1)]
        self.assertEqual(picked, ["X", "Y", "Z"])

    def test_labels_outside_order_are_ignored_gracefully(self):
        bank = _toy()
        # label_order omits Z; a query near the Z cluster should fall back to a uniform prior.
        dist = bank.vote(np.array([[-1.0, 0.0]]), k=2, label_name="y", label_order=["X", "Y"])
        np.testing.assert_allclose(dist.sum(axis=1), 1.0, atol=1e-9)


class ExcludeTest(unittest.TestCase):
    def test_leave_one_out_excludes_self(self):
        bank = _toy()
        # Query is exactly entry 'a'. With k=1 and no exclusion it retrieves itself (X). Excluding
        # 'a' forces the next-nearest ('b', also X here) -- but crucially not 'a' itself.
        q = np.array([[1.0, 0.0]])
        idx_incl, _ = bank.retrieve(q, k=1)
        idx_excl, _ = bank.retrieve(q, k=1, exclude_ids=["a"])
        self.assertEqual(bank._case_ids[int(idx_incl[0, 0])], "a")
        self.assertNotEqual(bank._case_ids[int(idx_excl[0, 0])], "a")


class ConstructionTest(unittest.TestCase):
    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            FrozenKNNMemory(np.eye(3), ["a", "a", "b"], {"y": ["1", "2", "3"]}, name="dup")

    def test_rejects_label_length_mismatch(self):
        with self.assertRaises(ValueError):
            FrozenKNNMemory(np.eye(3), ["a", "b", "c"], {"y": ["1", "2"]}, name="len")

    def test_rejects_query_dim_mismatch(self):
        bank = _toy()  # 2-D bank
        with self.assertRaises(ValueError):
            bank.retrieve(np.array([[1.0, 0.0, 0.0]]), k=1)

    def test_build_memory_orders_rows_by_case_list(self):
        class C:
            def __init__(self, n):
                self.case_num = n

        cases = [C(10), C(20), C(30)]
        feats = {"10": np.array([1.0, 0.0]), "20": np.array([0.0, 1.0]), "30": np.array([-1.0, 0.0])}
        bank = build_memory(feats, cases, {"y": ["a", "b", "c"]}, name="ordered")
        self.assertEqual(len(bank), 3)
        # Nearest to [0,1] is case 20 -> label b.
        dist = bank.vote(np.array([[0.0, 1.0]]), k=1, label_name="y", label_order=["a", "b", "c"])
        self.assertEqual(["a", "b", "c"][int(dist.argmax())], "b")


if __name__ == "__main__":
    unittest.main()
