import tempfile
import unittest
from pathlib import Path

from medguard.research.results_ledger import (
    ArtifactRef, ResultRecord, ResultsLedger, artifact_sha256, audit_ledger,
)


class ResultsLedgerTest(unittest.TestCase):
    def test_directory_hash_ignores_bytecode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "artifact"
            target.mkdir()
            (target / "result.json").write_text("{}\n", encoding="utf-8")
            before = artifact_sha256(root, "artifact")
            (target / "cache.pyc").write_bytes(b"noise")
            self.assertEqual(before, artifact_sha256(root, "artifact"))

    def test_failed_result_cannot_be_kept(self):
        record = ResultRecord(
            result_id="failed", date="2026-07-13", status="failed",
            evidence_level="verified", claim="failed", protocol="p",
            modality=["cxr"], dataset_split="attack_test", metrics={},
            decision="keep", paper_role="not-admissible", limitations=[],
            code_paths=["code.py"], artifacts=[ArtifactRef("a.json", "0" * 64, "result")],
        )
        self.assertIn("failed/abandoned result cannot have keep decision", record.validate())

    def test_secret_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            artifact_sha256(Path.cwd(), ".env")

    def test_repository_results_ledger_is_valid(self):
        root = Path(__file__).resolve().parents[2]
        audit = audit_ledger(root / "research" / "RESULTS_LEDGER.json", root=root)
        self.assertTrue(audit["valid"], audit["errors"])
        self.assertGreater(audit["n_records"], 0)

    def test_declared_retirement_allows_only_referenced_missing_code(self):
        record = ResultRecord(
            result_id="retired", date="2026-07-16", status="verified",
            evidence_level="verified", claim="historical", protocol="p",
            modality=["dermoscopy"], dataset_split="development", metrics={},
            decision="demote", paper_role="development-only", limitations=[],
            code_paths=["old/module.py"],
            artifacts=[ArtifactRef("artifact.json", "0" * 64, "result")],
        )
        ledger = ResultsLedger([record], code_retirement={"code_paths": ["old/module.py"]})
        self.assertFalse(ledger.validate())
        bad = ResultsLedger([record], code_retirement={"code_paths": ["unrelated.py"]})
        self.assertIn("__code_retirement__", bad.validate())


if __name__ == "__main__":
    unittest.main()
