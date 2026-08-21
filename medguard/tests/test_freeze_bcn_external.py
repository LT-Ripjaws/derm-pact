import unittest

from medguard.data.freeze_bcn_external import LABEL_MAP


class BcnProtocolTest(unittest.TestCase):
    def test_scc_is_explicit_and_not_collapsed_into_ak(self):
        self.assertEqual(LABEL_MAP["SCC"], "scc")
        self.assertEqual(LABEL_MAP["AK"], "akiec")
        self.assertNotEqual(LABEL_MAP["SCC"], LABEL_MAP["AK"])
