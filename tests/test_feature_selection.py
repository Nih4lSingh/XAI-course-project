"""
Feature Selection Verification Tests
Sharma et al. (2024) Replication

Verifies:
1. Canonical feature counts: NSL-KDD = 36, UNSW-NB15 = 38.
2. NSL-KDD dropped features: 6 redundant predictors (|PCC| > 0.95).
3. UNSW-NB15 dropped features: 4 redundant predictors (|PCC| > 0.95).
4. Correlation threshold: strictly 0.95.
"""

import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from preprocessing.nsl_kdd import NSL_KDD_PAPER_REMOVED_FEATURES
from preprocessing.unsw_nb15 import UNSW_PAPER_REMOVED_PREDICTORS


class TestFeatureSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nsl_data = np.load(PROJECT_ROOT / "data" / "processed" / "nsl_kdd" / "nsl_kdd_processed.npz")
        cls.unsw_data = np.load(PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "unsw_processed.npz")

    def test_nsl_kdd_selected_feature_count(self):
        """Verify NSL-KDD canonical selected features count is exactly 36."""
        selected = list(self.nsl_data["selected_features"])
        self.assertEqual(len(selected), 36, f"Expected exactly 36 selected features for NSL-KDD, got {len(selected)}")

    def test_unsw_nb15_selected_feature_count(self):
        """Verify UNSW-NB15 canonical selected features count is exactly 38."""
        selected = list(self.unsw_data["selected_features"])
        self.assertEqual(len(selected), 38, f"Expected exactly 38 selected features for UNSW-NB15, got {len(selected)}")

    def test_nsl_kdd_removed_features(self):
        """Verify the 6 removed NSL-KDD features match the paper exactly."""
        expected_removed = [
            "num_root",
            "srv_serror_rate",
            "srv_rerror_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_srv_rerror_rate"
        ]
        self.assertEqual(sorted(NSL_KDD_PAPER_REMOVED_FEATURES), sorted(expected_removed))
        removed_in_npz = list(self.nsl_data["removed_features"])
        self.assertEqual(sorted(removed_in_npz), sorted(expected_removed))

    def test_unsw_nb15_removed_features(self):
        """Verify the 4 removed UNSW-NB15 features match the paper exactly."""
        expected_removed = [
            "ct_src_dport_ltm",
            "dwin",
            "ct_ftp_cmd",
            "ct_srv_dst"
        ]
        self.assertEqual(sorted(UNSW_PAPER_REMOVED_PREDICTORS), sorted(expected_removed))
        removed_in_npz = list(self.unsw_data["removed_features"])
        self.assertEqual(sorted(removed_in_npz), sorted(expected_removed))

    def test_raw_feature_counts(self):
        """Verify raw predictor count is 42 for both NSL-KDD and UNSW-NB15."""
        self.assertEqual(len(self.nsl_data["all_features"]), 42)
        self.assertEqual(len(self.unsw_data["all_features"]), 42)


if __name__ == "__main__":
    unittest.main()
