"""
Target Leakage and Data Integrity Tests
Sharma et al. (2024) Replication

Verifies:
1. Target Separation: 'label', 'attack_cat', and 'id' are strictly absent from feature arrays and lists.
2. Disjoint Splits: Train, Validation, and Test index sets have zero intersection.
3. Split Completeness: Sum of split sizes equals total records.
4. Target Domain: y contains only integers {0, 1, 2, 3, 4} with no negative or NaN values.
5. Value Range: Normalized features strictly bounded in [0.0, 1.0].
"""

import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestTargetLeakageAndIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nsl_data = np.load(PROJECT_ROOT / "data" / "processed" / "nsl_kdd" / "nsl_kdd_processed.npz")
        cls.unsw_data = np.load(PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "unsw_processed.npz")
        
        cls.nsl_train_idx = np.load(PROJECT_ROOT / "splits" / "nsl_kdd_train_indices.npy")
        cls.nsl_val_idx = np.load(PROJECT_ROOT / "splits" / "nsl_kdd_val_indices.npy")
        cls.nsl_test_idx = np.load(PROJECT_ROOT / "splits" / "nsl_kdd_test_indices.npy")
        
        cls.unsw_train_idx = np.load(PROJECT_ROOT / "splits" / "unsw_train_indices.npy")
        cls.unsw_val_idx = np.load(PROJECT_ROOT / "splits" / "unsw_val_indices.npy")
        cls.unsw_test_idx = np.load(PROJECT_ROOT / "splits" / "unsw_test_indices.npy")

    def test_nsl_kdd_no_target_in_features(self):
        """Verify ground truth target label is not present in NSL-KDD features."""
        all_features = list(self.nsl_data["all_features"])
        selected_features = list(self.nsl_data["selected_features"])

        forbidden = {"label", "target", "attack_cat", "id"}
        self.assertTrue(forbidden.isdisjoint(set(all_features)), f"Forbidden columns found in NSL all_features: {forbidden.intersection(set(all_features))}")
        self.assertTrue(forbidden.isdisjoint(set(selected_features)), f"Forbidden columns found in NSL selected_features: {forbidden.intersection(set(selected_features))}")

    def test_unsw_nb15_no_target_in_features(self):
        """Verify ground truth target label is not present in UNSW-NB15 features."""
        all_features = list(self.unsw_data["all_features"])
        selected_features = list(self.unsw_data["selected_features"])

        forbidden = {"label", "target", "attack_cat", "id", "attack_clean"}
        self.assertTrue(forbidden.isdisjoint(set(all_features)), f"Forbidden columns found in UNSW all_features: {forbidden.intersection(set(all_features))}")
        self.assertTrue(forbidden.isdisjoint(set(selected_features)), f"Forbidden columns found in UNSW selected_features: {forbidden.intersection(set(selected_features))}")

    def test_nsl_kdd_disjoint_splits(self):
        """Verify train, val, and test splits are strictly mutually exclusive."""
        train_s = set(self.nsl_train_idx)
        val_s = set(self.nsl_val_idx)
        test_s = set(self.nsl_test_idx)

        self.assertEqual(len(train_s.intersection(val_s)), 0, "Train and Val splits overlap in NSL-KDD")
        self.assertEqual(len(train_s.intersection(test_s)), 0, "Train and Test splits overlap in NSL-KDD")
        self.assertEqual(len(val_s.intersection(test_s)), 0, "Val and Test splits overlap in NSL-KDD")

        total = len(train_s) + len(val_s) + len(test_s)
        self.assertEqual(total, 125973, f"Expected 125,973 total samples, got {total}")

    def test_unsw_nb15_disjoint_splits(self):
        """Verify train, val, and test splits are strictly mutually exclusive."""
        train_s = set(self.unsw_train_idx)
        val_s = set(self.unsw_val_idx)
        test_s = set(self.unsw_test_idx)

        self.assertEqual(len(train_s.intersection(val_s)), 0, "Train and Val splits overlap in UNSW-NB15")
        self.assertEqual(len(train_s.intersection(test_s)), 0, "Train and Test splits overlap in UNSW-NB15")
        self.assertEqual(len(val_s.intersection(test_s)), 0, "Val and Test splits overlap in UNSW-NB15")

        total = len(train_s) + len(val_s) + len(test_s)
        self.assertEqual(total, 185124, f"Expected 185,124 total samples, got {total}")

    def test_target_value_domain(self):
        """Verify target arrays contain exactly valid 5-class integers [0..4] without NaNs."""
        for name, data in [("NSL-KDD", self.nsl_data), ("UNSW-NB15", self.unsw_data)]:
            for split in ["y_train", "y_val", "y_test"]:
                y = data[split]
                self.assertFalse(np.isnan(y).any(), f"NaN values detected in {name} {split}")
                unique_classes = set(np.unique(y))
                self.assertTrue(unique_classes.issubset({0, 1, 2, 3, 4}), f"Invalid class labels in {name} {split}: {unique_classes}")

    def test_feature_normalization_range(self):
        """Verify all scaled features lie within [0.0, 1.0] (allowing floating precision tolerance)."""
        eps = 1e-5
        for name, data in [("NSL-KDD", self.nsl_data), ("UNSW-NB15", self.unsw_data)]:
            for feat_key in ["X_selected_train", "X_selected_test", "X_all_train", "X_all_test"]:
                X = data[feat_key]
                self.assertGreaterEqual(float(X.min()), 0.0 - eps, f"Feature values below 0.0 in {name} {feat_key}")
                self.assertLessEqual(float(X.max()), 1.0 + eps, f"Feature values above 1.0 in {name} {feat_key}")


if __name__ == "__main__":
    unittest.main()
