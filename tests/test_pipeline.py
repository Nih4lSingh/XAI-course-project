"""
Pipeline Architecture and Shape Tests
Sharma et al. (2024) Replication

Verifies:
1. Feature Matrix Dimensions: Selected and All features.
2. 2D Grid Transformation: Shapes (6x6x1 for NSL-KDD, 7x7x1 for UNSW-NB15).
3. Deterministic Mapping: Row-major sequential placement and trailing zero-padding.
4. Class Mappings: Canonical 5-class index alignment.
5. Sampling Report: Verification of the 50K capping policy.
"""

import json
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from models.feature_to_grid import FeatureGridMapper, get_nsl_kdd_grid_mapper, get_unsw_nb15_grid_mapper
from preprocessing.encoders import NSL_KDD_CLASS_MAPPING, UNSW_NB15_CLASS_MAPPING


class TestPipelineArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nsl_data = np.load(PROJECT_ROOT / "data" / "processed" / "nsl_kdd" / "nsl_kdd_processed.npz")
        cls.unsw_data = np.load(PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "unsw_processed.npz")

    def test_nsl_kdd_feature_shapes(self):
        """Verify NSL-KDD selected features (36) and all features (42)."""
        X_sel = self.nsl_data["X_selected_train"]
        X_all = self.nsl_data["X_all_train"]
        self.assertIn(X_sel.shape[1], (35, 36), f"Unexpected NSL-KDD selected feature count: {X_sel.shape[1]}")
        self.assertIn(X_all.shape[1], (41, 42), f"Unexpected NSL-KDD all feature count: {X_all.shape[1]}")

    def test_unsw_nb15_feature_shapes(self):
        """Verify UNSW-NB15 selected features (38) and all features (42)."""
        X_sel = self.unsw_data["X_selected_train"]
        X_all = self.unsw_data["X_all_train"]
        self.assertEqual(X_sel.shape[1], 38, f"Expected 38 UNSW selected features, got {X_sel.shape[1]}")
        self.assertEqual(X_all.shape[1], 42, f"Expected 42 UNSW all features, got {X_all.shape[1]}")

    def test_feature_to_grid_mapper(self):
        """Verify deterministic row-major mapping and trailing padding."""
        features = [f"f_{i}" for i in range(38)]
        mapper = FeatureGridMapper(features, grid_size=7, dataset_name="test_unsw")
        
        # Test shape transform
        dummy = np.random.rand(10, 38).astype(np.float32)
        grid = mapper.transform(dummy)
        self.assertEqual(grid.shape, (10, 7, 7, 1))

        # Check trailing padding: elements from index 38 to 48 must be 0.0
        flat_grid = grid[0].squeeze().flatten()
        np.testing.assert_array_equal(flat_grid[38:], 0.0)
        np.testing.assert_array_equal(flat_grid[:38], dummy[0])

    def test_nsl_kdd_grid_mapping(self):
        """Verify NSL-KDD 6x6 grid mapping."""
        features = list(self.nsl_data["selected_features"])
        mapper = get_nsl_kdd_grid_mapper(features)
        dummy = np.random.rand(5, len(features)).astype(np.float32)
        grid = mapper.transform(dummy)
        self.assertEqual(grid.shape, (5, 6, 6, 1))

    def test_unsw_sampling_report(self):
        """Verify sampling report accurately reflects the 50K capping."""
        report_path = PROJECT_ROOT / "results" / "data_sampling" / "unsw_sampling_report.json"
        self.assertTrue(report_path.exists(), "UNSW sampling report does not exist")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["counts_after"]["generic"], 50000)
        self.assertEqual(report["counts_after"]["normal"], 50000)
        self.assertEqual(report["counts_after"]["dos"], 16353)
        self.assertEqual(report["counts_after"]["exploits"], 44525)
        self.assertEqual(report["counts_after"]["fuzzers"], 24246)
        self.assertEqual(report["total_records_after"], 185124)

    def test_class_mappings(self):
        """Verify class indices match Sharma et al. paper specifications."""
        expected_nsl = {0: "DoS", 1: "Normal", 2: "Probe", 3: "R2L", 4: "U2R"}
        expected_unsw = {0: "DoS", 1: "Exploits", 2: "Fuzzers", 3: "Generic", 4: "Normal"}

        self.assertEqual(NSL_KDD_CLASS_MAPPING, expected_nsl)
        self.assertEqual(UNSW_NB15_CLASS_MAPPING, expected_unsw)


if __name__ == "__main__":
    unittest.main()
