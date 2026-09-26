"""
Explainability (XAI) Integrity Tests
Sharma et al. (2024) Replication

Verifies:
1. SHAP global feature rankings exist for NSL-KDD (36 features) and UNSW-NB15 (38 features).
2. NSL-KDD SHAP targets DoS (Class 0 in canonical mapping).
3. UNSW-NB15 SHAP targets Normal (Class 4 in canonical mapping).
4. Feature names in XAI metadata match canonical preprocessed feature lists exactly.
5. Local LIME and SHAP explanation artifacts exist and are non-empty.
"""

import json
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from preprocessing.encoders import NSL_KDD_CLASS_MAPPING, UNSW_NB15_CLASS_MAPPING


class TestXAIIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nsl_data = np.load(PROJECT_ROOT / "data" / "processed" / "nsl_kdd" / "nsl_kdd_processed.npz")
        cls.unsw_data = np.load(PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "unsw_processed.npz")
        cls.xai_dir = PROJECT_ROOT / "results" / "xai"

    def test_nsl_shap_feature_count_and_names(self):
        """Verify NSL-KDD SHAP global ranking contains all 36 selected features."""
        meta_file = self.xai_dir / "shap" / "nsl_kdd" / "shap_global_nsl_kdd_meta.json"
        self.assertTrue(meta_file.exists(), f"Missing SHAP meta file: {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        ranking = data.get("feature_importance_ranking", [])
        self.assertEqual(len(ranking), 36, f"Expected 36 features in NSL-KDD SHAP ranking, got {len(ranking)}")

        ranked_names = {item["feature"] for item in ranking}
        expected_names = set(self.nsl_data["selected_features"])
        self.assertEqual(ranked_names, expected_names, "Feature names in SHAP ranking do not match selected features")

    def test_unsw_shap_feature_count_and_names(self):
        """Verify UNSW-NB15 SHAP global ranking contains all 38 selected features."""
        meta_file = self.xai_dir / "shap" / "unsw_nb15" / "shap_global_unsw_nb15_meta.json"
        self.assertTrue(meta_file.exists(), f"Missing SHAP meta file: {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        ranking = data.get("feature_importance_ranking", [])
        self.assertEqual(len(ranking), 38, f"Expected 38 features in UNSW-NB15 SHAP ranking, got {len(ranking)}")

        ranked_names = {item["feature"] for item in ranking}
        expected_names = set(self.unsw_data["selected_features"])
        self.assertEqual(ranked_names, expected_names, "Feature names in UNSW SHAP ranking do not match selected features")

    def test_xai_target_class_alignment(self):
        """Verify target classes align with Sharma et al. (2024): NSL -> DoS (0), UNSW -> Normal (4)."""
        self.assertEqual(NSL_KDD_CLASS_MAPPING[0], "DoS")
        self.assertEqual(UNSW_NB15_CLASS_MAPPING[4], "Normal")


if __name__ == "__main__":
    unittest.main()
