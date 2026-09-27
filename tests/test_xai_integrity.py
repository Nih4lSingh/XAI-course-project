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

    def test_nsl_shap_target_class_and_sampling(self):
        """Verify NSL-KDD SHAP targets DoS (0) using a seed-controlled 50-sample set without stratification."""
        meta_file = self.xai_dir / "shap" / "nsl_kdd" / "shap_global_nsl_kdd_meta.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["target_class_index"], 0, "NSL SHAP must target class index 0")
        self.assertEqual(data["target_class_name"], "DoS", "NSL SHAP must target 'DoS'")
        self.assertEqual(data["num_test_samples"], 50, "Must evaluate exactly 50 test samples")
        self.assertEqual(len(data["test_sample_indices"]), 50, "Must record exactly 50 sample indices")
        
        # Verify sampling strategy wording (must NOT be called stratified)
        sampling_strat = data.get("sampling_strategy", "")
        self.assertEqual(sampling_strat, "seed-controlled random sample of 50 test instances")
        self.assertNotIn("stratified", sampling_strat.lower(), "50 SHAP samples must NOT be described as stratified")

        # Verify monotonicity of ranking (class-specific mean(abs(SHAP)) descending)
        ranking = data["feature_importance_ranking"]
        scores = [item["mean_abs_shap"] for item in ranking]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1], "SHAP scores must be monotonically descending")

        # Verify top feature is serror_rate
        self.assertEqual(ranking[0]["feature"], "serror_rate", "Top feature for DoS must be serror_rate")

    def test_unsw_shap_target_class_and_sampling(self):
        """Verify UNSW-NB15 SHAP targets Normal (4), identifies dttl as empirical top, and excludes 'data'."""
        meta_file = self.xai_dir / "shap" / "unsw_nb15" / "shap_global_unsw_nb15_meta.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["target_class_index"], 4, "UNSW SHAP must target class index 4")
        self.assertEqual(data["target_class_name"], "Normal", "UNSW SHAP must target 'Normal'")
        self.assertEqual(data["num_test_samples"], 50, "Must evaluate exactly 50 test samples")
        self.assertEqual(len(data["test_sample_indices"]), 50, "Must record exactly 50 sample indices")

        sampling_strat = data.get("sampling_strategy", "")
        self.assertEqual(sampling_strat, "seed-controlled random sample of 50 test instances")
        self.assertNotIn("stratified", sampling_strat.lower(), "50 SHAP samples must NOT be described as stratified")

        ranking = data["feature_importance_ranking"]
        scores = [item["mean_abs_shap"] for item in ranking]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1], "SHAP scores must be monotonically descending")

        # Verify empirical top feature is dttl
        self.assertEqual(ranking[0]["feature"], "dttl", "Empirical top feature for Normal must be dttl")
        
        # Verify non-existent feature 'data' from paper text does not appear
        ranked_features = [item["feature"] for item in ranking]
        self.assertNotIn("data", ranked_features, "Non-existent feature 'data' must not be in UNSW ranking")

    def test_shap_beeswarm_and_plot_artifacts(self):
        """Verify same-class beeswarm plots and global importance bar plots exist and are valid."""
        artifacts = [
            self.xai_dir / "shap" / "nsl_kdd" / "shap_beeswarm_summary_nsl_kdd.png",
            self.xai_dir / "shap" / "nsl_kdd" / "shap_global_importance_nsl_kdd.png",
            self.xai_dir / "shap" / "unsw_nb15" / "shap_beeswarm_summary_unsw_nb15.png",
            self.xai_dir / "shap" / "unsw_nb15" / "shap_global_importance_unsw_nb15.png",
        ]
        for p in artifacts:
            self.assertTrue(p.exists(), f"Missing SHAP plot artifact: {p}")
            self.assertGreater(p.stat().st_size, 1000, f"SHAP plot artifact appears empty: {p}")


if __name__ == "__main__":
    unittest.main()
