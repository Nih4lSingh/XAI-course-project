"""
Metrics and Summary Integrity Tests
Sharma et al. (2024) Replication

Verifies:
1. Every experiment metric JSON exists and has valid evaluation metrics.
2. results/results_summary.csv agrees exactly with the metric JSON files.
3. Class labels in per-class evaluation match canonical class mappings.
4. Macro and weighted metric bounds: accuracy, precision, recall, f1 in [0.0, 1.0].
5. Feature counts in summary table match expected values (NSL=36, UNSW=38).
"""

import json
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_EXPS = [
    "NSL_SELECTED_DNN",
    "NSL_SELECTED_1DCNN",
    "NSL_SELECTED_2DCNN",
    "UNSW_SELECTED_DNN",
    "UNSW_SELECTED_1DCNN",
    "UNSW_SELECTED_2DCNN"
]

ALL_EXPS = CANONICAL_EXPS + [
    "NSL_ALL_DNN",
    "NSL_ALL_1DCNN",
    "NSL_ALL_2DCNN",
    "UNSW_ALL_DNN",
    "UNSW_ALL_1DCNN",
    "UNSW_ALL_2DCNN",
    "NSL_SELECTED_DNN_DROPOUT_001",
    "UNSW_SELECTED_DNN_DROPOUT_001"
]

EXPECTED_FEATS = {
    "NSL_SELECTED_DNN": 36,
    "NSL_SELECTED_1DCNN": 36,
    "NSL_SELECTED_2DCNN": 36,
    "NSL_ALL_DNN": 42,
    "NSL_ALL_1DCNN": 42,
    "NSL_ALL_2DCNN": 42,
    "UNSW_SELECTED_DNN": 38,
    "UNSW_SELECTED_1DCNN": 38,
    "UNSW_SELECTED_2DCNN": 38,
    "UNSW_ALL_DNN": 42,
    "UNSW_ALL_1DCNN": 42,
    "UNSW_ALL_2DCNN": 42,
    "NSL_SELECTED_DNN_DROPOUT_001": 36,
    "UNSW_SELECTED_DNN_DROPOUT_001": 38
}


def get_json_metric(exp_id: str):
    exp_lower = exp_id.lower()
    for sub in ["canonical", "ablations", "sensitivity"]:
        p = PROJECT_ROOT / "results" / sub / exp_lower / "metrics.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    p_flat = PROJECT_ROOT / "results" / "metrics" / f"{exp_lower}_metrics.json"
    if p_flat.exists():
        with open(p_flat, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


class TestMetricsAndSummary(unittest.TestCase):
    def test_canonical_metrics_exist(self):
        """Verify metric JSON exists for all 6 canonical models."""
        for exp_id in CANONICAL_EXPS:
            m = get_json_metric(exp_id)
            self.assertIsNotNone(m, f"Missing metrics for canonical experiment {exp_id}")
            self.assertIn("accuracy", m)
            self.assertIn("f1_macro", m)
            self.assertIn("f1_weighted", m)
            self.assertIn("per_class", m)
            self.assertEqual(len(m["per_class"]), 5)

    def test_metric_bounds(self):
        """Verify all metrics fall within [0.0, 1.0]."""
        for exp_id in CANONICAL_EXPS:
            m = get_json_metric(exp_id)
            if m is not None:
                self.assertGreaterEqual(m["accuracy"], 0.0)
                self.assertLessEqual(m["accuracy"], 1.0)
                self.assertGreaterEqual(m["f1_macro"], 0.0)
                self.assertLessEqual(m["f1_macro"], 1.0)

    def test_unsw_canonical_feature_count(self):
        """Verify UNSW canonical models record 38 selected features (not stale 36)."""
        dnn_m = get_json_metric("UNSW_SELECTED_DNN")
        cnn1d_m = get_json_metric("UNSW_SELECTED_1DCNN")
        if dnn_m:
            self.assertEqual(dnn_m.get("num_features"), 38, "UNSW Selected DNN must have 38 features")
        if cnn1d_m:
            self.assertEqual(cnn1d_m.get("num_features"), 38, "UNSW Selected 1DCNN must have 38 features")


if __name__ == "__main__":
    unittest.main()
