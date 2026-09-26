import unittest

import pandas as pd

import compare_results_to_paper as comparison


def paper_row(**overrides):
    row = {
        "config_id": "batch032-difficulty-adamw",
        "batch_size": 32,
        "feature_mode": "difficulty",
        "optimizer": "adamw",
        "seed": 123,
    }
    row.update({metric.column: metric.target for metric in comparison.PAPER_METRICS})
    row.update(overrides)
    return row


class PaperComparisonTests(unittest.TestCase):
    def test_exact_paper_row_ranks_first(self) -> None:
        exact = paper_row(seed=1)
        farther = paper_row(seed=2, test_accuracy=0.90, u2r_recall=0.05)
        ranked = comparison.score_rows(pd.DataFrame([farther, exact]))
        self.assertEqual(int(ranked.iloc[0]["seed"]), 1)
        self.assertEqual(float(ranked.iloc[0]["paper_raw_rmse"]), 0.0)

    def test_rounding_aware_score_accepts_published_precision(self) -> None:
        close = paper_row(test_accuracy=0.9944, r2l_recall=0.514)
        ranked = comparison.score_rows(pd.DataFrame([close]))
        self.assertEqual(
            int(ranked.iloc[0]["paper_round_match_count"]),
            len(comparison.PAPER_METRICS),
        )
        self.assertEqual(float(ranked.iloc[0]["paper_rounding_aware_rmse"]), 0.0)

    def test_configuration_means_are_ranked(self) -> None:
        rows = [paper_row(seed=1), paper_row(seed=2)]
        rows.append(
            paper_row(
                config_id="batch128-zero_pad-adam_l2",
                batch_size=128,
                feature_mode="zero_pad",
                optimizer="adam_l2",
                seed=3,
                test_accuracy=0.80,
            )
        )
        ranked = comparison.aggregate_configurations(pd.DataFrame(rows))
        self.assertEqual(ranked.iloc[0]["config_id"], "batch032-difficulty-adamw")
        self.assertEqual(int(ranked.iloc[0]["seed_count"]), 2)

    def test_missing_metric_is_rejected(self) -> None:
        row = paper_row()
        del row["u2r_f1"]
        with self.assertRaisesRegex(ValueError, "u2r_f1"):
            comparison.score_rows(pd.DataFrame([row]))


if __name__ == "__main__":
    unittest.main()
