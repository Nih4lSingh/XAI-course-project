import unittest

import pandas as pd

import compare_unsw_results_to_paper as comparison


class UnswComparisonTests(unittest.TestCase):
    def paper_row(self) -> dict[str, object]:
        row = {
            "config_id": "batch032-adam_l2",
            "batch_size": 32,
            "optimizer": "adam_l2",
            "seed": 7,
        }
        row.update({metric.column: metric.target for metric in comparison.PAPER_METRICS})
        return row

    def test_exact_paper_row_scores_zero(self) -> None:
        ranked = comparison.score_rows(pd.DataFrame([self.paper_row()]))
        self.assertEqual(float(ranked.iloc[0]["paper_raw_rmse"]), 0.0)
        self.assertEqual(int(ranked.iloc[0]["paper_round_match_count"]), 17)

    def test_closer_row_ranks_first(self) -> None:
        exact = self.paper_row()
        farther = dict(exact)
        farther["seed"] = 8
        farther["test_accuracy"] = 0.95
        ranked = comparison.score_rows(pd.DataFrame([farther, exact]))
        self.assertEqual(int(ranked.iloc[0]["seed"]), 7)


if __name__ == "__main__":
    unittest.main()
