from pathlib import Path
import unittest

import numpy as np

import unsw_nb15_2dcnn as unsw


class UnswPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = unsw.prepare_paper_features(
            Path("Datasets/UNSW-NB15"), sample_seed=20260909
        )

    def test_paper_sample_and_shape(self) -> None:
        self.assertEqual(self.data.features.shape, (185124, 1, 7, 7))
        self.assertEqual(self.data.labels.shape, (185124,))
        self.assertEqual(self.data.metadata["retained_predictor_count"], 38)
        self.assertEqual(self.data.metadata["zero_padding_count"], 11)

    def test_expected_class_counts(self) -> None:
        self.assertEqual(
            self.data.metadata["class_counts"],
            {
                "DoS": 16353,
                "Exploits": 44525,
                "Fuzzers": 24246,
                "Generic": 50000,
                "Normal": 50000,
            },
        )

    def test_values_are_normalized_and_padding_is_zero(self) -> None:
        self.assertGreaterEqual(float(self.data.features.min()), 0.0)
        self.assertLessEqual(float(self.data.features.max()), 1.0)
        flattened = self.data.features.reshape(len(self.data.features), 49)
        self.assertTrue(np.all(flattened[:, -11:] == 0.0))

    def test_stratified_split_is_disjoint_and_exhaustive(self) -> None:
        parts = unsw.stratified_partition(self.data.labels, (0.60, 0.15, 0.25), 17)
        self.assertEqual(tuple(len(part) for part in parts), (111075, 27769, 46280))
        combined = np.concatenate(parts)
        self.assertEqual(len(np.unique(combined)), len(self.data.labels))
        np.testing.assert_array_equal(np.sort(combined), np.arange(len(self.data.labels)))


if __name__ == "__main__":
    unittest.main()
