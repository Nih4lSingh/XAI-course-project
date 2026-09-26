from pathlib import Path
import unittest

import numpy as np

import nsl_kdd_gpu_sweep as sweep


class GpuSweepTests(unittest.TestCase):
    def test_default_matrix_has_twelve_configurations(self) -> None:
        configs = sweep.experiment_matrix()
        self.assertEqual(len(configs), 12)
        self.assertEqual(len({config.config_id for config in configs}), 12)

    def test_seed_selection_is_unique_and_reproducible(self) -> None:
        first = sweep.select_random_seeds(64, 20260908)
        second = sweep.select_random_seeds(64, 20260908)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 64)

    def test_both_feature_modes_produce_six_by_six_inputs(self) -> None:
        source = Path("Datasets/NSL-KDD/KDDTrain+.txt")
        for mode in sweep.DEFAULT_FEATURE_MODES:
            data = sweep.prepare_features(source, mode)
            self.assertEqual(data.features.shape, (125973, 1, 6, 6))
            self.assertGreaterEqual(float(data.features.min()), 0.0)
            self.assertLessEqual(float(data.features.max()), 1.0)
            if mode == "zero_pad":
                self.assertTrue(np.all(data.features[:, 0, -1, -1] == 0.0))

    def test_seed_splits_are_disjoint_and_exhaustive(self) -> None:
        source = Path("Datasets/NSL-KDD/KDDTrain+.txt")
        data = sweep.prepare_features(source, "zero_pad")
        splits = sweep.build_seed_splits(data.labels, [11, 22])
        self.assertEqual(splits[0].shape, (2, 75584))
        self.assertEqual(splits[1].shape, (2, 18895))
        self.assertEqual(splits[2].shape, (2, 31494))
        expected = np.arange(len(data.labels))
        for replica in range(2):
            combined = np.concatenate([part[replica] for part in splits])
            self.assertEqual(len(np.unique(combined)), len(data.labels))
            np.testing.assert_array_equal(np.sort(combined), expected)

    def test_vectorized_cpu_forward_keeps_replicas_independent(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        model = sweep.VectorizedCNN.create([11, 22], "cpu")
        values = torch.zeros(2, 3, 1, 6, 6)
        logits = model(values)
        self.assertEqual(tuple(logits.shape), (2, 3, 5))
        self.assertFalse(torch.equal(logits[0], logits[1]))


if __name__ == "__main__":
    unittest.main()
