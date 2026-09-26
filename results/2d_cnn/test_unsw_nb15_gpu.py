import unittest

import numpy as np

import unsw_nb15_gpu_all_concurrent as concurrent
import unsw_nb15_gpu_sweep as sweep


class UnswGpuTests(unittest.TestCase):
    def test_progress_duration_format(self) -> None:
        self.assertEqual(concurrent.format_duration(65), "01:05")
        self.assertEqual(concurrent.format_duration(3661), "1:01:01")

    def test_default_job_contains_384_models(self) -> None:
        configs = sweep.experiment_matrix()
        seeds = sweep.select_random_seeds(64, 20260909)
        self.assertEqual(len(configs), 6)
        self.assertEqual(len({config.config_id for config in configs}), 6)
        self.assertEqual(len(configs) * len(seeds), 384)

    def test_accumulation_preserves_effective_batch_sizes(self) -> None:
        self.assertEqual(concurrent.accumulation_steps(32), 1)
        self.assertEqual(concurrent.accumulation_steps(64), 2)
        self.assertEqual(concurrent.accumulation_steps(128), 4)

    def test_vectorized_model_accepts_seven_by_seven_inputs(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        model = sweep.VectorizedCNN.create([11, 22], "cpu")
        values = torch.zeros(2, 3, 1, 7, 7)
        logits = model(values)
        self.assertEqual(tuple(logits.shape), (2, 3, 5))
        self.assertFalse(torch.equal(logits[0], logits[1]))

    def test_unsw_metric_names_and_values(self) -> None:
        confusion = np.stack([np.eye(5, dtype=np.int64) * 3])
        metrics = sweep.metrics_from_confusions(confusion)
        self.assertIn("exploits_f1", metrics)
        self.assertIn("generic_recall", metrics)
        np.testing.assert_allclose(metrics["macro_f1"], [1.0])


if __name__ == "__main__":
    unittest.main()
