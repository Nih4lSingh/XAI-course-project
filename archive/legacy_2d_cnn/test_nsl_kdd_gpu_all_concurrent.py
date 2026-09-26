import unittest

import nsl_kdd_gpu_all_concurrent as concurrent
import nsl_kdd_gpu_sweep as sweep


class AllConcurrentScheduleTests(unittest.TestCase):
    def test_default_job_contains_768_models(self) -> None:
        configs = sweep.experiment_matrix()
        seeds = sweep.select_random_seeds(64, 20260908)
        self.assertEqual(len(configs) * len(seeds), 768)

    def test_accumulation_steps_preserve_batch_sizes(self) -> None:
        self.assertEqual(concurrent.accumulation_steps(32), 1)
        self.assertEqual(concurrent.accumulation_steps(64), 2)
        self.assertEqual(concurrent.accumulation_steps(128), 4)
        with self.assertRaises(ValueError):
            concurrent.accumulation_steps(48)

    def test_optimizer_step_counts_match_effective_batches(self) -> None:
        total_microbatches = 2362
        expected = {1: 2362, 2: 1181, 4: 591}
        for steps, expected_count in expected.items():
            actual = sum(
                concurrent.optimizer_step_is_due(index, total_microbatches, steps)
                for index in range(total_microbatches)
            )
            self.assertEqual(actual, expected_count)

    def test_final_partial_accumulation_group_is_rescaled(self) -> None:
        total_microbatches = 2362
        self.assertEqual(
            concurrent.accumulation_group_size(2360, total_microbatches, 4), 2
        )
        self.assertEqual(
            concurrent.accumulation_group_size(2361, total_microbatches, 4), 2
        )


if __name__ == "__main__":
    unittest.main()
