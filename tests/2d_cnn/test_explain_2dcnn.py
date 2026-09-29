import unittest

import numpy as np

import explain_2dcnn as xai


class ExplainabilityMathTests(unittest.TestCase):
    def test_weighted_ridge_recovers_linear_relationship(self) -> None:
        design = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        target = 0.2 + 0.4 * design[:, 0] - 0.1 * design[:, 1]
        intercept, coefficients, fitted = xai.weighted_ridge(
            design, target, np.ones(4), alpha=1e-12
        )
        self.assertAlmostEqual(intercept, 0.2, places=8)
        np.testing.assert_allclose(coefficients, [0.4, -0.1], atol=1e-8)
        self.assertAlmostEqual(xai.weighted_r2(target, fitted, np.ones(4)), 1.0)

    def test_kernel_shap_recovers_additive_probability_model(self) -> None:
        def predictor(values: np.ndarray) -> np.ndarray:
            p1 = 0.1 + values @ np.asarray([0.1, 0.2, 0.3])
            return np.column_stack([1.0 - p1, p1])

        result = xai.kernel_shap_explain(
            predictor,
            np.ones(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.arange(3),
            1,
            samples=512,
            rng=np.random.default_rng(7),
        )
        self.assertAlmostEqual(result.base_value, 0.1, places=7)
        np.testing.assert_allclose(result.values, [0.1, 0.2, 0.3], atol=2e-5)
        self.assertAlmostEqual(result.model_prediction, 0.7, places=7)
        self.assertLess(abs(result.efficiency_error), 2e-5)

    def test_lime_weights_have_expected_direction(self) -> None:
        def predictor(values: np.ndarray) -> np.ndarray:
            p1 = np.clip(0.3 + 0.4 * values[:, 0] - 0.2 * values[:, 1], 0, 1)
            return np.column_stack([1.0 - p1, p1])

        rng = np.random.default_rng(11)
        background = rng.uniform(0, 1, size=(200, 2)).astype(np.float32)
        result = xai.lime_explain(
            predictor,
            np.asarray([1.0, 1.0], dtype=np.float32),
            background,
            np.arange(2),
            1,
            samples=1000,
            rng=np.random.default_rng(13),
        )
        self.assertGreater(result.coefficients[0], 0)
        self.assertLess(result.coefficients[1], 0)
        self.assertGreater(result.weighted_r2, 0.6)
        self.assertLess(result.weighted_rmse, 0.1)

    def test_padding_features_can_be_excluded(self) -> None:
        names = ["duration", "service", "__zero_padding_1__"]
        indices = np.asarray(
            [index for index, name in enumerate(names) if not name.startswith("__zero_padding")]
        )
        np.testing.assert_array_equal(indices, [0, 1])

    def test_checkpoint_config_ids_are_reconstructed(self) -> None:
        nsl = {
            "batch_size": 32,
            "feature_mode": "zero_pad",
            "optimizer": "adam_l2",
        }
        unsw = {"batch_size": 64, "optimizer": "adamw"}
        self.assertEqual(
            xai.checkpoint_config_id(nsl, "nsl_kdd"),
            "batch032-zero_pad-adam_l2",
        )
        self.assertEqual(
            xai.checkpoint_config_id(unsw, "unsw_nb15"), "batch064-adamw"
        )


if __name__ == "__main__":
    unittest.main()
