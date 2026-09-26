from pathlib import Path
import unittest

import numpy as np

import nsl_kdd_2dcnn as experiment


DATA_DIR = Path("Datasets/NSL-KDD")


class NslKddPipelineTests(unittest.TestCase):
    def test_all_local_attack_labels_are_mapped(self) -> None:
        for filename in ("KDDTrain+.txt", "KDDTest+.txt"):
            frame = experiment.read_nsl_kdd(DATA_DIR / filename)
            labels = experiment.map_attack_labels(frame["label"])
            self.assertTrue(set(np.unique(labels)).issubset(set(range(5))))

    def test_paper_protocol_shapes_and_range(self) -> None:
        data = experiment.prepare_paper_protocol(DATA_DIR / "KDDTrain+.txt", 42)
        self.assertEqual(data.x_train.shape, (75584, 6, 6, 1))
        self.assertEqual(data.x_val.shape, (18895, 6, 6, 1))
        self.assertEqual(data.x_test.shape, (31494, 6, 6, 1))
        self.assertGreaterEqual(float(data.x_train.min()), 0.0)
        self.assertLessEqual(float(data.x_train.max()), 1.0)
        self.assertTrue(data.metadata["difficulty_retained"])

    def test_official_protocol_uses_untouched_test_file(self) -> None:
        data = experiment.prepare_official_protocol(
            DATA_DIR / "KDDTrain+.txt", DATA_DIR / "KDDTest+.txt", 42
        )
        self.assertEqual(data.x_test.shape, (22544, 6, 6, 1))
        self.assertEqual(len(data.metadata["feature_names"]), 36)
        self.assertEqual(data.metadata["feature_names"][-1], "__zero_padding__")
        self.assertTrue(np.all(data.x_test[:, -1, -1, 0] == 0.0))
        self.assertFalse(data.metadata["difficulty_retained"])

    def test_confusion_matrix_and_report(self) -> None:
        truth = np.array([0, 0, 1, 2, 3, 4], dtype=np.int64)
        predicted = np.array([0, 1, 1, 2, 4, 4], dtype=np.int64)
        matrix = experiment.confusion_matrix(truth, predicted)
        report = experiment.classification_report(matrix)
        self.assertEqual(int(matrix.sum()), len(truth))
        self.assertAlmostEqual(
            float(report.loc[report["class"] == "DoS", "recall"].iloc[0]), 0.5
        )


if __name__ == "__main__":
    unittest.main()
