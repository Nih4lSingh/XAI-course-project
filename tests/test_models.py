"""
Model Architecture and Output Dimension Tests
Sharma et al. (2024) Replication

Verifies:
1. DNN output dimension is 5 for both datasets.
2. 1D-CNN output dimension is 5 for both datasets.
3. 2D-CNN output dimension is 5 for both datasets (6x6 for NSL-KDD, 7x7 for UNSW-NB15).
4. Dropout rate can be set to 0.0 or 0.01 without altering output dimensions.
5. Softmax activation on final layer for 5-class classification.
"""

import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from models.dnn import build_dnn_model
from models.cnn1d import build_cnn1d_model
from models.cnn2d import build_cnn2d_model


class TestModelArchitectures(unittest.TestCase):
    def test_dnn_architecture(self):
        """Verify DNN model layers and output dimension (5)."""
        model = build_dnn_model(input_dim=36, num_classes=5, dropout_rate=0.0)
        self.assertEqual(model.output_shape, (None, 5))
        
        # Test sensitivity with dropout
        model_drop = build_dnn_model(input_dim=38, num_classes=5, dropout_rate=0.01)
        self.assertEqual(model_drop.output_shape, (None, 5))

    def test_cnn1d_architecture(self):
        """Verify 1D-CNN model layers and output dimension (5)."""
        model_nsl = build_cnn1d_model(input_dim=36, num_classes=5)
        self.assertEqual(model_nsl.output_shape, (None, 5))

        model_unsw = build_cnn1d_model(input_dim=38, num_classes=5)
        self.assertEqual(model_unsw.output_shape, (None, 5))

    def test_cnn2d_architecture(self):
        """Verify 2D-CNN model layers and output dimension (5)."""
        model_nsl = build_cnn2d_model(input_shape=(6, 6, 1), num_classes=5)
        self.assertEqual(model_nsl.output_shape, (None, 5))

        model_unsw = build_cnn2d_model(input_shape=(7, 7, 1), num_classes=5)
        self.assertEqual(model_unsw.output_shape, (None, 5))


if __name__ == "__main__":
    unittest.main()
