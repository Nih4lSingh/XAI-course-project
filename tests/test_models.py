"""
Model Architecture and Output Dimension Tests
Sharma et al. (2024) Replication

Verifies exact layer configurations:
1. DNN:
   - Exactly three Dense hidden layers with 64 units and ReLU
   - Final Dense layer with 5 units and Softmax
   - Evaluated with dropout=0.0 and dropout=0.01

2. 1D-CNN:
   - Conv1D(64, kernel=3, ReLU)
   - MaxPool1D(2)
   - Conv1D(32, kernel=3, ReLU)
   - Flatten
   - Final Dense(5, Softmax)

3. 2D-CNN (Paper Fig. 5 Topology):
   - Conv2D(64, kernel=(3,3), ReLU)
   - MaxPool2D((2,2))
   - Conv2D(32, kernel=(3,3), ReLU)
   - MaxPool2D((2,2))
   - Conv2D(32, kernel=(3,3), ReLU)
   - MaxPool2D((2,2))
   - Flatten
   - Final Dense(5, Softmax)
"""

import unittest
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from models.dnn import build_dnn_model
from models.cnn1d import build_cnn1d_model
from models.cnn2d import build_cnn2d_model


class TestModelArchitectures(unittest.TestCase):
    def test_dnn_architecture_exact(self):
        """Verify DNN model has exactly three Dense(64) hidden layers and Dense(5, softmax)."""
        model = build_dnn_model(input_dim=36, num_classes=5, dropout_rate=0.0)
        self.assertEqual(model.output_shape, (None, 5))
        
        dense_layers = [l for l in model.layers if isinstance(l, layers.Dense)]
        self.assertEqual(len(dense_layers), 4, f"Expected 4 Dense layers, found {len(dense_layers)}")
        self.assertEqual(dense_layers[0].units, 64)
        self.assertEqual(dense_layers[1].units, 64)
        self.assertEqual(dense_layers[2].units, 64)
        self.assertEqual(dense_layers[3].units, 5)
        self.assertEqual(dense_layers[3].activation.__name__, "softmax")

    def test_dnn_dropout_sensitivity(self):
        """Verify DNN with dropout=0.01 maintains exact dimensions."""
        model_drop = build_dnn_model(input_dim=38, num_classes=5, dropout_rate=0.01)
        self.assertEqual(model_drop.output_shape, (None, 5))
        dropout_layers = [l for l in model_drop.layers if isinstance(l, layers.Dropout)]
        self.assertEqual(len(dropout_layers), 3, "Expected 3 dropout layers when dropout_rate > 0")

    def test_cnn1d_architecture_exact(self):
        """Verify 1D-CNN has Conv1D(64, k=3) -> MaxPool1D(2) -> Conv1D(32, k=3) -> Flatten -> Dense(5)."""
        model = build_cnn1d_model(input_dim=36, num_classes=5)
        self.assertEqual(model.output_shape, (None, 5))

        conv_layers = [l for l in model.layers if isinstance(l, layers.Conv1D)]
        pool_layers = [l for l in model.layers if isinstance(l, layers.MaxPooling1D)]
        dense_layers = [l for l in model.layers if isinstance(l, layers.Dense)]

        self.assertEqual(len(conv_layers), 2, "Expected 2 Conv1D layers")
        self.assertEqual(conv_layers[0].filters, 64)
        self.assertEqual(conv_layers[0].kernel_size, (3,))
        self.assertEqual(conv_layers[1].filters, 32)
        self.assertEqual(conv_layers[1].kernel_size, (3,))

        self.assertEqual(len(pool_layers), 1, "Expected 1 MaxPool1D layer")
        self.assertEqual(pool_layers[0].pool_size, (2,))

        self.assertEqual(len(dense_layers), 1, "Expected 1 final Dense layer")
        self.assertEqual(dense_layers[0].units, 5)

    def test_cnn2d_architecture_exact(self):
        """Verify 2D-CNN implements Fig. 5: Conv64 -> Pool -> Conv32 -> Pool -> Conv32 -> Pool -> Dense(5)."""
        for shape in [(6, 6, 1), (7, 7, 1)]:
            model = build_cnn2d_model(input_shape=shape, num_classes=5)
            self.assertEqual(model.output_shape, (None, 5))

            conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D)]
            pool_layers = [l for l in model.layers if isinstance(l, layers.MaxPooling2D)]
            dense_layers = [l for l in model.layers if isinstance(l, layers.Dense)]

            self.assertEqual(len(conv_layers), 3, f"Expected 3 Conv2D layers for shape {shape}")
            self.assertEqual(conv_layers[0].filters, 64)
            self.assertEqual(conv_layers[0].kernel_size, (3, 3))
            self.assertEqual(conv_layers[1].filters, 32)
            self.assertEqual(conv_layers[1].kernel_size, (3, 3))
            self.assertEqual(conv_layers[2].filters, 32)
            self.assertEqual(conv_layers[2].kernel_size, (3, 3))

            self.assertEqual(len(pool_layers), 3, f"Expected 3 MaxPool2D layers for shape {shape}")
            for p in pool_layers:
                self.assertEqual(p.pool_size, (2, 2))

            self.assertEqual(len(dense_layers), 1, f"Expected 1 final Dense layer for shape {shape}")
            self.assertEqual(dense_layers[0].units, 5)


if __name__ == "__main__":
    unittest.main()
