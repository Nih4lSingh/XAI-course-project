"""
2D Convolutional Neural Network (2D-CNN) Architecture
Replication of Sharma et al. (2024)

Paper Specification (Section 4.3):
- 3 Convolution Layers (EXPLICIT)
- Filters: 64 -> 32 -> 32 (EXPLICIT)
- Kernel Size: 3 x 3 (EXPLICIT)
- Activation: ReLU (EXPLICIT)
- Pooling: 2 x 2 Max Pooling (EXPLICIT)
- Output Layer: Dense(5, activation='softmax') (EXPLICIT)
- Optimizer: Adam(learning_rate=0.001) (EXPLICIT)
- Weight Decay: 0.0001 (EXPLICIT)
- Epochs: 20 (EXPLICIT)

Input Grid Shapes:
- NSL-KDD: (6, 6, 1)
- UNSW-NB15: (7, 7, 1)
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_cnn2d_model(
    input_shape: Tuple[int, int, int] = (6, 6, 1),
    num_classes: int = 5,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    name: str = "Sharma_2DCNN"
) -> keras.Model:
    """
    Constructs and compiles the paper-faithful 3-layer 2D-CNN.
    """
    l2_reg = regularizers.l2(weight_decay) if weight_decay > 0 else None

    inputs = keras.Input(shape=input_shape, name="grid_input")
    
    # Conv2D Layer 1: 64 filters, 3x3 kernel
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv2d_1"
    )(inputs)
    
    # Max Pooling: 2x2
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2d_1")(x)
    
    # Conv2D Layer 2: 32 filters, 3x3 kernel
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv2d_2"
    )(x)
    
    # Conv2D Layer 3: 32 filters, 3x3 kernel
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv2d_3"
    )(x)
    
    # Flatten
    x = layers.Flatten(name="flatten")(x)
    
    # Output Layer
    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
