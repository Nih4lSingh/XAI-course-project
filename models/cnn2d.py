"""
2D Convolutional Neural Network (2D-CNN) Architecture
Replication of Sharma et al. (2024)

Paper Specification (Fig. 5 & Section 4.3):
- Input: 2D feature grid ((6, 6, 1) for NSL-KDD, (7, 7, 1) for UNSW-NB15)
- Conv2D Layer 1: 64 filters, 3x3 kernel, ReLU activation
- MaxPool2D Layer 1: 2x2 pooling (padding='same' to preserve spatial dimensions)
- Conv2D Layer 2: 32 filters, 3x3 kernel, ReLU activation
- MaxPool2D Layer 2: 2x2 pooling (padding='same')
- Conv2D Layer 3: 32 filters, 3x3 kernel, ReLU activation
- MaxPool2D Layer 3: 2x2 pooling (padding='same')
- Flatten
- Output Layer: Dense(5, activation='softmax')

Implementation Note on Pooling Padding:
Because the paper does not specify the pooling padding boundary behavior for small
grids (6x6 and 7x7), padding='same' is used deterministically on Conv2D and MaxPooling2D
so that both 6x6 (NSL-KDD) and 7x7 (UNSW-NB15) are valid and fully executable without
silently removing any of the 3 pooling layers shown in Fig. 5.
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
    Constructs and compiles the paper-faithful 3-conv / 3-pooling 2D-CNN (Fig. 5).
    """
    l2_reg = regularizers.l2(weight_decay) if weight_decay > 0 else None

    inputs = keras.Input(shape=input_shape, name="grid_input")
    
    # Block 1: Conv2D(64, 3x3) -> MaxPool2D(2x2)
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv2d_1"
    )(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="maxpool2d_1")(x)
    
    # Block 2: Conv2D(32, 3x3) -> MaxPool2D(2x2)
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv2d_2"
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="maxpool2d_2")(x)
    
    # Block 3: Conv2D(32, 3x3) -> MaxPool2D(2x2)
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv2d_3"
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="maxpool2d_3")(x)
    
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


if __name__ == "__main__":
    m6 = build_cnn2d_model((6, 6, 1))
    m7 = build_cnn2d_model((7, 7, 1))
    print("6x6 model output:", m6.output_shape)
    print("7x7 model output:", m7.output_shape)
