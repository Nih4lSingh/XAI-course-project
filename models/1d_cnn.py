"""
1D Convolutional Neural Network architecture used in the Sharma et al. (2024)
replication pipeline.

Architecture:
Input -> Conv1D(64, kernel=3) -> MaxPool1D(2)
      -> Conv1D(32, kernel=3) -> MaxPool1D(2)
      -> Conv1D(32, kernel=3) -> MaxPool1D(2)
      -> Flatten -> Dense(5, softmax)

The 64/32/32 filter progression is an implementation assumption because
the paper does not fully specify the 1D-CNN filter configuration.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
KERNEL_SIZE = 3
POOL_SIZE = 2
ACTIVATION = "relu"
NUM_CLASSES = 5


def build_1d_cnn(input_shape):
    """
    Build the 1D-CNN used in the replication pipeline.

    Parameters
    ----------
    input_shape : tuple
        For example:
        - NSL-KDD: (36, 1)
        - UNSW-NB15: (38, 1)

    Returns
    -------
    keras.Model
        Compiled 1D-CNN model.
    """

    inputs = keras.Input(
        shape=input_shape,
        name="input"
    )

    # Conv1D block 1
    x = layers.Conv1D(
        filters=64,
        kernel_size=KERNEL_SIZE,
        activation=ACTIVATION,
        padding="same",
        name="conv1"
    )(inputs)

    x = layers.MaxPooling1D(
        pool_size=POOL_SIZE,
        name="pool1"
    )(x)

    # Conv1D block 2
    x = layers.Conv1D(
        filters=32,
        kernel_size=KERNEL_SIZE,
        activation=ACTIVATION,
        padding="same",
        name="conv2"
    )(x)

    x = layers.MaxPooling1D(
        pool_size=POOL_SIZE,
        name="pool2"
    )(x)

    # Conv1D block 3
    x = layers.Conv1D(
        filters=32,
        kernel_size=KERNEL_SIZE,
        activation=ACTIVATION,
        padding="same",
        name="conv3"
    )(x)

    x = layers.MaxPooling1D(
        pool_size=POOL_SIZE,
        name="pool3"
    )(x)

    # Flatten
    x = layers.Flatten(
        name="flatten"
    )(x)

    # Output layer
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="output"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Sharma_2024_1D_CNN"
    )

    # This matches the optimizer used in the pipeline.
    optimizer = keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    print("NSL-KDD 1D-CNN")
    nsl_model = build_1d_cnn((36, 1))
    nsl_model.summary()

    print("\nUNSW-NB15 1D-CNN")
    unsw_model = build_1d_cnn((38, 1))
    unsw_model.summary()
