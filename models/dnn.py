"""
Dense Neural Network architecture used in the Sharma et al. (2024)
replication pipeline.

Architecture:
Input -> Dense(64, ReLU) -> Dropout(0.01)
      -> Dense(64, ReLU) -> Dropout(0.01)
      -> Dense(64, ReLU) -> Dropout(0.01)
      -> Dense(5, Softmax)

The paper contains an inconsistency regarding dropout (0.01 in one place
and 0 in another). The replication pipeline used dropout = 0.01.
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.01
NUM_CLASSES = 5


def build_dnn(input_dim, dropout_rate=DROPOUT_RATE):
    """
    Build the DNN used in the replication pipeline.

    Parameters
    ----------
    input_dim : int
        Number of input features.
        For example:
        - NSL-KDD: 36
        - UNSW-NB15: 38

    dropout_rate : float
        Dropout applied after each hidden Dense layer.

    Returns
    -------
    tf.keras.Sequential
        Compiled DNN model.
    """

    model = Sequential([
        Dense(
            64,
            activation="relu",
            input_shape=(input_dim,)
        ),

        Dropout(dropout_rate),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(dropout_rate),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(dropout_rate),

        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    optimizer = tf.keras.optimizers.Adam(
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
    print("NSL-KDD DNN")
    nsl_model = build_dnn(36)
    nsl_model.summary()

    print("\nUNSW-NB15 DNN")
    unsw_model = build_dnn(38)
    unsw_model.summary()
