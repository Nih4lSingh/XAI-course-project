"""2D-CNN with explicit square padding of feature vectors."""
from __future__ import annotations
from math import ceil, sqrt
from typing import Any
import numpy as np
import tensorflow as tf

def reshape_for_2d(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    side = ceil(sqrt(matrix.shape[1]))
    padded = np.pad(matrix, ((0, 0), (0, side * side - matrix.shape[1])))
    return padded.reshape((-1, side, side, 1)), side

def build_cnn_2d(side: int, n_classes: int, config: dict[str, Any]) -> tf.keras.Model:
    if n_classes != 5:
        raise ValueError("The Sharma et al. replication requires five output classes.")
    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(side, side, 1))]
    for filters in config["filters"]:
        layers.extend([tf.keras.layers.Conv2D(filters, config["kernel_size"], padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(config["weight_decay"])), tf.keras.layers.MaxPooling2D(2, padding="same")])
    layers.extend([tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dropout(config["dropout"])])
    output = tf.keras.layers.Dense(5, activation="softmax")
    model = tf.keras.Sequential(layers + [output], name="cnn_2d")
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
