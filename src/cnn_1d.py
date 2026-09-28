"""Configurable 1D-CNN for encoded feature sequences."""
from __future__ import annotations
from typing import Any
import tensorflow as tf

def reshape_for_1d(matrix):
    return matrix[..., None]

def build_cnn_1d(input_features: int, n_classes: int, config: dict[str, Any]) -> tf.keras.Model:
    if n_classes != 5:
        raise ValueError("The Sharma et al. replication requires five output classes.")
    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(input_features, 1))]
    for filters in config["filters"]:
        layers.extend([tf.keras.layers.Conv1D(filters, config["kernel_size"], padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(config["weight_decay"])), tf.keras.layers.MaxPooling1D(2)])
    layers.extend([tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dropout(config["dropout"])])
    output = tf.keras.layers.Dense(5, activation="softmax")
    model = tf.keras.Sequential(layers + [output], name="cnn_1d")
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
