"""Configurable dense neural-network baseline."""
from __future__ import annotations
from typing import Any
import tensorflow as tf

def build_dnn(input_features: int, n_classes: int, config: dict[str, Any]) -> tf.keras.Model:
    if n_classes != 5:
        raise ValueError("The Sharma et al. replication requires five output classes.")
    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(input_features,))]
    for units in config["hidden_units"]:
        layers.extend([tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(config["weight_decay"])), tf.keras.layers.Dropout(config["dropout"])])
    output = tf.keras.layers.Dense(5, activation="softmax")
    model = tf.keras.Sequential(layers + [output], name="dnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
