"""Neural network architectures and training helpers for the replication."""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def split_indices(y: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    train_val, test = train_test_split(
        all_idx, test_size=0.25, random_state=seed, stratify=y
    )
    train, val = train_test_split(
        train_val, test_size=0.20, random_state=seed, stratify=y[train_val]
    )
    return train, val, test


def _optimizer() -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-4)


def build_model(name: str, n_features: int, n_classes: int) -> tf.keras.Model:
    name = name.lower()
    if name == "dnn":
        inputs = tf.keras.Input(shape=(n_features,))
        x = inputs
        for _ in range(3):
            x = tf.keras.layers.Dense(64, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.01)(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    elif name == "cnn1d":
        inputs = tf.keras.Input(shape=(n_features, 1))
        x = inputs
        for filters in (64, 32, 32):
            x = tf.keras.layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPooling1D(2, padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    elif name == "cnn2d":
        side = int(np.ceil(np.sqrt(n_features)))
        inputs = tf.keras.Input(shape=(side, side, 1))
        x = inputs
        for filters in (64, 32, 32):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D(2, padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    else:
        raise ValueError(f"Unknown model name: {name}")
    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=_optimizer(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def reshape_inputs(name: str, X: np.ndarray) -> np.ndarray:
    if name == "dnn":
        return X
    if name == "cnn1d":
        return X[..., np.newaxis]
    if name == "cnn2d":
        side = int(np.ceil(np.sqrt(X.shape[1])))
        padded = np.pad(X, ((0, 0), (0, side * side - X.shape[1])))
        return padded.reshape((-1, side, side, 1))
    raise ValueError(f"Unknown model name: {name}")


def train_and_score(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[tf.keras.Model, dict[str, Any], np.ndarray]:
    seed_everything(SEED)
    model = build_model(name, X.shape[1], len(class_names))
    x_train = reshape_inputs(name, X[train_idx])
    x_val = reshape_inputs(name, X[val_idx])
    x_test = reshape_inputs(name, X[test_idx])
    started = time.perf_counter()
    history = model.fit(
        x_train, y[train_idx], validation_data=(x_val, y[val_idx]),
        epochs=20, batch_size=32, shuffle=True, verbose=2,
    )
    train_seconds = time.perf_counter() - started
    probabilities = model.predict(x_test, batch_size=1024, verbose=0)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "model": name,
        "test_accuracy": float(accuracy_score(y[test_idx], predictions)),
        "macro_f1": float(f1_score(y[test_idx], predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y[test_idx], predictions, average="weighted", zero_division=0)),
        "train_seconds": float(train_seconds),
        "test_rows": int(len(test_idx)),
        "classification_report": classification_report(
            y[test_idx], predictions, labels=np.arange(len(class_names)),
            target_names=class_names, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y[test_idx], predictions, labels=np.arange(len(class_names))
        ).tolist(),
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }
    return model, metrics, probabilities
