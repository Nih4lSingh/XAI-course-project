"""
Training & Evaluation Pipeline Utilities
Replication of Sharma et al. (2024)

Orchestrates:
1. Model training for exact 20 epochs without undocumented early stopping.
2. Adam optimizer (lr=0.001, weight decay=0.0001).
3. Recording of loss and accuracy progression.
4. Evaluation on test set.
5. Generation of confusion matrix and training curves.
6. Checkpoint and prediction persistence.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from evaluation.confusion_matrix import plot_and_save_confusion_matrix
from evaluation.metrics import compute_all_metrics
from evaluation.timing import PerformanceTimer
from preprocessing.encoders import NSL_KDD_CLASS_MAPPING, UNSW_NB15_CLASS_MAPPING


def plot_and_save_training_curves(
    history: tf.keras.callbacks.History,
    output_dir: Path,
    experiment_id: str
):
    """Plots and saves loss and accuracy curves across 20 epochs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.history["loss"]) + 1)

    # 1. Accuracy Curve
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history.history["accuracy"], "b-o", label="Training Accuracy", markersize=4)
    if "val_accuracy" in history.history:
        plt.plot(epochs, history.history["val_accuracy"], "r--s", label="Validation Accuracy", markersize=4)
    plt.title(f"{experiment_id} - Accuracy vs Epoch", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id.lower()}_accuracy.png", dpi=300)
    plt.close()

    # 2. Loss Curve
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history.history["loss"], "b-o", label="Training Loss", markersize=4)
    if "val_loss" in history.history:
        plt.plot(epochs, history.history["val_loss"], "r--s", label="Validation Loss", markersize=4)
    plt.title(f"{experiment_id} - Loss vs Epoch", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Sparse Categorical Cross-Entropy Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_id.lower()}_loss.png", dpi=300)
    plt.close()


def train_and_evaluate_model(
    experiment_id: str,
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    dropout_rate: float = 0.0,
    num_real_features: Optional[int] = None,
    padding_zeros: int = 0,
    results_base: Optional[Path] = None,
    verbose: int = 1
) -> Dict:
    """
    Executes full training and test evaluation cycle.
    """
    if results_base is None:
        results_base = PROJECT_ROOT / "results"

    models_dir = results_base / "models"
    metrics_dir = results_base / "metrics"
    cm_dir = results_base / "confusion_matrices"
    curves_dir = results_base / "training_curves"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\n[EXPERIMENT] Starting {experiment_id}\n{'='*70}")
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape[1:]} | Epochs: {epochs} | Batch size: {batch_size}")

    # Start timer
    timer = PerformanceTimer(experiment_id)
    timer.start()

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    elapsed_ms = timer.stop()
    total_sec = float(timer.total_seconds)
    total_ms = float(elapsed_ms if elapsed_ms > 0 else total_sec * 1000.0)
    print(f"[TIMING] Completed {epochs} epochs in {total_sec:.2f} s ({total_ms:.1f} ms)")

    # Save model weights/architecture
    model_save_path = models_dir / f"{experiment_id.lower()}.keras"
    model.save(model_save_path)
    print(f"[CHECKPOINT] Saved model to {model_save_path}")

    # Evaluate on test set
    y_prob = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    input_shape = list(X_train.shape[1:])
    input_size = int(np.prod(input_shape))
    if num_real_features is None:
        num_real_features = input_size - padding_zeros

    # Compute metrics
    metrics = compute_all_metrics(y_test, y_pred, class_names=class_names)
    metrics["experiment_id"] = experiment_id
    metrics["dataset"] = "NSL-KDD" if "NSL" in experiment_id.upper() else "UNSW-NB15"
    metrics["model"] = "2DCNN" if "2DCNN" in experiment_id.upper() else ("1DCNN" if "1DCNN" in experiment_id.upper() else "DNN")
    metrics["feature_mode"] = "selected" if "SELECTED" in experiment_id.upper() else "all"
    metrics["num_real_features"] = int(num_real_features)
    metrics["num_features"] = int(num_real_features)  # Backward compatible alias
    metrics["input_size"] = int(input_size)
    metrics["input_shape"] = input_shape
    metrics["padding_zeros"] = int(padding_zeros)
    metrics["seed"] = 42
    metrics["epochs"] = int(epochs)
    metrics["batch_size"] = int(batch_size)
    metrics["learning_rate"] = float(learning_rate)
    metrics["weight_decay"] = float(weight_decay)
    metrics["dropout_rate"] = float(dropout_rate)
    metrics["training_time_seconds"] = total_sec
    metrics["training_time_ms"] = total_ms

    print(f"[RESULTS] Test Accuracy: {metrics['accuracy']:.4f} | F1 (Macro): {metrics['f1_macro']:.4f} | F1 (Weighted): {metrics['f1_weighted']:.4f}")

    # Save predictions
    np.save(metrics_dir / f"{experiment_id.lower()}_y_pred.npy", y_pred)
    np.save(metrics_dir / f"{experiment_id.lower()}_y_prob.npy", y_prob)

    # Save confusion matrix
    plot_and_save_confusion_matrix(
        y_test,
        y_pred,
        class_names=class_names,
        output_path=cm_dir / f"{experiment_id.lower()}_cm.png",
        title=f"Confusion Matrix - {experiment_id}"
    )

    # Save training curves
    plot_and_save_training_curves(history, curves_dir, experiment_id)

    # Save experiment metrics JSON
    with open(metrics_dir / f"{experiment_id.lower()}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save timer profile
    timer.save(metrics_dir / f"{experiment_id.lower()}_timing.json")

    return metrics
