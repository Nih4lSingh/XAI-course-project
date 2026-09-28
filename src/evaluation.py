"""Metrics, timing, confusion matrices, and training curves."""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

def predict_labels(model: Any, matrix: np.ndarray) -> tuple[np.ndarray, float]:
    start = perf_counter()
    scores = model.predict(matrix, verbose=0)
    elapsed = perf_counter() - start
    labels = scores.argmax(axis=1)
    return labels, elapsed

def evaluate_model(model: Any, matrix: np.ndarray, targets: np.ndarray, class_names: list[str], train_seconds: float, feature_count: int) -> tuple[dict[str, float], np.ndarray]:
    predictions, inference_seconds = predict_labels(model, matrix)
    loss, _ = model.evaluate(matrix, targets, verbose=0)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average="weighted", zero_division=0)
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=range(len(class_names)), zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(targets, predictions), "loss": loss, "precision_weighted": precision,
        "recall_weighted": recall, "f1_weighted": f1, "train_seconds": train_seconds,
        "inference_seconds": inference_seconds, "feature_count": feature_count,
    }
    for index, name in enumerate(class_names):
        metrics[f"precision_{name}"] = class_precision[index]
        metrics[f"recall_{name}"] = class_recall[index]
        metrics[f"f1_{name}"] = class_f1[index]
    return metrics, confusion_matrix(targets, predictions, labels=range(len(class_names)))

def save_evaluation(output_dir: Path, model_name: str, dataset_name: str, metrics: dict[str, float], matrix: np.ndarray, class_names: list[str], history: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame([{"model": model_name, "dataset": dataset_name, "metric": key, "replication_value": value, "unit": "seconds" if "seconds" in key else "proportion", "run_status": "generated", "protocol": "paper_multiclass"} for key, value in metrics.items()])
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        if "protocol" not in existing:
            existing["protocol"] = "nonpaper_legacy"
        existing = existing[~((existing["model"] == model_name) & (existing["dataset"] == dataset_name))]
        new_rows = pd.concat([existing, new_rows], ignore_index=True)
    new_rows.to_csv(metrics_path, index=False)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{model_name} confusion matrix - {dataset_name}")
    plt.tight_layout(); plt.savefig(output_dir / f"confusion_matrix_{dataset_name}.png", dpi=160); plt.close()
    plt.figure(figsize=(7, 4))
    for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
        if key in history.history: plt.plot(history.history[key], label=key)
    plt.xlabel("Epoch"); plt.ylabel("Value"); plt.title(f"{model_name} training curves - {dataset_name}"); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / f"training_curves_{dataset_name}.png", dpi=160); plt.close()
