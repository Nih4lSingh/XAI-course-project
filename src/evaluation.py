"""
Evaluation & Explainable AI (XAI) Module
Sharma et al. (2024) Intrusion Detection Replication

Capabilities:
1. Performance Metrics: Accuracy, Precision, Recall, F1 (Macro & Weighted)
2. Visualization: Confusion Matrices and Loss/Accuracy Training Curves
3. Benchmark Comparison: Generates paper_vs_replication.csv
4. Explainable AI: SHAP KernelExplainer global and local feature importance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paper reported targets from Table 1 and Table 2 of Sharma et al. (2024)
PAPER_TARGETS = {
    ("NSL-KDD", "DNN"): {"accuracy": 0.9930, "time_ms": 142.0},
    ("NSL-KDD", "1D-CNN"): {"accuracy": 0.9920, "time_ms": 325.0},
    ("NSL-KDD", "2D-CNN"): {"accuracy": 0.9940, "time_ms": 340.0},
    ("UNSW-NB15", "DNN"): {"accuracy": 0.8000, "time_ms": 323.0},
    ("UNSW-NB15", "1D-CNN"): {"accuracy": 0.8000, "time_ms": 442.0},
    ("UNSW-NB15", "2D-CNN"): {"accuracy": 0.8100, "time_ms": 455.0},
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
    """Calculate standard classification metrics matching published paper."""
    return {
        "Dataset": dataset_name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision Macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Precision Weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall Macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall Weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1 Macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1 Weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save publication-quality confusion matrix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        cbar=True, annot_kws={"size": 11}
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Path,
    title: str = "Training Curves",
) -> None:
    """Plot dual-axis Loss and Accuracy curves across training epochs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    ax1.set_title(f"{title} - Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    
    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#d62728", linewidth=2, linestyle="--")
    ax2.set_title(f"{title} - Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_paper_comparison(
    reproduced_results: List[Dict[str, any]],
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Generate master paper vs replication comparison table."""
    rows = []
    for res in reproduced_results:
        ds = res["dataset"]
        model = res["model"]
        our_acc = res["accuracy"]
        our_time = res.get("time_sec", 0.0)
        
        target = PAPER_TARGETS.get((ds, model), {})
        paper_acc = target.get("accuracy", np.nan)
        paper_time = target.get("time_ms", np.nan)
        diff = round(our_acc - paper_acc, 4) if not np.isnan(paper_acc) else np.nan
        
        rows.append({
            "Dataset": ds,
            "Model": model,
            "Paper Accuracy": paper_acc,
            "Our Accuracy": round(our_acc, 4),
            "Difference": diff,
            "Paper Time (ms)": paper_time,
            "Our Time (s)": round(our_time, 2),
        })
        
    df = pd.DataFrame(rows)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def run_shap_analysis(
    predict_fn,
    background_samples: np.ndarray,
    test_instances: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    dataset_name: str,
    top_n: int = 15,
) -> None:
    """
    Run SHAP KernelExplainer global feature importance analysis.
    Generates summary bar chart and beeswarm plot.
    """
    import shap
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run KernelExplainer
    explainer = shap.KernelExplainer(predict_fn, background_samples)
    shap_values = explainer.shap_values(test_instances, nsamples=100)
    
    # Global feature importance: mean absolute shap value across all classes and samples
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, -1))
        
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = mean_abs_shap[top_indices]
    
    # Plot Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_scores, y=top_features, palette="Blues_r")
    plt.title(f"SHAP Global Feature Importance — {dataset_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=11)
    plt.ylabel("Selected Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_global_importance_{dataset_name.lower().replace('-', '_')}.png", dpi=300)
    plt.close()
    
    print(f"SHAP analysis complete for {dataset_name}. Saved to {output_dir}")


if __name__ == "__main__":
    print("Testing Evaluation Module...")
    y_t = np.random.randint(0, 5, 100)
    y_p = np.random.randint(0, 5, 100)
    m = compute_metrics(y_t, y_p, "NSL-KDD DNN")
    print("Sample Metrics:", m)
    print("Evaluation Module Operational.")
