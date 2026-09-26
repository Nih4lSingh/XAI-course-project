"""
Confusion Matrix Visualization Module
Replication of Sharma et al. (2024)

Plots normalized and raw confusion matrices adhering to the paper's canonical class order.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = False
):
    """
    Plots and saves confusion matrix plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        normalize="true" if normalize else None
    )

    plt.figure(figsize=(8, 6))
    fmt = ".2%" if normalize else "d"
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontsize=13, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Actual Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[CM] Saved confusion matrix to {output_path}")
