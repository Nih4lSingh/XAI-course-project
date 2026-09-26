"""
Evaluation Metrics Module
Replication of Sharma et al. (2024)

Calculates:
- Overall Accuracy
- Precision (Macro and Weighted)
- Recall (Macro and Weighted)
- F1-score (Macro and Weighted)
- Per-class metrics
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Computes complete evaluation metrics for multi-class classification.
    """
    acc = float(accuracy_score(y_true, y_pred))
    
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class_dict = {}
    for idx, name in enumerate(class_names):
        per_class_dict[name] = {
            "precision": float(per_class_prec[idx]) if idx < len(per_class_prec) else 0.0,
            "recall": float(per_class_rec[idx]) if idx < len(per_class_rec) else 0.0,
            "f1_score": float(per_class_f1[idx]) if idx < len(per_class_f1) else 0.0
        }

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "per_class": per_class_dict
    }


def format_metrics_table(results_dict: Dict) -> pd.DataFrame:
    """Formats metrics dictionary into a clean pandas DataFrame."""
    rows = []
    for exp_id, metrics in results_dict.items():
        rows.append({
            "experiment_id": exp_id,
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"]
        })
    return pd.DataFrame(rows)
