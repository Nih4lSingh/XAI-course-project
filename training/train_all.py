"""
Master Experiment Orchestrator
Replication of Sharma et al. (2024)

Executes the complete 12-model training matrix:
1. NSL-KDD + Selected Features: DNN, 1D-CNN, 2D-CNN (Paper replication)
2. NSL-KDD + All Features: DNN, 1D-CNN, 2D-CNN (Ablation baseline)
3. UNSW-NB15 + Selected Features: DNN, 1D-CNN, 2D-CNN (Paper replication)
4. UNSW-NB15 + All Features: DNN, 1D-CNN, 2D-CNN (Ablation baseline)

Plus Sensitivity Analysis:
- NSL-KDD DNN with Dropout = 0.01
- UNSW-NB15 DNN with Dropout = 0.01

Compiles:
- results/results_summary.csv
- results/paper_vs_reproduction.csv
- reports/tables/results_table.md
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from training.train_experiment import run_experiment


PRIMARY_12_EXPERIMENTS = [
    # 1. NSL-KDD Paper Replication (Selected Features)
    "NSL_SELECTED_DNN",
    "NSL_SELECTED_1DCNN",
    "NSL_SELECTED_2DCNN",
    
    # 2. NSL-KDD Ablation Baseline (All Features)
    "NSL_ALL_DNN",
    "NSL_ALL_1DCNN",
    "NSL_ALL_2DCNN",

    # 3. UNSW-NB15 Paper Replication (Selected Features)
    "UNSW_SELECTED_DNN",
    "UNSW_SELECTED_1DCNN",
    "UNSW_SELECTED_2DCNN",

    # 4. UNSW-NB15 Ablation Baseline (All Features)
    "UNSW_ALL_DNN",
    "UNSW_ALL_1DCNN",
    "UNSW_ALL_2DCNN"
]

SENSITIVITY_EXPERIMENTS = [
    ("NSL_SELECTED_DNN_DROPOUT_001", "NSL_SELECTED_DNN", 0.01),
    ("UNSW_SELECTED_DNN_DROPOUT_001", "UNSW_SELECTED_DNN", 0.01)
]

PAPER_REPORTED_METRICS = {
    "NSL_SELECTED_DNN": {"paper_acc": 0.993, "paper_time_ms": 142.0},
    "NSL_SELECTED_1DCNN": {"paper_acc": 0.992, "paper_time_ms": 325.0},
    "NSL_SELECTED_2DCNN": {"paper_acc": 0.994, "paper_time_ms": 340.0},
    "UNSW_SELECTED_DNN": {"paper_acc": 0.800, "paper_time_ms": 323.0},
    "UNSW_SELECTED_1DCNN": {"paper_acc": 0.800, "paper_time_ms": 442.0},
    "UNSW_SELECTED_2DCNN": {"paper_acc": 0.810, "paper_time_ms": 455.0},
}


def run_all_experiments(epochs: int = 20, batch_size: int = 64, run_sensitivity: bool = True, skip_existing: bool = True) -> pd.DataFrame:
    results_dir = PROJECT_ROOT / "results"
    reports_dir = PROJECT_ROOT / "reports" / "tables"
    metrics_dir = results_dir / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    print("\n" + "=" * 75)
    print("STARTING EXECUTION OF THE 12-MODEL MATRIX (SHARMA ET AL. 2024 REPLICATION)")
    print("=" * 75)

    for idx, exp_id in enumerate(PRIMARY_12_EXPERIMENTS, 1):
        metric_file = metrics_dir / f"{exp_id.lower()}_metrics.json"
        if skip_existing and metric_file.exists():
            print(f"\n>>> [RESUME] Skipping already trained experiment [{idx}/{len(PRIMARY_12_EXPERIMENTS)}]: {exp_id}")
            with open(metric_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            print(f"\n>>> Running Experiment [{idx}/{len(PRIMARY_12_EXPERIMENTS)}]: {exp_id}")
            metrics = run_experiment(
                experiment_id=exp_id,
                epochs=epochs,
                batch_size=batch_size,
                dropout_rate=0.0,
                verbose=1
            )
        
        parts = exp_id.split("_")
        dataset_label = "NSL-KDD" if "NSL" in parts[0] else "UNSW-NB15"
        feature_mode = "Selected" if "SELECTED" in parts[1] else "All"
        model_name = parts[2]

        record = {
            "dataset": dataset_label,
            "feature_mode": feature_mode,
            "model": model_name,
            "num_features": metrics["num_features"],
            "accuracy": round(metrics["accuracy"], 4),
            "precision_macro": round(metrics["precision_macro"], 4),
            "recall_macro": round(metrics["recall_macro"], 4),
            "f1_macro": round(metrics["f1_macro"], 4),
            "precision_weighted": round(metrics["precision_weighted"], 4),
            "recall_weighted": round(metrics["recall_weighted"], 4),
            "f1_weighted": round(metrics["f1_weighted"], 4),
            "training_time_s": round(metrics["training_time_seconds"], 2),
            "training_time_ms": round(metrics["training_time_ms"], 1),
            "experiment_id": exp_id
        }
        summary_records.append(record)

    # Optional Sensitivity Experiments
    if run_sensitivity:
        print("\n" + "=" * 75)
        print("RUNNING DROPOUT SENSITIVITY EXPERIMENTS (DROPOUT = 0.01)")
        print("=" * 75)
        for exp_label, base_exp, d_rate in SENSITIVITY_EXPERIMENTS:
            metric_file = metrics_dir / f"{exp_label.lower()}_metrics.json"
            if skip_existing and metric_file.exists():
                print(f"\n>>> [RESUME] Skipping already trained sensitivity experiment: {exp_label}")
                with open(metric_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                print(f"\n>>> Running Sensitivity Experiment: {exp_label}")
                metrics = run_experiment(
                    experiment_id=exp_label,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout_rate=d_rate,
                    verbose=1
                )
            parts = base_exp.split("_")
            dataset_label = "NSL-KDD" if "NSL" in parts[0] else "UNSW-NB15"
            record = {
                "dataset": dataset_label,
                "feature_mode": "Selected (Dropout=0.01)",
                "model": "DNN",
                "num_features": metrics["num_features"],
                "accuracy": round(metrics["accuracy"], 4),
                "precision_macro": round(metrics["precision_macro"], 4),
                "recall_macro": round(metrics["recall_macro"], 4),
                "f1_macro": round(metrics["f1_macro"], 4),
                "precision_weighted": round(metrics["precision_weighted"], 4),
                "recall_weighted": round(metrics["recall_weighted"], 4),
                "f1_weighted": round(metrics["f1_weighted"], 4),
                "training_time_s": round(metrics["training_time_seconds"], 2),
                "training_time_ms": round(metrics["training_time_ms"], 1),
                "experiment_id": exp_label
            }
            summary_records.append(record)

    # Convert to DataFrame
    df_summary = pd.DataFrame(summary_records)
    summary_csv_path = results_dir / "results_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"\n[SAVED] Master results table saved to {summary_csv_path}")

    # Build Comparison Table with Paper
    comparison_records = []
    for exp_id, paper_data in PAPER_REPORTED_METRICS.items():
        match = df_summary[df_summary["experiment_id"] == exp_id]
        if not match.empty:
            our_acc = match.iloc[0]["accuracy"]
            diff = our_acc - paper_data["paper_acc"]
            comparison_records.append({
                "Dataset": match.iloc[0]["dataset"],
                "Model": match.iloc[0]["model"],
                "Paper Accuracy": paper_data["paper_acc"],
                "Our Accuracy": our_acc,
                "Difference": round(diff, 4),
                "Paper Time (ms)": paper_data["paper_time_ms"],
                "Our Time (s)": match.iloc[0]["training_time_s"]
            })

    df_comp = pd.DataFrame(comparison_records)
    comp_csv_path = results_dir / "paper_vs_reproduction.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"[SAVED] Paper vs Reproduction table saved to {comp_csv_path}")

    # Export Markdown table
    md_table_path = reports_dir / "results_table.md"
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write("# Master Results Summary (12 Model Experiments)\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n\n# Paper Reported vs Reproduction Comparison\n\n")
        f.write(df_comp.to_markdown(index=False))
    print(f"[SAVED] Markdown summary table saved to {md_table_path}")

    return df_summary


if __name__ == "__main__":
    run_all_experiments(epochs=20, batch_size=64)
