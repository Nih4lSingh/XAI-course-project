"""
Results Table Generator
Generates canonical summary tables and paper comparison tables directly from machine-readable JSON metrics.
"""

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRIMARY_12_EXPERIMENTS = [
    # Canonical Paper Replication (Selected Features)
    ("NSL-KDD", "Selected", "DNN", 36, "NSL_SELECTED_DNN"),
    ("NSL-KDD", "Selected", "1DCNN", 36, "NSL_SELECTED_1DCNN"),
    ("NSL-KDD", "Selected", "2DCNN", 36, "NSL_SELECTED_2DCNN"),
    ("UNSW-NB15", "Selected", "DNN", 38, "UNSW_SELECTED_DNN"),
    ("UNSW-NB15", "Selected", "1DCNN", 38, "UNSW_SELECTED_1DCNN"),
    ("UNSW-NB15", "Selected", "2DCNN", 38, "UNSW_SELECTED_2DCNN"),
    
    # Ablation Baseline (All Features)
    ("NSL-KDD", "All", "DNN", 42, "NSL_ALL_DNN"),
    ("NSL-KDD", "All", "1DCNN", 42, "NSL_ALL_1DCNN"),
    ("NSL-KDD", "All", "2DCNN", 42, "NSL_ALL_2DCNN"),
    ("UNSW-NB15", "All", "DNN", 42, "UNSW_ALL_DNN"),
    ("UNSW-NB15", "All", "1DCNN", 42, "UNSW_ALL_1DCNN"),
    ("UNSW-NB15", "All", "2DCNN", 42, "UNSW_ALL_2DCNN"),
]

SENSITIVITY_EXPERIMENTS = [
    ("NSL-KDD", "Selected (Dropout=0.01)", "DNN", 36, "NSL_SELECTED_DNN_DROPOUT_001"),
    ("UNSW-NB15", "Selected (Dropout=0.01)", "DNN", 38, "UNSW_SELECTED_DNN_DROPOUT_001"),
]

PAPER_REPORTED_METRICS = {
    "NSL_SELECTED_DNN": {"paper_acc": 0.9930, "paper_time_ms": 142.0},
    "NSL_SELECTED_1DCNN": {"paper_acc": 0.9920, "paper_time_ms": 325.0},
    "NSL_SELECTED_2DCNN": {"paper_acc": 0.9940, "paper_time_ms": 340.0},
    "UNSW_SELECTED_DNN": {"paper_acc": 0.8000, "paper_time_ms": 323.0},
    "UNSW_SELECTED_1DCNN": {"paper_acc": 0.8000, "paper_time_ms": 442.0},
    "UNSW_SELECTED_2DCNN": {"paper_acc": 0.8100, "paper_time_ms": 455.0},
}


def find_metric_file(exp_id: str) -> Path:
    exp_lower = exp_id.lower()
    # Check canonical first
    can_path = PROJECT_ROOT / "results" / "canonical" / exp_lower / "metrics.json"
    if can_path.exists():
        return can_path
    # Check ablations
    abl_path = PROJECT_ROOT / "results" / "ablations" / exp_lower / "metrics.json"
    if abl_path.exists():
        return abl_path
    # Check sensitivity
    sens_path = PROJECT_ROOT / "results" / "sensitivity" / exp_lower / "metrics.json"
    if sens_path.exists():
        return sens_path
    # Check flat metrics dir
    flat_path = PROJECT_ROOT / "results" / "metrics" / f"{exp_lower}_metrics.json"
    if flat_path.exists():
        return flat_path
    raise FileNotFoundError(f"Metric file not found for {exp_id}")


def generate_tables():
    results_dir = PROJECT_ROOT / "results"
    reports_dir = PROJECT_ROOT / "reports" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []
    
    # Process all experiments
    all_exps = PRIMARY_12_EXPERIMENTS + SENSITIVITY_EXPERIMENTS
    for dataset, mode, model, expected_feats, exp_id in all_exps:
        metric_file = find_metric_file(exp_id)
        with open(metric_file, "r", encoding="utf-8") as f:
            m = json.load(f)

        record = {
            "dataset": dataset,
            "feature_mode": mode,
            "model": model,
            "num_features": expected_feats,
            "accuracy": round(float(m["accuracy"]), 4),
            "precision_macro": round(float(m["precision_macro"]), 4),
            "recall_macro": round(float(m["recall_macro"]), 4),
            "f1_macro": round(float(m["f1_macro"]), 4),
            "precision_weighted": round(float(m["precision_weighted"]), 4),
            "recall_weighted": round(float(m["recall_weighted"]), 4),
            "f1_weighted": round(float(m["f1_weighted"]), 4),
            "training_time_s": round(float(m.get("training_time_seconds", 0.0)), 2),
            "training_time_ms": round(float(m.get("training_time_ms", 0.0)), 1),
            "experiment_id": exp_id
        }
        summary_records.append(record)

    df_summary = pd.DataFrame(summary_records)
    summary_csv = results_dir / "results_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"[SAVED] {summary_csv}")

    # Paper comparison table
    comp_records = []
    for exp_id, pdata in PAPER_REPORTED_METRICS.items():
        row = df_summary[df_summary["experiment_id"] == exp_id].iloc[0]
        our_acc = row["accuracy"]
        paper_acc = pdata["paper_acc"]
        diff = round(our_acc - paper_acc, 4)
        comp_records.append({
            "Dataset": row["dataset"],
            "Model": row["model"],
            "Paper Accuracy": paper_acc,
            "Our Accuracy": our_acc,
            "Difference": f"{diff:+.4f}",
            "Paper Training Time (ms)": pdata["paper_time_ms"],
            "Our Total Training Time (s)": row["training_time_s"],
            "Runtime Note": "The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons."
        })

    df_comp = pd.DataFrame(comp_records)
    comp_csv = results_dir / "paper_vs_reproduction.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(f"[SAVED] {comp_csv}")

    def df_to_markdown(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        lines = [
            "| " + " | ".join(str(h) for h in headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join(lines)

    # Markdown export
    md_file = reports_dir / "results_table.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Master Results Summary (12 Experiments + Sensitivity)\n\n")
        f.write(df_to_markdown(df_summary))
        f.write("\n\n---\n\n")
        f.write("# Paper Reported vs Reproduction Comparison (Canonical 6 Models)\n\n")
        f.write(df_to_markdown(df_comp))
        f.write("\n\n*Note: Differences are reported as (Our Accuracy - Paper Accuracy). Paper training times (142/325/340 ms for NSL-KDD; 323/442/455 ms for UNSW-NB15) are labeled by the authors as training time. Our reproduction measures total 20-epoch wall-clock training time in seconds. The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons.*\n")
    print(f"[SAVED] {md_file}")

    return df_summary, df_comp


if __name__ == "__main__":
    generate_tables()
