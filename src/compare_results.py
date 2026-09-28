"""Copy locally generated metrics into the replication side of the comparison ledger."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MODEL_DIR = {"DNN": "dnn", "1D-CNN": "1d_cnn", "2D-CNN": "2d_cnn"}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=False)
    parser.parse_args()
    path = Path("results/comparison/paper_vs_replication.csv")
    comparison = pd.read_csv(path, keep_default_na=False)
    for model, directory in MODEL_DIR.items():
        metrics_path = Path("results") / directory / "metrics.csv"
        if not metrics_path.exists(): continue
        metrics = pd.read_csv(metrics_path)
        for index, row in comparison.iterrows():
            if row["model"] != model: continue
            dataset = "nsl" if row["dataset"] == "NSL-KDD" else "unsw"
            if "protocol" not in metrics:
                continue
            match = metrics[(metrics["dataset"] == dataset) & (metrics["metric"] == row["metric"]) & (metrics["protocol"] == "paper_multiclass")]
            if not match.empty: comparison.at[index, "replication_value"] = str(match.iloc[-1]["replication_value"])
    comparison.to_csv(path, index=False)
    print(f"Updated replication values in {path}")

if __name__ == "__main__": main()
