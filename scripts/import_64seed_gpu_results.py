"""
Import 64-seed GPU results into the canonical multi-seed results structure
and regenerate the consolidated benchmark table and markdown report.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.multiseed_sweep import (
    PAPER_BENCHMARKS,
    MultiSeedExperimentRunner,
    generate_consolidated_report,
    rounding_aware_error,
    check_rounding_match,
)

def import_results():
    output_dir = PROJECT_ROOT / "results" / "multiseed"
    runner = MultiSeedExperimentRunner(output_dir=output_dir)

    # 1. 1D-CNN (NSL-KDD and UNSW-NB15)
    csv_1d = PROJECT_ROOT / "results" / "1d_cnn_gpu" / "metrics_per_seed.csv"
    if csv_1d.is_file():
        df_1d = pd.read_csv(csv_1d)
        for ds, exp_id in [("NSL-KDD", "NSL_SELECTED_1DCNN"), ("UNSW-NB15", "UNSW_SELECTED_1DCNN")]:
            sub = df_1d[df_1d["dataset"] == ds].copy()
            if not sub.empty:
                target = PAPER_BENCHMARKS[exp_id]["paper_accuracy"]
                decimals = PAPER_BENCHMARKS[exp_id]["decimals"]
                
                rows = []
                for _, r in sub.iterrows():
                    acc = float(r["accuracy"])
                    raw_err = acc - target
                    abs_err = abs(raw_err)
                    round_err = rounding_aware_error(acc, target, decimals)
                    round_match = check_rounding_match(acc, target, decimals)
                    
                    rows.append({
                        "experiment_id": exp_id,
                        "dataset": "nsl_kdd" if "NSL" in exp_id else "unsw_nb15",
                        "model": "1DCNN",
                        "seed": int(r["seed"]),
                        "batch_size": 128,
                        "epochs": 20,
                        "split_mode": "canonical",
                        "train_time_sec": round(float(r.get("elapsed_seconds", 0.0)) / len(sub), 2),
                        "test_loss": float(r["test_loss"]),
                        "test_accuracy": acc,
                        "precision_macro": float(r["precision_macro"]),
                        "recall_macro": float(r["recall_macro"]),
                        "f1_macro": float(r["f1_macro"]),
                        "precision_weighted": float(r["precision_weighted"]),
                        "recall_weighted": float(r["recall_weighted"]),
                        "f1_weighted": float(r["f1_weighted"]),
                        "paper_accuracy": target,
                        "paper_decimals": decimals,
                        "raw_error": raw_err,
                        "absolute_error": abs_err,
                        "rounding_aware_error": round_err,
                        "matches_paper_rounding": round_match,
                    })
                
                exp_dir = output_dir / exp_id.lower()
                exp_dir.mkdir(parents=True, exist_ok=True)
                runs_df = pd.DataFrame(rows)
                runs_df.to_csv(exp_dir / "seed_runs.csv", index=False)
                
                ranked_df = runs_df.sort_values(["rounding_aware_error", "absolute_error"], ascending=[True, True]).reset_index(drop=True)
                ranked_df.to_csv(exp_dir / "ranked_seeds.csv", index=False)
                
                summary = runner.compute_statistics(runs_df, exp_id)
                with open(exp_dir / "summary_statistics.json", "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"[{exp_id}] Successfully imported {len(runs_df)} seeds. Mean: {summary['accuracy_mean']:.6f} +/- {summary['accuracy_std']:.6f}")

    # 2. DNN (NSL-KDD and UNSW-NB15)
    csv_dnn = PROJECT_ROOT / "results" / "dnn_gpu" / "metrics_per_seed.csv"
    if csv_dnn.is_file():
        df_dnn = pd.read_csv(csv_dnn)
        for ds, exp_id in [("NSL-KDD", "NSL_SELECTED_DNN"), ("UNSW-NB15", "UNSW_SELECTED_DNN")]:
            sub = df_dnn[df_dnn["dataset"] == ds].copy()
            if not sub.empty:
                target = PAPER_BENCHMARKS[exp_id]["paper_accuracy"]
                decimals = PAPER_BENCHMARKS[exp_id]["decimals"]
                
                rows = []
                for _, r in sub.iterrows():
                    acc = float(r["accuracy"])
                    raw_err = acc - target
                    abs_err = abs(raw_err)
                    round_err = rounding_aware_error(acc, target, decimals)
                    round_match = check_rounding_match(acc, target, decimals)
                    
                    rows.append({
                        "experiment_id": exp_id,
                        "dataset": "nsl_kdd" if "NSL" in exp_id else "unsw_nb15",
                        "model": "DNN",
                        "seed": int(r["seed"]),
                        "batch_size": 128,
                        "epochs": 20,
                        "split_mode": "canonical",
                        "train_time_sec": round(float(r.get("elapsed_seconds", 0.0)) / len(sub), 2),
                        "test_loss": float(r["test_loss"]),
                        "test_accuracy": acc,
                        "precision_macro": float(r["precision_macro"]),
                        "recall_macro": float(r["recall_macro"]),
                        "f1_macro": float(r["f1_macro"]),
                        "precision_weighted": float(r["precision_weighted"]),
                        "recall_weighted": float(r["recall_weighted"]),
                        "f1_weighted": float(r["f1_weighted"]),
                        "paper_accuracy": target,
                        "paper_decimals": decimals,
                        "raw_error": raw_err,
                        "absolute_error": abs_err,
                        "rounding_aware_error": round_err,
                        "matches_paper_rounding": round_match,
                    })
                
                exp_dir = output_dir / exp_id.lower()
                exp_dir.mkdir(parents=True, exist_ok=True)
                runs_df = pd.DataFrame(rows)
                runs_df.to_csv(exp_dir / "seed_runs.csv", index=False)
                
                ranked_df = runs_df.sort_values(["rounding_aware_error", "absolute_error"], ascending=[True, True]).reset_index(drop=True)
                ranked_df.to_csv(exp_dir / "ranked_seeds.csv", index=False)
                
                summary = runner.compute_statistics(runs_df, exp_id)
                with open(exp_dir / "summary_statistics.json", "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"[{exp_id}] Successfully imported {len(runs_df)} seeds. Mean: {summary['accuracy_mean']:.6f} +/- {summary['accuracy_std']:.6f}")

    # 3. Consolidate all 6 models
    summaries = {}
    for exp_id in ["NSL_SELECTED_DNN", "NSL_SELECTED_1DCNN", "NSL_SELECTED_2DCNN", "UNSW_SELECTED_DNN", "UNSW_SELECTED_1DCNN", "UNSW_SELECTED_2DCNN"]:
        exp_dir = output_dir / exp_id.lower()
        json_path = exp_dir / "summary_statistics.json"
        if json_path.is_file():
            with open(json_path) as f:
                summaries[exp_id] = json.load(f)

    comp_df, report_md = generate_consolidated_report(output_dir, summaries)
    print("\n=== CONSOLIDATED CROSS-MODEL COMPARISON ===")
    print(comp_df.to_string(index=False))

if __name__ == "__main__":
    import_results()
