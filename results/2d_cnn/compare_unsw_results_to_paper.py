"""Rank UNSW-NB15 runs by similarity to the paper's 2D-CNN results."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PaperMetric:
    column: str
    target: float
    decimals: int
    label: str


PAPER_METRICS = (
    PaperMetric("test_accuracy", 0.81, 2, "Accuracy"),
    PaperMetric("test_loss", 0.40, 2, "Loss"),
    PaperMetric("dos_precision", 0.57, 2, "DoS precision"),
    PaperMetric("dos_recall", 0.04, 2, "DoS recall"),
    PaperMetric("dos_f1", 0.07, 2, "DoS F1"),
    PaperMetric("exploits_precision", 0.69, 2, "Exploits precision"),
    PaperMetric("exploits_recall", 0.91, 2, "Exploits recall"),
    PaperMetric("exploits_f1", 0.78, 2, "Exploits F1"),
    PaperMetric("fuzzers_precision", 0.60, 2, "Fuzzers precision"),
    PaperMetric("fuzzers_recall", 0.80, 2, "Fuzzers recall"),
    PaperMetric("fuzzers_f1", 0.69, 2, "Fuzzers F1"),
    PaperMetric("generic_precision", 1.00, 2, "Generic precision"),
    PaperMetric("generic_recall", 0.97, 2, "Generic recall"),
    PaperMetric("generic_f1", 0.99, 2, "Generic F1"),
    PaperMetric("normal_precision", 0.91, 2, "Normal precision"),
    PaperMetric("normal_recall", 0.81, 2, "Normal recall"),
    PaperMetric("normal_f1", 0.86, 2, "Normal F1"),
)

IDENTIFIER_COLUMNS = ["config_id", "batch_size", "optimizer", "seed"]


def validate_results(results: pd.DataFrame, identifiers=None) -> None:
    identifiers = IDENTIFIER_COLUMNS if identifiers is None else identifiers
    required = set(identifiers) | {metric.column for metric in PAPER_METRICS}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError("Results are missing required columns: " + ", ".join(missing))
    if results.empty:
        raise ValueError("The results table contains no runs.")
    metric_columns = [metric.column for metric in PAPER_METRICS]
    if results[metric_columns].isna().any().any():
        raise ValueError("Results contain missing paper-comparison values.")


def rounding_aware_error(values: pd.Series, target: float, decimals: int) -> pd.Series:
    half_unit = 0.5 * 10 ** (-decimals)
    lower = target - half_unit
    upper = target + half_unit
    array = values.astype(float)
    return pd.Series(
        np.where(array < lower, lower - array, np.where(array > upper, array - upper, 0.0)),
        index=values.index,
    )


def score_rows(results: pd.DataFrame, identifiers=None) -> pd.DataFrame:
    validate_results(results, identifiers)
    ranked = results.copy()
    raw_squared = np.zeros(len(ranked), dtype=np.float64)
    rounded_squared = np.zeros(len(ranked), dtype=np.float64)
    round_matches = np.zeros(len(ranked), dtype=np.int64)
    for metric in PAPER_METRICS:
        raw_error = ranked[metric.column].astype(float) - metric.target
        rounded_error = rounding_aware_error(
            ranked[metric.column], metric.target, metric.decimals
        )
        ranked[f"error_{metric.column}"] = raw_error.abs()
        raw_squared += raw_error.to_numpy() ** 2
        rounded_squared += rounded_error.to_numpy() ** 2
        round_matches += (rounded_error.to_numpy() == 0.0).astype(np.int64)
    count = len(PAPER_METRICS)
    ranked["paper_raw_rmse"] = np.sqrt(raw_squared / count)
    ranked["paper_rounding_aware_rmse"] = np.sqrt(rounded_squared / count)
    ranked["paper_round_match_count"] = round_matches
    ranked["paper_metric_count"] = count
    return ranked.sort_values(
        ["paper_rounding_aware_rmse", "paper_raw_rmse"], ascending=[True, True]
    ).reset_index(drop=True)


def aggregate_configurations(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric.column for metric in PAPER_METRICS]
    groups = ["config_id", "batch_size", "optimizer"]
    means = results.groupby(groups, as_index=False)[metrics].mean()
    counts = results.groupby(groups, as_index=False).size().rename(columns={"size": "seed_count"})
    return score_rows(means.merge(counts, on=groups, validate="one_to_one"), groups + ["seed_count"])


def metric_details(row: pd.Series) -> pd.DataFrame:
    rows = []
    for metric in PAPER_METRICS:
        observed = float(row[metric.column])
        rounded_error = float(
            rounding_aware_error(pd.Series([observed]), metric.target, metric.decimals).iloc[0]
        )
        rows.append(
            {
                "metric": metric.label,
                "column": metric.column,
                "paper_value": metric.target,
                "paper_decimals": metric.decimals,
                "observed_value": observed,
                "absolute_error": abs(observed - metric.target),
                "matches_paper_rounding": rounded_error == 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize(ranked_runs: pd.DataFrame, ranked_configs: pd.DataFrame) -> dict[str, object]:
    best_run = ranked_runs.iloc[0]
    best_config = ranked_configs.iloc[0]
    return {
        "method": (
            "Equal-weight RMSE across accuracy, loss, and the 15 per-class "
            "precision/recall/F1 values. Published rounding intervals are ranked first."
        ),
        "paper_accuracy_target": 0.81,
        "paper_loss_target": 0.40,
        "best_individual_run": {
            "config_id": str(best_run["config_id"]),
            "seed": int(best_run["seed"]),
            "rounding_aware_rmse": float(best_run["paper_rounding_aware_rmse"]),
            "raw_rmse": float(best_run["paper_raw_rmse"]),
            "round_matches": int(best_run["paper_round_match_count"]),
            "metric_count": int(best_run["paper_metric_count"]),
        },
        "best_configuration_mean": {
            "config_id": str(best_config["config_id"]),
            "seed_count": int(best_config["seed_count"]),
            "rounding_aware_rmse": float(best_config["paper_rounding_aware_rmse"]),
            "raw_rmse": float(best_config["paper_raw_rmse"]),
            "round_matches": int(best_config["paper_round_match_count"]),
            "metric_count": int(best_config["paper_metric_count"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare UNSW-NB15 runs with Table 7.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/unsw_nb15_all_384_concurrent/all_seed_results.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results.is_file():
        raise FileNotFoundError(f"Training results not found: {args.results}")
    output_dir = args.output_dir or args.results.parent / "paper_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results)
    ranked_runs = score_rows(results)
    ranked_configs = aggregate_configurations(results)
    summary = summarize(ranked_runs, ranked_configs)
    ranked_runs.to_csv(output_dir / "ranked_individual_runs.csv", index=False)
    ranked_configs.to_csv(output_dir / "ranked_configuration_means.csv", index=False)
    metric_details(ranked_runs.iloc[0]).to_csv(
        output_dir / "best_run_metric_details.csv", index=False
    )
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Closest individual run:")
    print(json.dumps(summary["best_individual_run"], indent=2))
    print("Closest 64-seed configuration mean:")
    print(json.dumps(summary["best_configuration_mean"], indent=2))
    print(f"Detailed comparison written to {output_dir}")


if __name__ == "__main__":
    main()
