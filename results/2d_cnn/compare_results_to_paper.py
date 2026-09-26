"""Rank trained NSL-KDD runs by similarity to the paper's 2D-CNN results.

The script is read-only with respect to training outputs. It reads
``all_seed_results.csv`` and writes comparison reports beside that file.
"""

from __future__ import annotations

import argparse
import json
import math
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


# The paper's narrative reports 0.994 accuracy and 0.01 loss. Table 6 reports
# the following two-decimal per-class metrics for the 2D-CNN.
PAPER_METRICS = (
    PaperMetric("test_accuracy", 0.994, 3, "Accuracy"),
    PaperMetric("test_loss", 0.01, 2, "Loss"),
    PaperMetric("dos_precision", 1.00, 2, "DoS precision"),
    PaperMetric("dos_recall", 1.00, 2, "DoS recall"),
    PaperMetric("dos_f1", 1.00, 2, "DoS F1"),
    PaperMetric("normal_precision", 0.99, 2, "Normal precision"),
    PaperMetric("normal_recall", 1.00, 2, "Normal recall"),
    PaperMetric("normal_f1", 0.99, 2, "Normal F1"),
    PaperMetric("probe_precision", 0.98, 2, "Probe precision"),
    PaperMetric("probe_recall", 0.99, 2, "Probe recall"),
    PaperMetric("probe_f1", 0.99, 2, "Probe F1"),
    PaperMetric("r2l_precision", 0.93, 2, "R2L precision"),
    PaperMetric("r2l_recall", 0.51, 2, "R2L recall"),
    PaperMetric("r2l_f1", 0.66, 2, "R2L F1"),
    PaperMetric("u2r_precision", 0.60, 2, "U2R precision"),
    PaperMetric("u2r_recall", 0.23, 2, "U2R recall"),
    PaperMetric("u2r_f1", 0.33, 2, "U2R F1"),
)

IDENTIFIER_COLUMNS = [
    "config_id",
    "batch_size",
    "feature_mode",
    "optimizer",
    "seed",
]


def validate_results(
    results: pd.DataFrame, identifier_columns: list[str] | None = None
) -> None:
    identifiers = IDENTIFIER_COLUMNS if identifier_columns is None else identifier_columns
    required = set(identifiers) | {metric.column for metric in PAPER_METRICS}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(
            "Results are missing columns required for paper comparison: "
            + ", ".join(missing)
            + ". Re-run evaluation with the current GPU runner to emit "
            "per-class precision and recall."
        )
    if results.empty:
        raise ValueError("The results table contains no runs.")
    numeric_columns = [metric.column for metric in PAPER_METRICS]
    if results[numeric_columns].isna().any().any():
        bad = results[numeric_columns].columns[
            results[numeric_columns].isna().any()
        ].tolist()
        raise ValueError(f"Results contain missing comparison values in: {bad}")


def rounding_aware_error(values: pd.Series, target: float, decimals: int) -> pd.Series:
    """Distance from the interval that would round to the published value."""
    half_unit = 0.5 * 10 ** (-decimals)
    lower = target - half_unit
    upper = target + half_unit
    array = values.astype(float)
    return pd.Series(
        np.where(array < lower, lower - array, np.where(array > upper, array - upper, 0.0)),
        index=values.index,
    )


def score_rows(
    results: pd.DataFrame, identifier_columns: list[str] | None = None
) -> pd.DataFrame:
    validate_results(results, identifier_columns)
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

    metric_count = len(PAPER_METRICS)
    ranked["paper_raw_rmse"] = np.sqrt(raw_squared / metric_count)
    ranked["paper_rounding_aware_rmse"] = np.sqrt(
        rounded_squared / metric_count
    )
    ranked["paper_round_match_count"] = round_matches
    ranked["paper_metric_count"] = metric_count
    return ranked.sort_values(
        ["paper_rounding_aware_rmse", "paper_raw_rmse"],
        ascending=[True, True],
    ).reset_index(drop=True)


def aggregate_configurations(results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [metric.column for metric in PAPER_METRICS]
    group_columns = ["config_id", "batch_size", "feature_mode", "optimizer"]
    means = results.groupby(group_columns, as_index=False)[metric_columns].mean()
    counts = results.groupby(group_columns, as_index=False).size()
    means = means.merge(counts, on=group_columns, validate="one_to_one")
    means = means.rename(columns={"size": "seed_count"})
    return score_rows(means, group_columns + ["seed_count"])


def metric_details(row: pd.Series) -> pd.DataFrame:
    details = []
    for metric in PAPER_METRICS:
        observed = float(row[metric.column])
        details.append(
            {
                "metric": metric.label,
                "column": metric.column,
                "paper_value": metric.target,
                "paper_decimals": metric.decimals,
                "observed_value": observed,
                "absolute_error": abs(observed - metric.target),
                "matches_paper_rounding": (
                    round(observed, metric.decimals)
                    == round(metric.target, metric.decimals)
                ),
            }
        )
    return pd.DataFrame(details)


def summarize(
    ranked_runs: pd.DataFrame, ranked_configs: pd.DataFrame
) -> dict[str, object]:
    best_run = ranked_runs.iloc[0]
    best_config = ranked_configs.iloc[0]
    return {
        "method": (
            "Equal-weight RMSE across accuracy, loss, and the 15 per-class "
            "precision/recall/F1 values. Ranking first uses distance from each "
            "paper value's published rounding interval, then raw RMSE as a tie-breaker."
        ),
        "paper_accuracy_target": 0.994,
        "paper_loss_target": 0.01,
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
            "rounding_aware_rmse": float(
                best_config["paper_rounding_aware_rmse"]
            ),
            "raw_rmse": float(best_config["paper_raw_rmse"]),
            "round_matches": int(best_config["paper_round_match_count"]),
            "metric_count": int(best_config["paper_metric_count"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find NSL-KDD runs closest to the paper's 2D-CNN results."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/nsl_kdd_all_768_concurrent/all_seed_results.csv"),
        help="CSV produced by nsl_kdd_gpu_all_concurrent.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to a paper_comparison folder beside the results CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results.is_file():
        raise FileNotFoundError(
            f"Training results not found: {args.results}. Run the training job first."
        )
    output_dir = args.output_dir or args.results.parent / "paper_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results)
    ranked_runs = score_rows(results)
    ranked_configs = aggregate_configurations(results)
    summary = summarize(ranked_runs, ranked_configs)
    best_details = metric_details(ranked_runs.iloc[0])

    ranked_runs.to_csv(output_dir / "ranked_individual_runs.csv", index=False)
    ranked_configs.to_csv(output_dir / "ranked_configuration_means.csv", index=False)
    best_details.to_csv(output_dir / "best_run_metric_details.csv", index=False)
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
