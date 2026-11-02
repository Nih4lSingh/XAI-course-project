"""Run the Phase 1 reproduction for both datasets and all three neural models."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Resolve the enclosing project when the code lives in outputs/source_code;
# when cloned as a standalone repository, use the repository root instead.
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "outputs" / "source_code").is_dir()),
    SCRIPT_PATH.parents[1],
)
# Keep Keras and other libraries' temporary files inside the project. This is
# needed on managed Windows setups where the default package temp directory may
# not be writable by the Python process.
TEMP_DIRECTORY = PROJECT_ROOT / "work" / "tmp"
TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TEMP_DIRECTORY)
os.environ["TMP"] = str(TEMP_DIRECTORY)
tempfile.tempdir = str(TEMP_DIRECTORY)

import joblib
import numpy as np
import pandas as pd

from data_pipeline import prepare_dataset
from models import SEED, seed_everything, split_indices, train_and_score

DEFAULT_OUTPUT = PROJECT_ROOT / "work" / "results" / "phase1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run_dataset(dataset: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    if dataset == "nsl":
        prepared = prepare_dataset("nsl", nsl_train=args.nsl_train)
    else:
        prepared = prepare_dataset("unsw", unsw_train=args.unsw_train, unsw_test=args.unsw_test)
    X, y = prepared["X"], prepared["y"]
    train_idx, val_idx, test_idx = split_indices(y, SEED)
    out_dir = args.output / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({k: v for k, v in prepared.items() if k not in {"X", "y"}}, out_dir / "data_manifest.joblib")
    np.savez_compressed(out_dir / "processed_data.npz", X=X, y=y)
    np.savez_compressed(out_dir / "split_indices.npz", train=train_idx, validation=val_idx, test=test_idx)
    _write_json(out_dir / "data_manifest.json", {k: v for k, v in prepared.items() if k not in {"X", "y"}})

    rows = []
    for model_name in args.models:
        print(f"\n[{dataset}] training {model_name}: {len(train_idx)} train / {len(val_idx)} validation / {len(test_idx)} test")
        seed_everything(SEED)
        model, metrics, probabilities = train_and_score(
            model_name, X, y, prepared["class_names"], train_idx, val_idx, test_idx
        )
        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_dir / "model.keras")
        np.savez_compressed(
            model_dir / "test_outputs.npz",
            y_true=y[test_idx], probabilities=probabilities,
            predictions=probabilities.argmax(axis=1), test_indices=test_idx,
        )
        metrics["dataset"] = dataset
        metrics["seed"] = SEED
        metrics["class_names"] = prepared["class_names"]
        _write_json(model_dir / "metrics.json", metrics)
        rows.append({
            "dataset": dataset, "model": model_name,
            "rows": prepared["source_rows_after_filter_and_sampling"],
            "features": prepared["selected_feature_count"],
            "test_accuracy": metrics["test_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "train_seconds": metrics["train_seconds"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsl-train", type=Path, help="KDDTrain+.txt (no header)")
    parser.add_argument("--unsw-train", type=Path, help="UNSW_NB15_training-set.csv")
    parser.add_argument("--unsw-test", type=Path, help="UNSW_NB15_testing-set.csv")
    parser.add_argument("--dataset", choices=["nsl", "unsw", "both"], default="both")
    parser.add_argument(
        "--models", nargs="+", choices=["dnn", "cnn1d", "cnn2d"],
        default=["dnn", "cnn1d", "cnn2d"],
        help="Models to train; defaults to the paper's DNN, 1D-CNN, and 2D-CNN.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    datasets = ["nsl", "unsw"] if args.dataset == "both" else [args.dataset]
    all_rows = []
    for dataset in datasets:
        all_rows.extend(run_dataset(dataset, args))
    pd.DataFrame(all_rows).to_csv(args.output / "replication_summary.csv", index=False)
    print(f"\nSummary saved to {args.output / 'replication_summary.csv'}")


if __name__ == "__main__":
    main()
