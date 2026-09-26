"""
Unified Single Experiment CLI Runner
Sharma et al. (2024) Replication Master Pipeline

Usage examples:
  python run_experiment.py --dataset nsl_kdd --model dnn --features selected
  python run_experiment.py --dataset unsw_nb15 --model 2dcnn --features all --epochs 20
  python run_experiment.py --exp_id NSL_SELECTED_1DCNN
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_experiment import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single IDS replication experiment.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nsl_kdd", "unsw_nb15"],
        help="Dataset name ('nsl_kdd' or 'unsw_nb15')"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["dnn", "1dcnn", "2dcnn"],
        help="Model architecture ('dnn', '1dcnn', or '2dcnn')"
    )
    parser.add_argument(
        "--features",
        type=str,
        choices=["selected", "all"],
        default="selected",
        help="Feature selection mode ('selected' or 'all')"
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default=None,
        help="Direct Experiment ID (e.g. NSL_SELECTED_DNN, UNSW_ALL_2DCNN)"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for DNN (default: 0.0)")
    parser.add_argument("--verbose", type=int, default=1, help="Keras training verbosity (default: 1)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.exp_id:
        exp_id = args.exp_id.upper()
    else:
        if not args.dataset or not args.model:
            print("Error: Specify both --dataset and --model, or provide --exp_id.")
            sys.exit(1)
        dataset_prefix = "NSL" if "nsl" in args.dataset.lower() else "UNSW"
        feature_tag = args.features.upper()
        model_tag = args.model.upper()
        exp_id = f"{dataset_prefix}_{feature_tag}_{model_tag}"

    print(f"\n=======================================================")
    print(f"Executing Experiment: {exp_id}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, Dropout: {args.dropout}")
    print(f"=======================================================\n")

    metrics = run_experiment(
        experiment_id=exp_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_rate=args.dropout,
        verbose=args.verbose
    )

    print("\n--- Experiment Summary ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
