"""
Master Replication Orchestrator CLI
Sharma et al. (2024) Replication Master Pipeline

Runs all 12 experimental configurations (or selected subset) across
NSL-KDD and UNSW-NB15, generating comparative metrics, confusion matrices,
training curves, and paper-vs-reproduction tables.

Usage:
  python run_all.py
  python run_all.py --epochs 20 --batch_size 64 --no_sensitivity
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_all import run_all_experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full 12-model replication matrix.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs per model (default: 20)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--no_sensitivity", action="store_true", help="Skip dropout sensitivity analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    print("\nStarting execution of all experimental configurations...")
    df_results = run_all_experiments(
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_sensitivity=not args.no_sensitivity
    )
    print("\nReplication execution completed successfully!")
    print(df_results.to_string())


if __name__ == "__main__":
    main()
