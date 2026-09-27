"""
Master Replication Orchestrator CLI
Sharma et al. (2024) Replication Master Pipeline

Runs the 6 paper replication models across NSL-KDD and UNSW-NB15, generating
comparative metrics, confusion matrices, training curves, and paper-vs-reproduction tables.

Usage:
  python run_all.py
  python run_all.py --epochs 20 --batch_size 128
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_all import run_all_experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 6-model paper replication matrix.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs per model (default: 20)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--force", action="store_true", help="Force retraining even if metrics exist")
    return parser.parse_args()


def main():
    args = parse_args()
    print("\nStarting execution of the 6 canonical paper replication models...")
    df_results = run_all_experiments(
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_existing=not args.force
    )
    print("\nReplication execution completed successfully!")
    print(df_results.to_string())


if __name__ == "__main__":
    main()
