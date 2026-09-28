"""
Root CLI Runner for Multi-Seed Empirical Replication
Replication of Sharma et al. (2024) across ALL 6 Models

Usage Examples:
  # Quick smoke-test (2 seeds, 2 epochs) on NSL DNN:
  python run_multiseed.py --exp_id NSL_SELECTED_DNN --quick

  # Full multi-seed sweep (5 seeds) on all 6 models:
  python run_multiseed.py --all --num_seeds 5

  # Run multi-seed sweep on UNSW models with 10 seeds:
  python run_multiseed.py --dataset unsw --num_seeds 10

  # Run specific seeds:
  python run_multiseed.py --exp_id NSL_SELECTED_2DCNN --seeds 42,20240901,1414324351

  # Regenerate master report from existing seed runs:
  python run_multiseed.py --report_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from training.multiseed_sweep import (
    CANONICAL_6_MODELS,
    DEFAULT_MASTER_SEED_NSL,
    DEFAULT_MASTER_SEED_UNSW,
    MultiSeedExperimentRunner,
    generate_consolidated_report,
    select_random_seeds,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed empirical replication across all Sharma et al. (2024) models."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run multi-seed sweep across all 6 canonical models.",
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default=None,
        help="Specific experiment ID to run (e.g. NSL_SELECTED_DNN, UNSW_SELECTED_2DCNN).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of model IDs to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nsl", "unsw", "all"],
        default=None,
        help="Filter models by dataset ('nsl' or 'unsw').",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of pseudo-random seeds to evaluate (default: 5).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of explicit seeds (e.g., '42,100,20240901').",
    )
    parser.add_argument(
        "--master_seed_nsl",
        type=int,
        default=DEFAULT_MASTER_SEED_NSL,
        help=f"Master seed for NSL-KDD seed generator (default: {DEFAULT_MASTER_SEED_NSL}).",
    )
    parser.add_argument(
        "--master_seed_unsw",
        type=int,
        default=DEFAULT_MASTER_SEED_UNSW,
        help=f"Master seed for UNSW-NB15 seed generator (default: {DEFAULT_MASTER_SEED_UNSW}).",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["resplit", "canonical"],
        default="resplit",
        help="Split mode: 'resplit' (stratified partition per seed à la LordKarsSama) or 'canonical' (fixed split).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128 matching paper baseline).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs per seed (default: 20).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test mode: overrides epochs to 2 and num_seeds to 2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previously completed seed runs.",
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Only generate the consolidated report from existing seed runs without training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "multiseed"),
        help="Output directory for multi-seed sweep results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = MultiSeedExperimentRunner(
        output_dir=output_dir,
        split_mode=args.split_mode,
    )

    # Determine which models to run
    target_models: List[str] = []
    if args.all:
        target_models = list(CANONICAL_6_MODELS)
    elif args.dataset:
        if args.dataset == "nsl":
            target_models = [m for m in CANONICAL_6_MODELS if "NSL" in m]
        elif args.dataset == "unsw":
            target_models = [m for m in CANONICAL_6_MODELS if "UNSW" in m]
        else:
            target_models = list(CANONICAL_6_MODELS)
    elif args.models:
        target_models = args.models
    elif args.exp_id:
        target_models = [args.exp_id]
    elif args.report_only:
        target_models = list(CANONICAL_6_MODELS)
    else:
        # Default to all 6 models
        target_models = list(CANONICAL_6_MODELS)

    epochs = 2 if args.quick else args.epochs
    num_seeds = 2 if args.quick else args.num_seeds

    print("\n" + "=" * 65)
    print(" SHARMA ET AL. (2024) - MULTI-SEED REPLICATION FRAMEWORK")
    print(f" Models to evaluate ({len(target_models)}): {', '.join(target_models)}")
    print(f" Seeds per model:    {num_seeds}")
    print(f" Split mode:         {args.split_mode}")
    print(f" Batch size:         {args.batch_size}")
    print(f" Epochs:             {epochs}")
    print(f" Output directory:   {output_dir}")
    print("=" * 65 + "\n")

    summaries: Dict[str, Dict] = {}

    if not args.report_only:
        for exp_id in target_models:
            # Generate seeds for this experiment
            if args.seeds:
                seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
            else:
                master_seed = (
                    args.master_seed_nsl if "NSL" in exp_id.upper() else args.master_seed_unsw
                )
                seed_list = select_random_seeds(count=num_seeds, master_seed=master_seed)

            # Run sweep for this model
            runs_df, summary = runner.run_sweep(
                experiment_id=exp_id,
                seeds=seed_list,
                epochs=epochs,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
            summaries[exp_id] = summary

    # Load existing summaries if report_only or to consolidate all completed models
    for exp_id in CANONICAL_6_MODELS:
        if exp_id not in summaries:
            exp_csv = output_dir / exp_id.lower() / "seed_runs.csv"
            if exp_csv.is_file():
                df = pd.read_csv(exp_csv)
                if not df.empty:
                    summaries[exp_id] = runner.compute_statistics(df, exp_id)

    # Generate consolidated cross-model report
    if summaries:
        comp_df, report_md = generate_consolidated_report(output_dir, summaries)
        print("\n" + "=" * 65)
        print(" MASTER MULTI-SEED BENCHMARK TABLE")
        print("=" * 65)
        print(comp_df.to_string(index=False))
        print(f"\nConsolidated CSV:    {output_dir / 'multiseed_paper_comparison.csv'}")
        print(f"Consolidated Report: {output_dir / 'multiseed_report.md'}")
        print("=" * 65 + "\n")
    else:
        print("No completed seed runs found to report.")


if __name__ == "__main__":
    main()
