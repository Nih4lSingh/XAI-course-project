"""
Unified Explainability (XAI) CLI Runner
Sharma et al. (2024) Replication Master Pipeline

Generates SHAP explanations for trained DNN models:
- NSL-KDD: Global SHAP (50 test samples) + Local SHAP (DoS & Normal instances)
- UNSW-NB15: Global SHAP (50 test samples) + Local SHAP (Exploits & Normal instances)

Usage:
  python run_xai.py
  python run_xai.py --dataset nsl_kdd
  python run_xai.py --dataset unsw_nb15
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainability.run_xai import run_nsl_kdd_xai, run_unsw_nb15_xai, generate_xai_comparison_report


def parse_args():
    parser = argparse.ArgumentParser(description="Run SHAP explainability pipeline.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["both", "nsl_kdd", "unsw_nb15"],
        default="both",
        help="Dataset for XAI analysis (default: both)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_base = PROJECT_ROOT / "results" / "xai"
    output_base.mkdir(parents=True, exist_ok=True)

    nsl_meta = None
    unsw_meta = None

    if args.dataset in ("both", "nsl_kdd"):
        nsl_meta = run_nsl_kdd_xai(output_base)

    if args.dataset in ("both", "unsw_nb15"):
        unsw_meta = run_unsw_nb15_xai(output_base)

    if args.dataset == "both" and nsl_meta and unsw_meta:
        generate_xai_comparison_report(nsl_meta, unsw_meta, PROJECT_ROOT / "xai_report.md")

    print("\n[SUCCESS] XAI evaluation finished successfully.")


if __name__ == "__main__":
    main()
