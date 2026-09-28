"""
NSL-KDD Feature Selection Verification
Replication of Sharma et al. (2024)

Verifies:
1. Pearson correlation calculation on NSL-KDD predictors.
2. Identifies pairs with |PCC| > 0.95.
3. Confirms the 6 paper-reported removed features:
   - srv_serror_rate
   - dst_host_srv_rerror_rate
   - num_root
   - dst_host_serror_rate
   - dst_host_srv_serror_rate
   - srv_rerror_rate
4. Generates correlation heatmap and saves selected/removed feature text files.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from feature_selection.pearson_selector import PearsonCorrelationSelector
from feature_selection.feature_to_grid_mapping import get_nsl_kdd_grid_mapper
from preprocessing.nsl_kdd import (
    load_raw_nsl_kdd,
    NSL_KDD_CATEGORICAL_COLS,
    NSL_KDD_PREDICTOR_COLS,
    NSL_KDD_PAPER_REMOVED_FEATURES,
    NSL_KDD_PAPER_SELECTED_FEATURES
)
from preprocessing.encoders import CategoricalFeaturePipeline


def run_nsl_kdd_feature_selection(
    raw_dir: Path,
    output_dir: Path,
    reports_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RUNNING NSL-KDD FEATURE SELECTION VERIFICATION")
    print("=" * 60)

    # Load raw data and encode categoricals for correlation analysis
    df = load_raw_nsl_kdd(raw_dir)
    X_df = df[NSL_KDD_PREDICTOR_COLS].copy()
    encoder = CategoricalFeaturePipeline(NSL_KDD_CATEGORICAL_COLS)
    X_encoded = encoder.fit_transform(X_df)

    selector = PearsonCorrelationSelector(threshold=0.95)
    selector.fit(X_encoded, target_removed=NSL_KDD_PAPER_REMOVED_FEATURES)

    print(f"\nTotal predictor features: {len(NSL_KDD_PREDICTOR_COLS)}")
    print(f"Highly correlated pairs found (|PCC| > 0.95): {len(selector.correlated_pairs_)}")
    for f1, f2, corr in selector.correlated_pairs_:
        print(f"  {f1} <--> {f2}: PCC = {corr:+.4f}")

    print("\nPaper-specified removed features:")
    for f in NSL_KDD_PAPER_REMOVED_FEATURES:
        in_pairs = any(f in (p[0], p[1]) for p in selector.correlated_pairs_)
        status = "CONFIRMED in |PCC| > 0.95 pair" if in_pairs else "NOT in |PCC| > 0.95 pair"
        print(f"  - {f}: {status}")

    # Save heatmap
    selector.save_heatmap(
        reports_dir / "nsl_kdd_correlation_heatmap.png",
        title="NSL-KDD Pearson Correlation Heatmap"
    )

    # Save feature lists
    selector.save_feature_lists(
        output_dir / "nsl_kdd_selected_features.txt",
        output_dir / "nsl_kdd_removed_features.txt"
    )

    # Save grid mapping
    grid_mapper = get_nsl_kdd_grid_mapper(NSL_KDD_PAPER_SELECTED_FEATURES)
    grid_mapper.save_mapping(output_dir / "nsl_kdd_grid_mapping.json")

    print("\n[COMPLETE] NSL-KDD feature selection verification finished.")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    run_nsl_kdd_feature_selection(
        raw_dir=base / "data" / "raw" / "nsl_kdd",
        output_dir=base / "results" / "feature_selection",
        reports_dir=base / "reports" / "figures"
    )
