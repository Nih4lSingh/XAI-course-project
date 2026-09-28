"""
UNSW-NB15 Feature Selection Verification
Replication of Sharma et al. (2024)

Verifies:
1. Pearson correlation calculation on UNSW-NB15 predictors.
2. Identifies pairs with |PCC| > 0.95.
3. Confirms the paper-reported redundant features:
   - ct_src_dport_ltm
   - sloss / dloss (packet loss)
   - dwin
   - ct_ftp_cmd
   - ct_srv_dst
   (and documents target separation of 'label')
4. Generates correlation heatmap and saves selected/removed feature text files.
5. Saves 7x7 grid mapping.
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
from feature_selection.feature_to_grid_mapping import get_unsw_nb15_grid_mapper
from preprocessing.unsw_nb15 import (
    load_raw_unsw_nb15,
    UNSW_NON_PREDICTORS,
    UNSW_CATEGORICAL_COLS,
    UNSW_PAPER_REMOVED_PREDICTORS
)
from preprocessing.encoders import CategoricalFeaturePipeline, UNSW_NB15_TARGET_MAP


def run_unsw_feature_selection(
    raw_dir: Path,
    output_dir: Path,
    reports_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RUNNING UNSW-NB15 FEATURE SELECTION VERIFICATION")
    print("=" * 60)

    df = load_raw_unsw_nb15(raw_dir)
    df["attack_clean"] = df["attack_cat"].fillna("normal").astype(str).str.lower().str.strip()
    df["attack_clean"] = df["attack_clean"].replace({"backdoors": "backdoor"})
    df_5class = df[df["attack_clean"].isin(UNSW_NB15_TARGET_MAP.keys())].copy()

    predictor_cols = [
        c for c in df_5class.columns
        if c not in UNSW_NON_PREDICTORS and c not in ("attack_clean", "target")
    ]

    # Deterministic Categorical Label Encoding
    X_df = df_5class[predictor_cols].copy()
    encoder = CategoricalFeaturePipeline(UNSW_CATEGORICAL_COLS)
    X_encoded = encoder.fit_transform(X_df)

    selector = PearsonCorrelationSelector(threshold=0.95)
    selector.fit(X_encoded, target_removed=UNSW_PAPER_REMOVED_PREDICTORS)

    print(f"\nTotal predictor features: {len(predictor_cols)}")
    print(f"Highly correlated pairs found (|PCC| > 0.95): {len(selector.correlated_pairs_)}")
    for f1, f2, corr in selector.correlated_pairs_:
        print(f"  {f1} <--> {f2}: PCC = {corr:+.4f}")

    print("\nPaper-specified removed predictors:")
    for f in UNSW_PAPER_REMOVED_PREDICTORS:
        in_pairs = any(f in (p[0], p[1]) for p in selector.correlated_pairs_)
        status = "CONFIRMED in |PCC| > 0.95 pair" if in_pairs else "NOT in |PCC| > 0.95 pair"
        print(f"  - {f}: {status}")

    # Save heatmap
    selector.save_heatmap(
        reports_dir / "unsw_correlation_heatmap.png",
        title="UNSW-NB15 Pearson Correlation Heatmap"
    )

    # Save feature lists
    selected_cols = [c for c in predictor_cols if c not in UNSW_PAPER_REMOVED_PREDICTORS]
    selector.save_feature_lists(
        output_dir / "unsw_nb15_selected_features.txt",
        output_dir / "unsw_nb15_removed_features.txt"
    )

    # Save grid mapping
    grid_mapper = get_unsw_nb15_grid_mapper(selected_cols)
    grid_mapper.save_mapping(output_dir / "unsw_nb15_grid_mapping.json")

    print("\n[COMPLETE] UNSW-NB15 feature selection verification finished.")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    run_unsw_feature_selection(
        raw_dir=base / "data" / "raw" / "unsw_nb15",
        output_dir=base / "results" / "feature_selection",
        reports_dir=base / "reports" / "figures"
    )
