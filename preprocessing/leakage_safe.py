"""
Leakage-Safe Preprocessing Pipeline (Mode B)
Sharma et al. (2024) Replication Methodological Extension

Contrast with Paper-Faithful Mode A:
In Mode A (Paper-Faithful), preprocessing transformations were defined globally
prior to splitting, reflecting the exact published procedural narrative.

In Mode B (Leakage-Safe Modern ML Pipeline):
1. First, perform stratified Train (60%), Validation (15%), and Test (25%) split.
2. Fit categorical label encoders ONLY on the training split. Handle unseen categories gracefully.
3. Fit Min-Max normalization parameters (F_min, F_max) ONLY on the training split.
4. Transform validation and test sets strictly using training-derived parameters.
5. Perform Pearson correlation analysis (|PCC| > 0.95) strictly on the training partition.
6. Verify whether the identical set of redundant features is identified.
7. Save leakage-safe artifacts into results/sensitivity/leakage_safe/.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.encoders import (
    CategoricalFeaturePipeline,
    NSL_KDD_ATTACK_TO_5CLASS,
    NSL_KDD_CLASS_MAPPING,
    UNSW_NB15_TARGET_MAP,
    UNSW_NB15_CLASS_MAPPING
)
from preprocessing.normalization import DeterministicMinMaxScaler
from preprocessing.nsl_kdd import load_raw_nsl_kdd, NSL_KDD_COLUMNS, NSL_KDD_CATEGORICAL_COLS, NSL_KDD_PAPER_REMOVED_FEATURES
from preprocessing.unsw_nb15 import load_raw_unsw_nb15, apply_sampling_policy, UNSW_CATEGORICAL_COLS, UNSW_NON_PREDICTORS, UNSW_PAPER_REMOVED_PREDICTORS
from feature_selection.pearson import PearsonCorrelationSelector


def process_nsl_kdd_leakage_safe(output_dir: Optional[Path] = None, random_seed: int = 42) -> Dict:
    """Runs leakage-safe preprocessing on NSL-KDD."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "sensitivity" / "leakage_safe" / "nsl_kdd"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[LEAKAGE-SAFE] Processing NSL-KDD (Mode B)...")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "nsl_kdd"
    df = load_raw_nsl_kdd(raw_dir, use_train_only=True)

    # Clean target
    df["label_clean"] = df["label"].astype(str).str.lower().str.strip()
    df["target"] = df["label_clean"].map(NSL_KDD_ATTACK_TO_5CLASS).astype(int)
    y_full = df["target"].values

    # Predictors including difficulty_level (36 selected features)
    predictor_cols = [col for col in NSL_KDD_COLUMNS if col != "label"]
    X_df = df[predictor_cols].copy()

    # Step 1: Split data FIRST
    n_samples = len(df)
    all_indices = np.arange(n_samples)

    train_val_idx, test_idx = train_test_split(all_indices, test_size=0.25, random_state=random_seed, stratify=y_full)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.20, random_state=random_seed, stratify=y_full[train_val_idx])

    train_df = X_df.iloc[train_idx].copy()
    val_df = X_df.iloc[val_idx].copy()
    test_df = X_df.iloc[test_idx].copy()

    # Step 2: Fit Encoders ONLY on train_df
    encoder = CategoricalFeaturePipeline(NSL_KDD_CATEGORICAL_COLS)
    train_encoded = encoder.fit_transform(train_df)
    val_encoded = encoder.transform(val_df)
    test_encoded = encoder.transform(test_df)
    encoder.save(output_dir / "encoders_train_only.json")

    # Step 3: Fit Scaler ONLY on train_encoded
    scaler = DeterministicMinMaxScaler()
    train_scaled = scaler.fit_transform(train_encoded)
    val_scaled = scaler.transform(val_encoded)
    test_scaled = scaler.transform(test_encoded)
    scaler.save(output_dir / "scaler_train_only.json")

    # Step 4: Fit Pearson correlation ONLY on train_scaled
    selector = PearsonCorrelationSelector(threshold=0.95)
    selector.fit(train_scaled, target_removed=NSL_KDD_PAPER_REMOVED_FEATURES)
    selector.corr_matrix_.to_csv(output_dir / "train_correlation_matrix.csv")

    selected_cols = [c for c in predictor_cols if c not in selector.removed_features_]
    print(f"[LEAKAGE-SAFE] NSL-KDD train-only correlation identified {len(selector.removed_features_)} removed features:")
    print(f"  Removed: {selector.removed_features_}")
    print(f"  Matches Paper exactly: {set(selector.removed_features_) == set(NSL_KDD_PAPER_REMOVED_FEATURES)}")

    X_train_sel = train_scaled[selected_cols].values.astype(np.float32)
    X_val_sel = val_scaled[selected_cols].values.astype(np.float32)
    X_test_sel = test_scaled[selected_cols].values.astype(np.float32)

    # Save metadata comparing Mode A and Mode B
    meta = {
        "pipeline_mode": "Mode B (Leakage-Safe)",
        "dataset": "NSL-KDD",
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "selected_feature_count": len(selected_cols),
        "selected_features": selected_cols,
        "removed_features": selector.removed_features_,
        "fitted_on_train_only": True,
        "encoder_classes_learned_from_train": {col: list(encoder.encoders[col].classes_) for col in NSL_KDD_CATEGORICAL_COLS}
    }
    with open(output_dir / "nsl_leakage_safe_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    np.savez_compressed(
        output_dir / "nsl_leakage_safe_arrays.npz",
        X_train=X_train_sel,
        X_val=X_val_sel,
        X_test=X_test_sel,
        y_train=y_full[train_idx],
        y_val=y_full[val_idx],
        y_test=y_full[test_idx]
    )
    print(f"[LEAKAGE-SAFE] NSL-KDD arrays saved to {output_dir / 'nsl_leakage_safe_arrays.npz'}")
    return meta


def process_unsw_leakage_safe(output_dir: Optional[Path] = None, random_seed: int = 42) -> Dict:
    """Runs leakage-safe preprocessing on UNSW-NB15."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "sensitivity" / "leakage_safe" / "unsw_nb15"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[LEAKAGE-SAFE] Processing UNSW-NB15 (Mode B)...")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "unsw_nb15"
    df = load_raw_unsw_nb15(raw_dir)

    df["attack_clean"] = df["attack_cat"].fillna("normal").astype(str).str.lower().str.strip()
    df["attack_clean"] = df["attack_clean"].replace({"backdoors": "backdoor"})
    df_5class = df[df["attack_clean"].isin(UNSW_NB15_TARGET_MAP.keys())].copy()

    df_capped = apply_sampling_policy(df_5class, cap_generic=50000, cap_normal=50000, random_seed=random_seed)
    df_capped["target"] = df_capped["attack_clean"].map(UNSW_NB15_TARGET_MAP).astype(int)
    y_full = df_capped["target"].values

    predictor_cols = [c for c in df_capped.columns if c not in UNSW_NON_PREDICTORS and c not in ("attack_clean", "target")]
    X_df = df_capped[predictor_cols].copy()

    # Step 1: Split data FIRST
    n_samples = len(df_capped)
    all_indices = np.arange(n_samples)

    train_val_idx, test_idx = train_test_split(all_indices, test_size=0.25, random_state=random_seed, stratify=y_full)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.20, random_state=random_seed, stratify=y_full[train_val_idx])

    train_df = X_df.iloc[train_idx].copy()
    val_df = X_df.iloc[val_idx].copy()
    test_df = X_df.iloc[test_idx].copy()

    # Step 2: Fit Encoders ONLY on train_df
    encoder = CategoricalFeaturePipeline(UNSW_CATEGORICAL_COLS)
    train_encoded = encoder.fit_transform(train_df)
    val_encoded = encoder.transform(val_df)
    test_encoded = encoder.transform(test_df)
    encoder.save(output_dir / "encoders_train_only.json")

    # Step 3: Fit Scaler ONLY on train_encoded
    scaler = DeterministicMinMaxScaler()
    train_scaled = scaler.fit_transform(train_encoded)
    val_scaled = scaler.transform(val_encoded)
    test_scaled = scaler.transform(test_encoded)
    scaler.save(output_dir / "scaler_train_only.json")

    # Step 4: Fit Pearson correlation ONLY on train_scaled
    selector = PearsonCorrelationSelector(threshold=0.95)
    selector.fit(train_scaled, target_removed=UNSW_PAPER_REMOVED_PREDICTORS)
    selector.corr_matrix_.to_csv(output_dir / "train_correlation_matrix.csv")

    selected_cols = [c for c in predictor_cols if c not in selector.removed_features_]
    print(f"[LEAKAGE-SAFE] UNSW-NB15 train-only correlation identified {len(selector.removed_features_)} removed features:")
    print(f"  Removed: {selector.removed_features_}")

    X_train_sel = train_scaled[selected_cols].values.astype(np.float32)
    X_val_sel = val_scaled[selected_cols].values.astype(np.float32)
    X_test_sel = test_scaled[selected_cols].values.astype(np.float32)

    meta = {
        "pipeline_mode": "Mode B (Leakage-Safe)",
        "dataset": "UNSW-NB15",
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "selected_feature_count": len(selected_cols),
        "selected_features": selected_cols,
        "removed_features": selector.removed_features_,
        "fitted_on_train_only": True
    }
    with open(output_dir / "unsw_leakage_safe_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    np.savez_compressed(
        output_dir / "unsw_leakage_safe_arrays.npz",
        X_train=X_train_sel,
        X_val=X_val_sel,
        X_test=X_test_sel,
        y_train=y_full[train_idx],
        y_val=y_full[val_idx],
        y_test=y_full[test_idx]
    )
    print(f"[LEAKAGE-SAFE] UNSW-NB15 arrays saved to {output_dir / 'unsw_leakage_safe_arrays.npz'}")
    return meta


if __name__ == "__main__":
    process_nsl_kdd_leakage_safe()
    process_unsw_leakage_safe()
