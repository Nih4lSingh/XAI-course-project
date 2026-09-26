"""
UNSW-NB15 Preprocessing Pipeline
Replication of Sharma et al. (2024)

Steps:
1. Load raw UNSW-NB15 data (train.csv and test.csv partitions).
2. Filter records to the 5 designated classes ('UNSW-NBnew'):
   0: DoS, 1: Exploits, 2: Fuzzers, 3: Generic, 4: Normal.
3. Exclude 'id', 'attack_cat', and 'label' from the input predictor set X.
4. Perform deterministic label encoding on categorical features:
   ['proto', 'service', 'state'].
5. Perform min-max scaling to [0, 1].
6. Perform deterministic 60% Train / 15% Validation / 25% Test split.
7. Prepare Selected Features (excluding the 5 redundant predictors) and
   49-feature zero-padded matrix for 7x7 2D-CNN.
8. Save split indices and processed numpy arrays for exact reproducibility.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.encoders import (
    CategoricalFeaturePipeline,
    UNSW_NB15_CLASS_MAPPING,
    UNSW_NB15_TARGET_MAP
)
from preprocessing.normalization import DeterministicMinMaxScaler


UNSW_NON_PREDICTORS = ["id", "attack_cat", "label"]
UNSW_CATEGORICAL_COLS = ["proto", "service", "state"]

# 5 redundant predictor features from Sharma et al. Section 3.2
# ('label' is the 6th dropped column in the paper, handled as target separation)
UNSW_PAPER_REMOVED_PREDICTORS = [
    "ct_src_dport_ltm",
    "sloss",
    "dloss",
    "dwin",
    "ct_ftp_cmd",
    "ct_srv_dst"
]


def load_raw_unsw_nb15(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate training and testing sets."""
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path = data_dir / "UNSW_NB15_testing-set.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing UNSW-NB15 raw files in {data_dir}. Run data/download_datasets.py first.")

    df_train = pd.read_csv(train_path, encoding="utf-8-sig")
    df_test = pd.read_csv(test_path, encoding="utf-8-sig")
    df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    return df_combined


def process_unsw_nb15(
    raw_dir: Path,
    output_dir: Path,
    splits_dir: Path,
    random_seed: int = 42
) -> Dict[str, Union[np.ndarray, List[str]]]:
    """
    Executes the full paper-faithful UNSW-NB15 preprocessing pipeline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("[UNSW-NB15] Loading raw data...")
    df = load_raw_unsw_nb15(raw_dir)
    print(f"[UNSW-NB15] Raw shape: {df.shape}")

    # Standardize attack_cat strings
    df["attack_clean"] = df["attack_cat"].fillna("normal").astype(str).str.lower().str.strip()
    # Normalize variants like 'backdoors' -> 'backdoor'
    df["attack_clean"] = df["attack_clean"].replace({"backdoors": "backdoor"})

    # Filter strictly to the 5 classes evaluated by Sharma et al.
    df_5class = df[df["attack_clean"].isin(UNSW_NB15_TARGET_MAP.keys())].copy()
    print(f"[UNSW-NB15] Filtered to 5 classes ('UNSW-NBnew'). Shape: {df_5class.shape} (from {len(df)})")

    df_5class["target"] = df_5class["attack_clean"].map(UNSW_NB15_TARGET_MAP).astype(int)
    y_full = df_5class["target"].values

    print("[UNSW-NB15] Target class distribution:")
    for cls_idx, cls_name in UNSW_NB15_CLASS_MAPPING.items():
        count = (y_full == cls_idx).sum()
        pct = count * 100.0 / len(y_full)
        print(f"  Class {cls_idx} ({cls_name}): {count} samples ({pct:.2f}%)")

    # Determine eligible predictors (all columns except id, attack_cat, label, attack_clean, target)
    predictor_cols = [
        c for c in df_5class.columns
        if c not in UNSW_NON_PREDICTORS and c not in ("attack_clean", "target")
    ]
    print(f"[UNSW-NB15] Total eligible predictors: {len(predictor_cols)}")

    # Deterministic Categorical Label Encoding
    X_df = df_5class[predictor_cols].copy()
    encoder_pipeline = CategoricalFeaturePipeline(UNSW_CATEGORICAL_COLS)
    X_encoded_df = encoder_pipeline.fit_transform(X_df)
    encoder_pipeline.save(output_dir / "unsw_encoders.json")

    # Min-Max Normalization to [0, 1]
    scaler = DeterministicMinMaxScaler()
    X_scaled_df = scaler.fit_transform(X_encoded_df)
    scaler.save(output_dir / "unsw_scaler.json")

    # 60% Train / 15% Validation / 25% Test Split
    n_samples = len(df_5class)
    all_indices = np.arange(n_samples)

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.25,
        random_state=random_seed,
        stratify=y_full
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,
        random_state=random_seed,
        stratify=y_full[train_val_idx]
    )

    # Save split indices for exact reproducibility
    np.save(splits_dir / "unsw_train_indices.npy", train_idx)
    np.save(splits_dir / "unsw_val_indices.npy", val_idx)
    np.save(splits_dir / "unsw_test_indices.npy", test_idx)

    print(f"[UNSW-NB15] Split sizes -> Train: {len(train_idx)} (60%), Val: {len(val_idx)} (15%), Test: {len(test_idx)} (25%)")

    # Feature selection: identify removed and selected columns
    removed_cols = [col for col in UNSW_PAPER_REMOVED_PREDICTORS if col in predictor_cols]
    selected_cols = [col for col in predictor_cols if col not in removed_cols]

    print(f"[UNSW-NB15] Predictors removed: {removed_cols} (count: {len(removed_cols)})")
    print(f"[UNSW-NB15] Selected predictors: {len(selected_cols)}")

    # Extract feature matrices
    X_all = X_scaled_df[predictor_cols].values.astype(np.float32)
    X_selected = X_scaled_df[selected_cols].values.astype(np.float32)

    # 49-feature zero-padded matrix for 7x7 2D-CNN
    pad_len = 49 - X_selected.shape[1]
    if pad_len > 0:
        zero_pad = np.zeros((len(X_selected), pad_len), dtype=np.float32)
        X_selected_49 = np.hstack([X_selected, zero_pad])
    else:
        X_selected_49 = X_selected[:, :49]

    # For all-features 2D-CNN, also pad to 49
    pad_len_all = 49 - X_all.shape[1]
    if pad_len_all > 0:
        zero_pad_all = np.zeros((len(X_all), pad_len_all), dtype=np.float32)
        X_all_49 = np.hstack([X_all, zero_pad_all])
    else:
        X_all_49 = X_all[:, :49]

    # Save arrays
    np.savez_compressed(
        output_dir / "unsw_processed.npz",
        X_all_train=X_all[train_idx],
        X_all_val=X_all[val_idx],
        X_all_test=X_all[test_idx],
        X_selected_train=X_selected[train_idx],
        X_selected_val=X_selected[val_idx],
        X_selected_test=X_selected[test_idx],
        X_selected_49_train=X_selected_49[train_idx],
        X_selected_49_val=X_selected_49[val_idx],
        X_selected_49_test=X_selected_49[test_idx],
        X_all_49_train=X_all_49[train_idx],
        X_all_49_val=X_all_49[val_idx],
        X_all_49_test=X_all_49[test_idx],
        y_train=y_full[train_idx],
        y_val=y_full[val_idx],
        y_test=y_full[test_idx],
        all_features=np.array(predictor_cols),
        selected_features=np.array(selected_cols),
        removed_features=np.array(removed_cols)
    )

    print(f"[UNSW-NB15] Successfully saved processed arrays to {output_dir / 'unsw_processed.npz'}")
    return {
        "n_samples": n_samples,
        "n_all_features": len(predictor_cols),
        "n_selected_features": len(selected_cols),
        "selected_features": selected_cols,
        "removed_features": removed_cols
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    process_unsw_nb15(
        raw_dir=base_dir / "data" / "raw" / "unsw_nb15",
        output_dir=base_dir / "data" / "processed" / "unsw_nb15",
        splits_dir=base_dir / "splits"
    )
