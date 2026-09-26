"""
NSL-KDD Preprocessing Pipeline
Replication of Sharma et al. (2024)

Steps:
1. Load raw NSL-KDD data (KDDTrain+ and KDDTest+).
2. Assign official 41 feature column names + label + difficulty_level.
3. Map granular attack categories to the 5 canonical classes:
   0: DoS, 1: Normal, 2: Probe, 3: R2L, 4: U2R.
4. Perform deterministic label encoding on categorical features:
   ['protocol_type', 'service', 'flag'].
5. Perform min-max scaling to [0, 1].
6. Perform deterministic 60% Train / 15% Validation / 25% Test split.
7. Save split indices and processed numpy arrays for exact reproducibility.
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
    NSL_KDD_ATTACK_TO_5CLASS,
    NSL_KDD_CLASS_MAPPING
)
from preprocessing.normalization import DeterministicMinMaxScaler


NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

NSL_KDD_CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
NSL_KDD_PREDICTOR_COLS = [col for col in NSL_KDD_COLUMNS if col not in ("label", "difficulty_level")]

# Explicit 6 removed features from Sharma et al. Section 3.2
NSL_KDD_PAPER_REMOVED_FEATURES = [
    "srv_serror_rate",
    "dst_host_srv_rerror_rate",
    "num_root",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "srv_rerror_rate"
]

NSL_KDD_PAPER_SELECTED_FEATURES = [
    col for col in NSL_KDD_PREDICTOR_COLS if col not in NSL_KDD_PAPER_REMOVED_FEATURES
]


def load_raw_nsl_kdd(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate KDDTrain+ and KDDTest+ files."""
    train_path = data_dir / "KDDTrain+.txt"
    test_path = data_dir / "KDDTest+.txt"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing NSL-KDD raw files in {data_dir}. Run data/download_datasets.py first.")

    df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
    df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    return df_combined


def process_nsl_kdd(
    raw_dir: Path,
    output_dir: Path,
    splits_dir: Path,
    random_seed: int = 42
) -> Dict[str, Union[np.ndarray, List[str]]]:
    """
    Executes the full paper-faithful NSL-KDD preprocessing pipeline.
    Produces:
    - Train/Val/Test splits (60% / 15% / 25%)
    - All features matrices
    - Selected features matrices (35 features + optional 36-feature zero-padded for 6x6 grid)
    - Target vectors (5 classes)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("[NSL-KDD] Loading raw data...")
    df = load_raw_nsl_kdd(raw_dir)
    print(f"[NSL-KDD] Raw shape: {df.shape}")

    # 1. Map target attacks to 5 canonical classes
    df["label_clean"] = df["label"].astype(str).str.lower().str.strip()
    df["target"] = df["label_clean"].map(NSL_KDD_ATTACK_TO_5CLASS)

    # Check for unmapped labels
    unmapped = df[df["target"].isna()]
    if len(unmapped) > 0:
        print(f"[WARN] Found {len(unmapped)} records with unmapped labels. Dropping them.")
        df = df.dropna(subset=["target"])

    df["target"] = df["target"].astype(int)
    y_full = df["target"].values

    print("[NSL-KDD] Target class distribution:")
    for cls_idx, cls_name in NSL_KDD_CLASS_MAPPING.items():
        count = (y_full == cls_idx).sum()
        pct = count * 100.0 / len(y_full)
        print(f"  Class {cls_idx} ({cls_name}): {count} samples ({pct:.2f}%)")

    # 2. Extract predictors and perform Categorical Label Encoding
    X_df = df[NSL_KDD_PREDICTOR_COLS].copy()
    encoder_pipeline = CategoricalFeaturePipeline(NSL_KDD_CATEGORICAL_COLS)
    X_encoded_df = encoder_pipeline.fit_transform(X_df)
    encoder_pipeline.save(output_dir / "nsl_kdd_encoders.json")

    # 3. Min-Max Normalization to [0, 1]
    scaler = DeterministicMinMaxScaler()
    X_scaled_df = scaler.fit_transform(X_encoded_df)
    scaler.save(output_dir / "nsl_kdd_scaler.json")

    # 4. Create Train (60%), Val (15%), Test (25%) Split
    # Protocol: Split 25% for test; then split remaining 75% into 80% train (60% total) and 20% val (15% total)
    n_samples = len(df)
    all_indices = np.arange(n_samples)

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.25,
        random_state=random_seed,
        stratify=y_full
    )

    # 15% of total / 75% of total = 15/75 = 0.20
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,
        random_state=random_seed,
        stratify=y_full[train_val_idx]
    )

    # Save split indices for exact reproducibility
    np.save(splits_dir / "nsl_kdd_train_indices.npy", train_idx)
    np.save(splits_dir / "nsl_kdd_val_indices.npy", val_idx)
    np.save(splits_dir / "nsl_kdd_test_indices.npy", test_idx)

    print(f"[NSL-KDD] Split sizes -> Train: {len(train_idx)} (60%), Val: {len(val_idx)} (15%), Test: {len(test_idx)} (25%)")

    # 5. Extract Feature Sets
    # All Features: 41 predictors
    X_all = X_scaled_df[NSL_KDD_PREDICTOR_COLS].values.astype(np.float32)

    # Selected Features: 35 predictors
    X_selected = X_scaled_df[NSL_KDD_PAPER_SELECTED_FEATURES].values.astype(np.float32)

    # 36-feature version for 6x6 2D-CNN (append 1 zero-padding column)
    zero_pad = np.zeros((len(X_selected), 1), dtype=np.float32)
    X_selected_36 = np.hstack([X_selected, zero_pad])

    # Save arrays
    np.savez_compressed(
        output_dir / "nsl_kdd_processed.npz",
        X_all_train=X_all[train_idx],
        X_all_val=X_all[val_idx],
        X_all_test=X_all[test_idx],
        X_selected_train=X_selected[train_idx],
        X_selected_val=X_selected[val_idx],
        X_selected_test=X_selected[test_idx],
        X_selected_36_train=X_selected_36[train_idx],
        X_selected_36_val=X_selected_36[val_idx],
        X_selected_36_test=X_selected_36[test_idx],
        y_train=y_full[train_idx],
        y_val=y_full[val_idx],
        y_test=y_full[test_idx],
        all_features=np.array(NSL_KDD_PREDICTOR_COLS),
        selected_features=np.array(NSL_KDD_PAPER_SELECTED_FEATURES),
        removed_features=np.array(NSL_KDD_PAPER_REMOVED_FEATURES)
    )

    print(f"[NSL-KDD] Successfully saved processed arrays to {output_dir / 'nsl_kdd_processed.npz'}")
    return {
        "n_samples": n_samples,
        "n_all_features": len(NSL_KDD_PREDICTOR_COLS),
        "n_selected_features": len(NSL_KDD_PAPER_SELECTED_FEATURES),
        "selected_features": NSL_KDD_PAPER_SELECTED_FEATURES,
        "removed_features": NSL_KDD_PAPER_REMOVED_FEATURES
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    process_nsl_kdd(
        raw_dir=base_dir / "data" / "raw" / "nsl_kdd",
        output_dir=base_dir / "data" / "processed" / "nsl_kdd",
        splits_dir=base_dir / "splits"
    )
