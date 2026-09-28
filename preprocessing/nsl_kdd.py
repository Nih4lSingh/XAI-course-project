"""
NSL-KDD Preprocessing Module
Sharma et al. (2024) Replication Master Pipeline

Methodology:
1. Ingest KDDTrain+.txt ONLY (125,973 records).
2. Map attack types to the 5 canonical classes:
   0: DoS, 1: Normal, 2: Probe, 3: R2L, 4: U2R
3. Deterministic Label Encoding on categorical columns:
   ['protocol_type', 'service', 'flag']
4. Min-Max Normalization to [0, 1].
5. Stratified 60% Train / 15% Validation / 25% Test split (seed=42).
6. Pearson Feature Selection (|PCC| > 0.95):
   Remove 6 collinear features:
   ['srv_serror_rate', 'dst_host_srv_rerror_rate', 'num_root',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'srv_rerror_rate']
7. Feature Representations:
   - Selected Features: 36 features (with difficulty_level retained) -> 6x6 grid with 0 padding
     (or 35 features + 1 zero padding)
   - All Features: 42 features (or 41 without difficulty)
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

# 6 redundant features explicitly removed in Sharma et al. Section 3.2
NSL_KDD_PAPER_REMOVED_FEATURES = [
    "srv_serror_rate",
    "dst_host_srv_rerror_rate",
    "num_root",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "srv_rerror_rate"
]

NSL_KDD_PREDICTOR_COLS = [c for c in NSL_KDD_COLUMNS if c != "label"]
NSL_KDD_PAPER_SELECTED_FEATURES = [c for c in NSL_KDD_PREDICTOR_COLS if c not in NSL_KDD_PAPER_REMOVED_FEATURES]


def load_raw_nsl_kdd(raw_dir: Path, use_train_only: bool = True) -> pd.DataFrame:
    """
    Loads raw NSL-KDD dataset.
    Default paper-faithful mode ingests KDDTrain+.txt ONLY (125,973 records).
    """
    train_path = raw_dir / "KDDTrain+.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing KDDTrain+.txt in {raw_dir}. Run data/download_datasets.py first.")

    df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    if not use_train_only:
        test_path = raw_dir / "KDDTest+.txt"
        if test_path.exists():
            df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
            return pd.concat([df_train, df_test], axis=0, ignore_index=True)
    return df_train


def process_nsl_kdd(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
    retain_difficulty: bool = True,
    use_train_only: bool = True,
    random_seed: int = 42
) -> Dict[str, Union[np.ndarray, List[str], int]]:
    """
    Executes the NSL-KDD preprocessing pipeline.
    """
    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "data" / "raw" / "nsl_kdd"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "processed" / "nsl_kdd"
    if splits_dir is None:
        splits_dir = PROJECT_ROOT / "splits"

    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"[NSL-KDD] Loading raw data (use_train_only={use_train_only})...")
    df = load_raw_nsl_kdd(raw_dir, use_train_only=use_train_only)
    print(f"[NSL-KDD] Loaded {len(df)} records with shape {df.shape}")

    # Map attack labels to canonical 5 classes
    df["label_clean"] = df["label"].astype(str).str.lower().str.strip()
    df["target"] = df["label_clean"].map(NSL_KDD_ATTACK_TO_5CLASS)

    unmapped = df[df["target"].isna()]
    if len(unmapped) > 0:
        print(f"[WARN] Found {len(unmapped)} unmapped attack labels; dropping.")
        df = df.dropna(subset=["target"])

    df["target"] = df["target"].astype(int)
    y_full = df["target"].values

    print("[NSL-KDD] Target class distribution:")
    for cls_idx, cls_name in NSL_KDD_CLASS_MAPPING.items():
        count = (y_full == cls_idx).sum()
        pct = count * 100.0 / len(y_full)
        print(f"  Class {cls_idx} ({cls_name}): {count} samples ({pct:.2f}%)")

    # Define predictor columns
    if retain_difficulty:
        predictor_cols = [col for col in NSL_KDD_COLUMNS if col != "label"]
    else:
        predictor_cols = [col for col in NSL_KDD_COLUMNS if col not in ("label", "difficulty_level")]

    # Categorical Label Encoding
    X_df = df[predictor_cols].copy()
    encoder_pipeline = CategoricalFeaturePipeline(NSL_KDD_CATEGORICAL_COLS)
    X_encoded_df = encoder_pipeline.fit_transform(X_df)
    encoder_pipeline.save(output_dir / "nsl_kdd_encoders.json")

    # Min-Max Normalization to [0, 1]
    scaler = DeterministicMinMaxScaler()
    X_scaled_df = scaler.fit_transform(X_encoded_df)
    scaler.save(output_dir / "nsl_kdd_scaler.json")

    # Deterministic Stratified 60% Train / 15% Val / 25% Test Split
    n_samples = len(df)
    all_indices = np.arange(n_samples)

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.25,
        random_state=random_seed,
        stratify=y_full
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,  # 0.20 of 0.75 = 0.15 of total
        random_state=random_seed,
        stratify=y_full[train_val_idx]
    )

    np.save(splits_dir / "nsl_kdd_train_indices.npy", train_idx)
    np.save(splits_dir / "nsl_kdd_val_indices.npy", val_idx)
    np.save(splits_dir / "nsl_kdd_test_indices.npy", test_idx)

    print(f"[NSL-KDD] Split: Train={len(train_idx)} (60%), Val={len(val_idx)} (15%), Test={len(test_idx)} (25%)")

    # Selected vs All Features
    selected_cols = [col for col in predictor_cols if col not in NSL_KDD_PAPER_REMOVED_FEATURES]
    print(f"[NSL-KDD] All features count: {len(predictor_cols)}")
    print(f"[NSL-KDD] Selected features count: {len(selected_cols)}")

    X_all = X_scaled_df[predictor_cols].values.astype(np.float32)
    X_selected = X_scaled_df[selected_cols].values.astype(np.float32)

    # Prepare 36-feature representation for 6x6 grid
    if X_selected.shape[1] == 36:
        X_selected_36 = X_selected
    else:
        pad_len = 36 - X_selected.shape[1]
        zero_pad = np.zeros((len(X_selected), pad_len), dtype=np.float32)
        X_selected_36 = np.hstack([X_selected, zero_pad])

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
        all_features=np.array(predictor_cols),
        selected_features=np.array(selected_cols),
        removed_features=np.array(NSL_KDD_PAPER_REMOVED_FEATURES)
    )

    print(f"[NSL-KDD] Successfully saved processed arrays to {output_dir / 'nsl_kdd_processed.npz'}")
    return {
        "n_samples": n_samples,
        "n_all_features": len(predictor_cols),
        "n_selected_features": len(selected_cols),
        "selected_features": selected_cols,
        "removed_features": NSL_KDD_PAPER_REMOVED_FEATURES
    }


if __name__ == "__main__":
    process_nsl_kdd()
