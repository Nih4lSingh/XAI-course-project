"""
UNSW-NB15 Preprocessing Module
Sharma et al. (2024) Replication Master Pipeline

Methodology:
1. Ingest UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv.
2. Filter to the 5 designated attack categories ('UNSW-NBnew'):
   0: DoS, 1: Exploits, 2: Fuzzers, 3: Generic, 4: Normal
3. Sampling Policy (Sharma et al. Section 3.1):
   - Cap 'Generic' at 50,000 samples (deterministic seed=42)
   - Cap 'Normal' at 50,000 samples (deterministic seed=42)
   - Retain all DoS (16,353), Exploits (44,525), and Fuzzers (24,246)
   - Total records = 185,124
   - Save sampling metadata to results/data_sampling/unsw_sampling_report.json
4. Target Separation:
   - Strictly exclude 'id', 'attack_cat', and 'label' from feature matrix X.
5. Deterministic Label Encoding on categorical features:
   ['proto', 'service', 'state']
6. Min-Max Normalization to [0, 1].
7. Stratified 60% Train / 15% Validation / 25% Test split (seed=42).
8. Feature Selection (|PCC| > 0.95):
   - Drop 4 redundant traffic predictors:
     ['ct_src_dport_ltm', 'dwin', 'ct_ftp_cmd', 'ct_srv_dst']
   - Retain 'sloss' and 'dloss' (resolving the paper's 'loss' label ambiguity)
   - Selected features: 38 real predictors + 11 zeros padding = 49 (7x7 grid)
   - All features: 42 predictors + 7 zeros padding = 49 (7x7 grid)
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
    UNSW_NB15_CLASS_MAPPING,
    UNSW_NB15_TARGET_MAP
)
from preprocessing.normalization import DeterministicMinMaxScaler


UNSW_NON_PREDICTORS = ["id", "attack_cat", "label"]
UNSW_CATEGORICAL_COLS = ["proto", "service", "state"]

# 4 redundant traffic predictors removed by Pearson correlation in Sharma et al. Section 3.2
# ('label' was the 5th column dropped in the paper's correlation table, handled via target separation)
# ('loss' in paper corresponds to retaining sloss & dloss since 'loss' is not in CSV schema)
UNSW_PAPER_REMOVED_PREDICTORS = [
    "ct_src_dport_ltm",
    "dwin",
    "ct_ftp_cmd",
    "ct_srv_dst"
]


def load_raw_unsw_nb15(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate raw training and testing partitions."""
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path = data_dir / "UNSW_NB15_testing-set.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing UNSW-NB15 raw CSVs in {data_dir}. Run data/download_datasets.py first.")

    df_train = pd.read_csv(train_path, encoding="utf-8-sig")
    df_test = pd.read_csv(test_path, encoding="utf-8-sig")
    df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    return df_combined


def apply_sampling_policy(
    df: pd.DataFrame,
    cap_generic: int = 50000,
    cap_normal: int = 50000,
    random_seed: int = 42,
    sampling_report_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Applies the paper's 50K capping policy to Generic and Normal classes,
    preserving all DoS, Exploits, and Fuzzers records (Total: 185,124).
    """
    counts_before = df["attack_clean"].value_counts().to_dict()

    sampled_dfs = []
    for cls_name, grp in df.groupby("attack_clean"):
        if cls_name == "generic" and len(grp) > cap_generic:
            sampled_dfs.append(grp.sample(n=cap_generic, random_state=random_seed))
        elif cls_name == "normal" and len(grp) > cap_normal:
            sampled_dfs.append(grp.sample(n=cap_normal, random_state=random_seed))
        else:
            sampled_dfs.append(grp)

    df_sampled = pd.concat(sampled_dfs, axis=0, ignore_index=True)
    # Shuffle deterministically to prevent grouped ordering
    df_sampled = df_sampled.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    counts_after = df_sampled["attack_clean"].value_counts().to_dict()

    if sampling_report_path is not None:
        sampling_report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "dataset": "UNSW-NB15",
            "total_records_before": len(df),
            "total_records_after": len(df_sampled),
            "sampling_policy": "Cap Generic and Normal at 50,000; retain all DoS, Exploits, and Fuzzers",
            "random_seed": random_seed,
            "counts_before": counts_before,
            "counts_after": counts_after
        }
        with open(sampling_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[UNSW-NB15] Saved sampling report to {sampling_report_path}")

    return df_sampled


def process_unsw_nb15(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
    sampling_report_dir: Optional[Path] = None,
    cap_generic: int = 50000,
    cap_normal: int = 50000,
    random_seed: int = 42
) -> Dict[str, Union[np.ndarray, List[str], int]]:
    """
    Executes the UNSW-NB15 preprocessing pipeline.
    """
    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "data" / "raw" / "unsw_nb15"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "processed" / "unsw_nb15"
    if splits_dir is None:
        splits_dir = PROJECT_ROOT / "splits"
    if sampling_report_dir is None:
        sampling_report_dir = PROJECT_ROOT / "results" / "data_sampling"

    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    sampling_report_dir.mkdir(parents=True, exist_ok=True)

    print("[UNSW-NB15] Loading raw data...")
    df = load_raw_unsw_nb15(raw_dir)
    print(f"[UNSW-NB15] Loaded raw shape: {df.shape}")

    # Standardize attack category strings
    df["attack_clean"] = df["attack_cat"].fillna("normal").astype(str).str.lower().str.strip()
    df["attack_clean"] = df["attack_clean"].replace({"backdoors": "backdoor"})

    # Filter strictly to the 5 target classes evaluated by Sharma et al.
    df_5class = df[df["attack_clean"].isin(UNSW_NB15_TARGET_MAP.keys())].copy()
    print(f"[UNSW-NB15] Filtered to 5 classes: {len(df_5class)} records (from {len(df)})")

    # Apply the paper's 50K capping policy
    df_capped = apply_sampling_policy(
        df_5class,
        cap_generic=cap_generic,
        cap_normal=cap_normal,
        random_seed=random_seed,
        sampling_report_path=sampling_report_dir / "unsw_sampling_report.json"
    )
    print(f"[UNSW-NB15] Post-capping record count: {len(df_capped)}")

    # Map target strings to integer IDs [0..4]
    df_capped["target"] = df_capped["attack_clean"].map(UNSW_NB15_TARGET_MAP).astype(int)
    y_full = df_capped["target"].values

    print("[UNSW-NB15] Target class distribution:")
    for cls_idx, cls_name in UNSW_NB15_CLASS_MAPPING.items():
        count = (y_full == cls_idx).sum()
        pct = count * 100.0 / len(y_full)
        print(f"  Class {cls_idx} ({cls_name}): {count} samples ({pct:.2f}%)")

    # Exclude non-predictors (Target Separation)
    predictor_cols = [
        c for c in df_capped.columns
        if c not in UNSW_NON_PREDICTORS and c not in ("attack_clean", "target")
    ]
    print(f"[UNSW-NB15] Total eligible traffic predictors: {len(predictor_cols)}")

    # Categorical Label Encoding
    X_df = df_capped[predictor_cols].copy()
    encoder_pipeline = CategoricalFeaturePipeline(UNSW_CATEGORICAL_COLS)
    X_encoded_df = encoder_pipeline.fit_transform(X_df)
    encoder_pipeline.save(output_dir / "unsw_encoders.json")

    # Min-Max Normalization to [0, 1]
    scaler = DeterministicMinMaxScaler()
    X_scaled_df = scaler.fit_transform(X_encoded_df)
    scaler.save(output_dir / "unsw_scaler.json")

    # Deterministic Stratified 60% Train / 15% Val / 25% Test Split
    n_samples = len(df_capped)
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

    np.save(splits_dir / "unsw_train_indices.npy", train_idx)
    np.save(splits_dir / "unsw_val_indices.npy", val_idx)
    np.save(splits_dir / "unsw_test_indices.npy", test_idx)

    print(f"[UNSW-NB15] Split sizes -> Train: {len(train_idx)} (60%), Val: {len(val_idx)} (15%), Test: {len(test_idx)} (25%)")

    # Feature Selection: remove 4 redundant traffic predictors
    removed_cols = [col for col in UNSW_PAPER_REMOVED_PREDICTORS if col in predictor_cols]
    selected_cols = [col for col in predictor_cols if col not in removed_cols]

    print(f"[UNSW-NB15] Removed predictors ({len(removed_cols)}): {removed_cols}")
    print(f"[UNSW-NB15] Selected predictors ({len(selected_cols)}): {selected_cols}")

    X_all = X_scaled_df[predictor_cols].values.astype(np.float32)
    X_selected = X_scaled_df[selected_cols].values.astype(np.float32)

    # 49-feature zero-padded representation for 7x7 2D-CNN (38 features + 11 zeros)
    pad_len = 49 - X_selected.shape[1]
    if pad_len > 0:
        zero_pad = np.zeros((len(X_selected), pad_len), dtype=np.float32)
        X_selected_49 = np.hstack([X_selected, zero_pad])
    else:
        X_selected_49 = X_selected[:, :49]

    # For all-features 2D-CNN, pad from 42 to 49 (7 zeros)
    pad_len_all = 49 - X_all.shape[1]
    if pad_len_all > 0:
        zero_pad_all = np.zeros((len(X_all), pad_len_all), dtype=np.float32)
        X_all_49 = np.hstack([X_all, zero_pad_all])
    else:
        X_all_49 = X_all[:, :49]

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
    process_unsw_nb15()
