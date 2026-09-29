"""
Data Preprocessing Pipeline for Sharma et al. (2024) Replication
Supports NSL-KDD and UNSW-NB15 intrusion detection benchmarks.

Methodology:
- Exact 5-class filtering and mapping
- 60/15/25 stratified split
- Top feature selection dropping 6 correlated/redundant features:
  * NSL-KDD: 41 -> 36 selected features (including difficulty) -> 6x6 spatial grid
  * UNSW-NB15: 42 -> 38 selected features + 11 zero padding -> 49 features -> 7x7 spatial grid
- MinMax normalization to [0, 1]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Feature column definitions
NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "difficulty"
]

NSL_ATTACK_MAP = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L",
    "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R"
}

# The 6 dropped predictors in Sharma et al. (2024)
NSL_DROP_COLUMNS = ["land", "urgent", "num_failed_logins", "root_shell", "su_attempted", "num_shells"]

UNSW_CLASSES = ["Normal", "Generic", "Exploits", "DoS", "Fuzzers"]
UNSW_DROP_COLUMNS = ["ct_src_dport_ltm", "loss", "dwin", "ct_ftp_cmd", "label", "ct_srv_dst", "id"]


def find_data_file(relative_candidates: List[Path]) -> Path:
    for cand in relative_candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Could not locate dataset file. Checked: {[str(c) for c in relative_candidates]}")


def load_nsl_kdd(
    data_dir: Path | None = None,
    mode: str = "flat",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Load, preprocess, and split NSL-KDD dataset.
    Returns: dict with X_train, y_train, X_val, y_val, X_test, y_test, and classes.
    """
    # Check if preprocessed npz exists
    processed_npz = PROJECT_ROOT / "data" / "processed" / "nsl_kdd" / "nsl_kdd_processed.npz"
    if processed_npz.is_file():
        d = np.load(processed_npz, allow_pickle=True)
        X_tr = d["X_train_selected"].astype(np.float32)
        X_va = d["X_val_selected"].astype(np.float32)
        X_te = d["X_test_selected"].astype(np.float32)
        y_tr = d["y_train"].astype(np.int64)
        y_va = d["y_val"].astype(np.int64)
        y_te = d["y_test"].astype(np.int64)
        classes = list(d["classes"])
    else:
        root = data_dir or (PROJECT_ROOT / "data" / "raw" / "nsl_kdd")
        train_path = find_data_file([root / "KDDTrain+.txt", PROJECT_ROOT / "datasets" / "KDDTrain+.txt"])
        test_path = find_data_file([root / "KDDTest+.txt", PROJECT_ROOT / "datasets" / "KDDTest+.txt"])
        
        train_df = pd.read_csv(train_path, header=None, names=NSL_COLUMNS)
        test_df = pd.read_csv(test_path, header=None, names=NSL_COLUMNS)
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        df["attack"] = df["attack"].astype(str).str.strip().str.lower().str.rstrip(".")
        df["target"] = df["attack"].map(NSL_ATTACK_MAP)
        df = df.dropna(subset=["target"]).reset_index(drop=True)
        
        X = df.drop(columns=["target", "attack"] + NSL_DROP_COLUMNS, errors="ignore")
        y = df["target"]
        
        X_tr_raw, X_tmp, y_tr_text, y_tmp = train_test_split(X, y, test_size=0.40, random_state=seed, stratify=y)
        X_va_raw, X_te_raw, y_va_text, y_te_text = train_test_split(X_tmp, y_tmp, test_size=0.625, random_state=seed, stratify=y_tmp)
        
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr_text)
        y_va = le.transform(y_va_text)
        y_te = le.transform(y_te_text)
        classes = le.classes_.tolist()
        
        cats = X_tr_raw.select_dtypes(include=["object"]).columns.tolist()
        nums = [c for c in X_tr_raw.columns if c not in cats]
        
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr_cat = oe.fit_transform(X_tr_raw[cats])
        X_va_cat = oe.transform(X_va_raw[cats])
        X_te_cat = oe.transform(X_te_raw[cats])
        
        X_tr_num = X_tr_raw[nums].apply(pd.to_numeric, errors="coerce").fillna(0).values
        X_va_num = X_va_raw[nums].apply(pd.to_numeric, errors="coerce").fillna(0).values
        X_te_num = X_te_raw[nums].apply(pd.to_numeric, errors="coerce").fillna(0).values
        
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(np.hstack([X_tr_num, X_tr_cat])).astype(np.float32)
        X_va = scaler.transform(np.hstack([X_va_num, X_va_cat])).astype(np.float32)
        X_te = scaler.transform(np.hstack([X_te_num, X_te_cat])).astype(np.float32)

    if mode == "1d":
        X_tr = X_tr[..., np.newaxis]
        X_va = X_va[..., np.newaxis]
        X_te = X_te[..., np.newaxis]
    elif mode == "2d":
        # 36 features -> 6x6x1
        X_tr = X_tr.reshape(-1, 6, 6, 1)
        X_va = X_va.reshape(-1, 6, 6, 1)
        X_te = X_te.reshape(-1, 6, 6, 1)

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val": X_va, "y_val": y_va,
        "X_test": X_te, "y_test": y_te,
        "classes": classes,
        "dataset": "NSL-KDD",
    }


def load_unsw_nb15(
    data_dir: Path | None = None,
    mode: str = "flat",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Load, preprocess, and split UNSW-NB15 dataset.
    Returns: dict with X_train, y_train, X_val, y_val, X_test, y_test, and classes.
    """
    processed_npz = PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "unsw_processed.npz"
    if processed_npz.is_file():
        d = np.load(processed_npz, allow_pickle=True)
        if mode == "2d":
            # Uses 49 padded features (7x7x1)
            X_tr = d["X_train_grid"].astype(np.float32)
            X_va = d["X_val_grid"].astype(np.float32)
            X_te = d["X_test_grid"].astype(np.float32)
        else:
            X_tr = d["X_train_selected"].astype(np.float32)
            X_va = d["X_val_selected"].astype(np.float32)
            X_te = d["X_test_selected"].astype(np.float32)
            if mode == "1d":
                X_tr = X_tr[..., np.newaxis]
                X_va = X_va[..., np.newaxis]
                X_te = X_te[..., np.newaxis]
        y_tr = d["y_train"].astype(np.int64)
        y_va = d["y_val"].astype(np.int64)
        y_te = d["y_test"].astype(np.int64)
        classes = list(d["classes"])
    else:
        root = data_dir or (PROJECT_ROOT / "data" / "raw" / "unsw_nb15")
        train_path = find_data_file([root / "UNSW_NB15_training-set.csv", PROJECT_ROOT / "datasets" / "UNSW_NB15_training-set.csv"])
        test_path = find_data_file([root / "UNSW_NB15_testing-set.csv", PROJECT_ROOT / "datasets" / "UNSW_NB15_testing-set.csv"])
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
        df = df[df["attack_cat"].isin(UNSW_CLASSES)].copy()
        
        # Balance cap at 50,000 per class matching notebook / paper
        parts = []
        for c in UNSW_CLASSES:
            cdf = df[df["attack_cat"] == c]
            if len(cdf) > 50000:
                cdf = cdf.sample(n=50000, random_state=seed)
            parts.append(cdf)
        df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        
        X = df.drop(columns=["attack_cat"] + UNSW_DROP_COLUMNS, errors="ignore")
        y = df["attack_cat"]
        
        X_tr_raw, X_tmp, y_tr_text, y_tmp = train_test_split(X, y, test_size=0.40, random_state=seed, stratify=y)
        X_va_raw, X_te_raw, y_va_text, y_te_text = train_test_split(X_tmp, y_tmp, test_size=0.625, random_state=seed, stratify=y_tmp)
        
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr_text)
        y_va = le.transform(y_va_text)
        y_te = le.transform(y_te_text)
        classes = le.classes_.tolist()
        
        cats = X_tr_raw.select_dtypes(include=["object"]).columns.tolist()
        nums = [c for c in X_tr_raw.columns if c not in cats]
        
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr_cat = oe.fit_transform(X_tr_raw[cats])
        X_va_cat = oe.transform(X_va_raw[cats])
        X_te_cat = oe.transform(X_te_raw[cats])
        
        X_tr_num = X_tr_raw[nums].apply(pd.to_numeric, errors="coerce").fillna(0).values
        X_va_num = X_va_raw[nums].apply(pd.to_numeric, errors="coerce").fillna(0).values
        X_te_num = X_te_raw[nums].apply(pd.to_numeric, errors="coerce").fillna(0).values
        
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(np.hstack([X_tr_num, X_tr_cat])).astype(np.float32)
        X_va = scaler.transform(np.hstack([X_va_num, X_va_cat])).astype(np.float32)
        X_te = scaler.transform(np.hstack([X_te_num, X_te_cat])).astype(np.float32)
        
        if mode == "1d":
            X_tr = X_tr[..., np.newaxis]
            X_va = X_va[..., np.newaxis]
            X_te = X_te[..., np.newaxis]
        elif mode == "2d":
            # 38 real + 11 zero padding = 49 -> 7x7x1
            pad_tr = np.zeros((len(X_tr), 11), dtype=np.float32)
            pad_va = np.zeros((len(X_va), 11), dtype=np.float32)
            pad_te = np.zeros((len(X_te), 11), dtype=np.float32)
            X_tr = np.hstack([X_tr, pad_tr]).reshape(-1, 7, 7, 1)
            X_va = np.hstack([X_va, pad_va]).reshape(-1, 7, 7, 1)
            X_te = np.hstack([X_te, pad_te]).reshape(-1, 7, 7, 1)

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val": X_va, "y_val": y_va,
        "X_test": X_te, "y_test": y_te,
        "classes": classes,
        "dataset": "UNSW-NB15",
    }


def get_dataset(dataset_name: str, mode: str = "flat", seed: int = 42) -> Dict[str, np.ndarray]:
    name = dataset_name.lower().replace("-", "_")
    if "nsl" in name:
        return load_nsl_kdd(mode=mode, seed=seed)
    elif "unsw" in name:
        return load_unsw_nb15(mode=mode, seed=seed)
    raise ValueError(f"Unknown dataset: {dataset_name}. Expected 'nsl_kdd' or 'unsw_nb15'.")


if __name__ == "__main__":
    print("Testing NSL-KDD Loading...")
    nsl = load_nsl_kdd(mode="2d")
    print(f"NSL-KDD: X_train shape: {nsl['X_train'].shape}, classes: {nsl['classes']}")
    
    print("Testing UNSW-NB15 Loading...")
    unsw = load_unsw_nb15(mode="2d")
    print(f"UNSW-NB15: X_train shape: {unsw['X_train'].shape}, classes: {unsw['classes']}")
    print("Data Preprocessing Module operational.")
