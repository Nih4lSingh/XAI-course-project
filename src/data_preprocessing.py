"""Paper-specific preprocessing for Sharma et al. (2024)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

NSL_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
NSL_CLASSES = ["DoS", "Normal", "Probe", "R2L", "U2R"]
UNSW_CLASSES = ["Normal", "Generic", "Exploits", "DoS", "Fuzzers"]
UNSW_CLASS_MAP = {name.lower(): name for name in UNSW_CLASSES}
NSL_DROPPED = {
    "srv_serror_rate", "dst_host_srv_rerror_rate", "num_root", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "srv_rerror_rate",
}
# The paper names "loss"; the standard CSV has sloss/dloss. Neither is removed.
UNSW_DROPPED = {"ct_src_dport_ltm", "dwin", "ct_ftp_cmd", "ct_srv_dst"}
NSL_ATTACKS = {
    "DoS": {"apache2", "back", "land", "mailbomb", "neptune", "pod", "processtable", "smurf", "teardrop", "udpstorm", "worm"},
    "Probe": {"ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"},
    "R2L": {"ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf", "sendmail", "snmpgetattack", "snmpguess", "spy", "warezclient", "warezmaster", "xlock", "xsnoop"},
    "U2R": {"buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm"},
}
NSL_ATTACK_MAP = {attack: category for category, attacks in NSL_ATTACKS.items() for attack in attacks}


@dataclass
class DatasetBundle:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    selected_features: list[str]
    class_names: list[str]
    metadata: dict[str, Any]


def _read_nsl(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] not in {42, 43}:
        raise ValueError(f"NSL-KDD file {path} has {frame.shape[1]} columns; expected 42 or 43.")
    frame.columns = NSL_FEATURES + ["label"] + (["difficulty"] if frame.shape[1] == 43 else [])
    return frame


def _read_unsw(paths: list[str]) -> pd.DataFrame:
    if not paths:
        raise ValueError("UNSW-NB15 paths are required.")
    frame = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in paths], ignore_index=True)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    return frame


def _paper_split(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    """Create the paper's random, stratified 60/15/25 split."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=0.25, stratify=labels, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.20, stratify=y_train_val, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def _label_encode_and_scale(frame: pd.DataFrame, categorical: list[str]) -> tuple[np.ndarray, list[str]]:
    encoded = frame.copy()
    for column in categorical:
        encoded[column] = LabelEncoder().fit_transform(encoded[column].astype(str))
    return MinMaxScaler().fit_transform(encoded).astype(np.float32), encoded.columns.tolist()


def _bundle(name: str, frame: pd.DataFrame, labels: pd.Series, categorical: list[str], class_order: list[str], seed: int, metadata: dict[str, Any]) -> DatasetBundle:
    label_encoder = LabelEncoder().fit(class_order)
    features, feature_names = _label_encode_and_scale(frame, categorical)
    X_train, y_train, X_val, y_val, X_test, y_test = _paper_split(features, label_encoder.transform(labels), seed)
    metadata.update({
        "protocol": "Sharma et al. (2024) multiclass protocol",
        "split": {"train": 0.60, "validation": 0.15, "test": 0.25, "stratified": True},
        "encoding": "LabelEncoder for categorical predictors and class labels",
        "normalization": "MinMaxScaler fit before paper-style random split",
        "selected_feature_count": len(feature_names), "seed": seed,
    })
    return DatasetBundle(name, X_train, y_train, X_val, y_val, X_test, y_test, feature_names, label_encoder.classes_.tolist(), metadata)


def load_nsl_kdd(train_path: str, test_path: str, seed: int) -> DatasetBundle:
    frame = pd.concat([_read_nsl(train_path), _read_nsl(test_path)], ignore_index=True)
    raw_labels = frame.pop("label").astype(str).str.strip().str.rstrip(".").str.lower()
    labels = raw_labels.map(NSL_ATTACK_MAP).fillna("Normal")
    unknown = sorted(set(raw_labels) - set(NSL_ATTACK_MAP) - {"normal"})
    if unknown:
        raise ValueError(f"Unmapped NSL-KDD attacks: {unknown}")
    present_drops = NSL_DROPPED & set(frame.columns)
    frame = frame.drop(columns=sorted(present_drops))
    if frame.shape[1] != 36:
        raise ValueError(f"Paper protocol requires 36 NSL-KDD inputs; found {frame.shape[1]}.")
    return _bundle("nsl", frame, labels, ["protocol_type", "service", "flag"], NSL_CLASSES, seed, {
        "source_rows": len(frame), "dropped_features": sorted(present_drops), "matrix_shape": [6, 6],
        "difficulty_feature": "retained to produce the paper's stated 36 inputs",
    })


def load_unsw_nb15(train_paths: list[str], test_paths: list[str], seed: int) -> DatasetBundle:
    frame = pd.concat([_read_unsw(train_paths), _read_unsw(test_paths)], ignore_index=True)
    categories = frame["attack_cat"].astype(str).str.strip().str.lower().map(UNSW_CLASS_MAP)
    selected: list[pd.DataFrame] = []
    for category in UNSW_CLASSES:
        subset = frame.loc[categories == category].copy()
        if category in {"Normal", "Generic"}:
            subset = subset.sample(n=50_000, random_state=seed)
        if subset.empty:
            raise ValueError(f"UNSW-NB15 has no rows for required paper class {category!r}.")
        selected.append(subset)
    frame = pd.concat(selected, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    labels = frame.pop("attack_cat").astype(str).str.strip().str.lower().map(UNSW_CLASS_MAP)
    frame = frame.drop(columns=[column for column in ("id", "label") if column in frame])
    present_drops = UNSW_DROPPED & set(frame.columns)
    frame = frame.drop(columns=sorted(present_drops))
    if frame.shape[1] != 38:
        raise ValueError(f"Paper protocol requires 38 UNSW-NB15 inputs; found {frame.shape[1]}.")
    return _bundle("unsw", frame, labels, ["proto", "service", "state"], UNSW_CLASSES, seed, {
        "source_rows": len(frame), "class_sampling": {"Normal": 50_000, "Generic": 50_000, "other_selected_classes": "all rows"},
        "dropped_features": ["loss", *sorted(present_drops)], "matrix_shape": [7, 7],
    })


def prepare_dataset(dataset: str, config: dict[str, Any]) -> DatasetBundle:
    if dataset == "nsl_kdd":
        paths = config["datasets"]["nsl_kdd"]
        return load_nsl_kdd(paths["train_path"], paths["test_path"], config["seed"])
    if dataset == "unsw_nb15":
        paths = config["datasets"]["unsw_nb15"]
        return load_unsw_nb15(paths["train_paths"], paths["test_paths"], config["seed"])
    raise ValueError("dataset must be nsl_kdd or unsw_nb15")
