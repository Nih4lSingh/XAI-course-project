#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sharma et al. (2024) — 1D-CNN replication
Native Windows CUDA / PyTorch / 64 seeds × 2 datasets = 128 models

This ports the supplied Google-Colab/TensorFlow notebook to the local NVIDIA GPU
without changing its experiment definition.

PRESERVED FROM THE NOTEBOOK
---------------------------
Data:
  - NSL-KDD: combine KDDTrain+ + KDDTest+
  - UNSW-NB15: combine training-set + testing-set
  - fixed 60% train / 15% validation / 25% test split, seed 42
  - exact notebook feature removals, label mappings, class sampling,
    OrdinalEncoder behavior and MinMaxScaler behavior

1D-CNN:
  Conv1D(64, kernel=3, padding="same") + ReLU
  MaxPool1D(pool=2)
  Conv1D(32, kernel=3, padding="same") + ReLU
  MaxPool1D(pool=2)
  Conv1D(32, kernel=3, padding="same") + ReLU
  MaxPool1D(pool=2)
  Flatten
  Dense(5) / Softmax-equivalent cross entropy

Training:
  - 64 model seeds: 1..64
  - fixed data split seed: 42
  - epochs: 20
  - batch size: 128
  - learning rate: 0.001
  - AdamW weight decay: 0.0001
  - dropout: 0
  - FP32

GPU IMPLEMENTATION
------------------
The 64 independent models for each dataset are represented as one vectorized
parameter bank. The NSL-KDD and UNSW-NB15 banks execute concurrently on
separate CUDA streams. Thus all 128 independent seed×dataset models are active
concurrently without creating 128 Python processes.

Weights and optimizer states are independent for every seed. No model shares
parameters with another model.

Outputs:
  outputs_1dcnn_gpu/
    epoch_log.csv
    metrics_per_seed.csv
    metrics_summary.csv
    metrics.csv
    confusion_matrix_*_seed64.png
    *_accuracy_across_seeds.png
    *_f1_macro_across_seeds.png
    classification_report_*_seed64.txt
    model_*_seed64.pt

  dnn_1dcnn_results_gpu.zip
"""

from __future__ import annotations

# Set deterministic cuBLAS workspace config BEFORE importing torch.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# 0. EXPERIMENT CONFIGURATION — PRESERVED FROM THE NOTEBOOK
# =============================================================================

MODEL_SEEDS = list(range(1, 65))
DATA_SPLIT_SEED = 42

N_CLASSES = 5

CONV_FILTERS = [64, 32, 32]  # notebook [ASSUMPTION]
KERNEL_SIZE = 3               # [PAPER]
POOL_SIZE = 2                 # [PAPER]
ACTIVATION = "relu"           # [PAPER]

LEARNING_RATE = 0.001         # [PAPER]
WEIGHT_DECAY = 0.0001         # [PAPER]
EPOCHS = 20                   # [PAPER]
DROPOUT_RATE = 0.0            # [PAPER]
BATCH_SIZE = 128              # notebook [ASSUMPTION]

TRAIN_SIZE = 0.60
VAL_SIZE = 0.15
TEST_SIZE = 0.25

# Match Keras AdamW defaults as closely as practical.
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-7

TOTAL_MODELS = len(MODEL_SEEDS) * 2
TOTAL_MODEL_EPOCHS = TOTAL_MODELS * EPOCHS

assert abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) < 1e-9
assert DROPOUT_RATE == 0.0


# =============================================================================
# 1. LOCAL PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Your current directory layout:
# C:\Users\super\Downloads\nihalxai\
#   datasets\datasets\
#       KDDTrain+.txt
#       KDDTest+.txt
#       UNSW_NB15_training-set.csv
#       UNSW_NB15_testing-set.csv
#
# Flexible dataset path resolution
def resolve_data_paths() -> tuple[Path, Path, Path, Path]:
    nsl_train, nsl_test = None, None
    unsw_train, unsw_test = None, None

    nsl_dirs = [
        SCRIPT_DIR / "data" / "raw" / "nsl_kdd",
        SCRIPT_DIR / "datasets" / "datasets",
        SCRIPT_DIR / "datasets",
        SCRIPT_DIR,
    ]
    for d in nsl_dirs:
        if (d / "KDDTrain+.txt").is_file() and (d / "KDDTest+.txt").is_file():
            nsl_train = d / "KDDTrain+.txt"
            nsl_test = d / "KDDTest+.txt"
            break

    unsw_dirs = [
        SCRIPT_DIR / "data" / "raw" / "unsw_nb15",
        SCRIPT_DIR / "datasets" / "datasets",
        SCRIPT_DIR / "datasets",
        SCRIPT_DIR,
    ]
    for d in unsw_dirs:
        if (d / "UNSW_NB15_training-set.csv").is_file() and (d / "UNSW_NB15_testing-set.csv").is_file():
            unsw_train = d / "UNSW_NB15_training-set.csv"
            unsw_test = d / "UNSW_NB15_testing-set.csv"
            break

    if not (nsl_train and nsl_test and unsw_train and unsw_test):
        raise FileNotFoundError(
            "Could not locate all dataset files. Checked:\n"
            f"  - {SCRIPT_DIR / 'data' / 'raw'}\n"
            f"  - {SCRIPT_DIR / 'datasets'}\n"
            f"  - {SCRIPT_DIR}"
        )
    return nsl_train, nsl_test, unsw_train, unsw_test


NSL_KDD_TRAIN_PATH, NSL_KDD_TEST_PATH, UNSW_TRAIN_PATH, UNSW_TEST_PATH = resolve_data_paths()

OUTPUT_DIR = SCRIPT_DIR / "results" / "1d_cnn_gpu"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_LOG_PATH = OUTPUT_DIR / "epoch_log.csv"
PER_SEED_PATH = OUTPUT_DIR / "metrics_per_seed.csv"
SUMMARY_PATH = OUTPUT_DIR / "metrics_summary.csv"
ERROR_PATH = OUTPUT_DIR / "training_errors.csv"


# =============================================================================
# 2. CUDA / REPRODUCIBILITY
# =============================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available in this PyTorch environment.\n"
        f"PyTorch: {torch.__version__}\n"
        "Use the CUDA-enabled PyTorch environment that sees your RTX 5070 Ti."
    )

DEVICE = torch.device("cuda:0")
torch.cuda.set_device(DEVICE)

GPU_NAME = torch.cuda.get_device_name(0)

# Original notebook explicitly asks TensorFlow for deterministic ops.
# Preserve that intent where PyTorch supports it.
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

try:
    torch.backends.cuda.matmul.allow_tf32 = False
except Exception:
    pass

try:
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass

np.random.seed(DATA_SPLIT_SEED)
torch.manual_seed(DATA_SPLIT_SEED)
torch.cuda.manual_seed_all(DATA_SPLIT_SEED)


# =============================================================================
# 3. SHARED PROGRESS / LOGGING HELPERS
# =============================================================================

PROGRESS_LOCK = threading.RLock()
CSV_LOCK = threading.RLock()

experiment_start = 0.0
overall_bar = None
dataset_bars = {}


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def query_gpu_stats() -> str:
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ]
        line = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip().splitlines()[0]

        util, used, total, watts, temp = [
            x.strip() for x in line.split(",")
        ]

        return (
            f"GPU {util}% | VRAM {used}/{total} MiB | "
            f"{watts} W | {temp} C"
        )
    except Exception:
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return f"VRAM alloc/res {alloc:.0f}/{reserved:.0f} MiB"


def append_epoch_rows(rows: list[dict]) -> None:
    if not rows:
        return

    with CSV_LOCK:
        frame = pd.DataFrame(rows)
        frame.to_csv(
            EPOCH_LOG_PATH,
            mode="a",
            header=not EPOCH_LOG_PATH.exists(),
            index=False,
        )


def update_progress(
    dataset_name: str,
    epoch: int,
    epoch_seconds: float,
    train_acc: np.ndarray,
    val_acc: np.ndarray,
    n_train: int,
) -> None:
    elapsed = time.perf_counter() - experiment_start

    completed_this_update = len(MODEL_SEEDS)
    model_epochs_per_second = (
        completed_this_update / max(epoch_seconds, 1e-9)
    )
    train_samples_per_second = (
        n_train * len(MODEL_SEEDS) / max(epoch_seconds, 1e-9)
    )

    gpu_stats = query_gpu_stats()

    with PROGRESS_LOCK:
        dataset_bars[dataset_name].update(1)
        dataset_bars[dataset_name].set_postfix_str(
            f"train={train_acc.mean():.4f} "
            f"val={val_acc.mean():.4f} "
            f"{epoch_seconds:.2f}s"
        )

        overall_bar.update(completed_this_update)

        global_rate = overall_bar.n / max(elapsed, 1e-9)
        overall_bar.set_postfix_str(
            f"{global_rate:.2f} model-epochs/s | {gpu_stats}"
        )

        tqdm.write(
            f"[{dataset_name:9s}] "
            f"epoch {epoch:02d}/{EPOCHS} | "
            f"+{completed_this_update} model-epochs | "
            f"train mean/min/max "
            f"{train_acc.mean():.5f}/{train_acc.min():.5f}/{train_acc.max():.5f} | "
            f"val mean/min/max "
            f"{val_acc.mean():.5f}/{val_acc.min():.5f}/{val_acc.max():.5f} | "
            f"{epoch_seconds:.2f}s | "
            f"{model_epochs_per_second:.2f} model-epochs/s | "
            f"{train_samples_per_second / 1e6:.3f}M train-samples/s | "
            f"elapsed {fmt_time(elapsed)} | "
            f"{gpu_stats}"
        )


# =============================================================================
# 4. NSL-KDD PREPROCESSING — EXACT NOTEBOOK LOGIC
# =============================================================================

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "class", "difficulty",
]

NSL_LABEL_COLUMN = "class"

NSL_DROP_FEATURES = [
    "srv_serror_rate",
    "dst_host_srv_rerror_rate",
    "num_root",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "srv_rerror_rate",
]

NSL_ATTACK_MAP = {
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "mailbomb": "DoS",
    "worm": "DoS",

    "ipsweep": "Probe",
    "mscan": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "saint": "Probe",
    "satan": "Probe",

    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "named": "R2L",
    "sendmail": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "httptunnel": "R2L",

    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",
}


def read_nsl_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)

    if df.shape[1] != len(NSL_KDD_COLUMNS):
        df = pd.read_csv(path)

    if df.shape[1] != len(NSL_KDD_COLUMNS):
        raise ValueError(
            f"{path} has {df.shape[1]} columns; expected "
            f"{len(NSL_KDD_COLUMNS)}."
        )

    df.columns = NSL_KDD_COLUMNS
    return df


def numeric_part(
    data: pd.DataFrame,
    numerical_columns: list[str],
) -> pd.DataFrame:
    return (
        data[numerical_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def preprocess_nsl_kdd() -> dict:
    print("\nLoading NSL-KDD...")

    nsl_train_raw = read_nsl_file(NSL_KDD_TRAIN_PATH)
    nsl_test_raw = read_nsl_file(NSL_KDD_TEST_PATH)

    nsl_df = pd.concat(
        [nsl_train_raw, nsl_test_raw],
        ignore_index=True,
    )

    print("KDDTrain+ shape:", nsl_train_raw.shape)
    print("KDDTest+ shape :", nsl_test_raw.shape)
    print("Combined NSL-KDD shape:", nsl_df.shape)

    nsl_df[NSL_LABEL_COLUMN] = (
        nsl_df[NSL_LABEL_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.rstrip(".")
    )

    def map_nsl_label(label):
        if label == "normal":
            return "Normal"
        return NSL_ATTACK_MAP.get(label, None)

    nsl_df["target"] = nsl_df[NSL_LABEL_COLUMN].apply(map_nsl_label)

    unknown = nsl_df.loc[
        nsl_df["target"].isna(),
        NSL_LABEL_COLUMN,
    ].unique()

    if len(unknown) > 0:
        print("Unmapped labels:", unknown)
        nsl_df = nsl_df.dropna(subset=["target"]).copy()

    columns_to_drop = [
        NSL_LABEL_COLUMN,
        "target",
        "difficulty",
    ]
    columns_to_drop += [
        col for col in NSL_DROP_FEATURES
        if col in nsl_df.columns
    ]

    X = nsl_df.drop(
        columns=list(set(columns_to_drop)),
        errors="ignore",
    )
    y = nsl_df["target"].copy()

    print("Features after paper-reported removals:", X.shape[1])

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=DATA_SPLIT_SEED,
        stratify=y,
    )

    val_fraction = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction,
        random_state=DATA_SPLIT_SEED,
        stratify=y_temp,
    )

    categorical_columns = [
        col for col in X_train.columns
        if X_train[col].dtype == "object"
    ]
    numerical_columns = [
        col for col in X_train.columns
        if col not in categorical_columns
    ]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    X_train_cat = encoder.fit_transform(
        X_train[categorical_columns]
    )
    X_val_cat = encoder.transform(
        X_val[categorical_columns]
    )
    X_test_cat = encoder.transform(
        X_test[categorical_columns]
    )

    X_train_num = numeric_part(X_train, numerical_columns)
    X_val_num = numeric_part(X_val, numerical_columns)
    X_test_num = numeric_part(X_test, numerical_columns)

    X_train_processed = np.hstack(
        [X_train_num.values, X_train_cat]
    )
    X_val_processed = np.hstack(
        [X_val_num.values, X_val_cat]
    )
    X_test_processed = np.hstack(
        [X_test_num.values, X_test_cat]
    )

    scaler = MinMaxScaler()
    X_train_processed = scaler.fit_transform(X_train_processed)
    X_val_processed = scaler.transform(X_val_processed)
    X_test_processed = scaler.transform(X_test_processed)

    class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    class_to_id = {
        name: idx
        for idx, name in enumerate(class_names)
    }

    y_train_encoded = (
        y_train.map(class_to_id).astype(np.int64).values
    )
    y_val_encoded = (
        y_val.map(class_to_id).astype(np.int64).values
    )
    y_test_encoded = (
        y_test.map(class_to_id).astype(np.int64).values
    )

    # Keras notebook shape is [N, length, channels].
    # PyTorch Conv1d shape is [N, channels, length].
    X_train_cnn = np.asarray(
        X_train_processed[:, np.newaxis, :],
        dtype=np.float32,
    )
    X_val_cnn = np.asarray(
        X_val_processed[:, np.newaxis, :],
        dtype=np.float32,
    )
    X_test_cnn = np.asarray(
        X_test_processed[:, np.newaxis, :],
        dtype=np.float32,
    )

    return {
        "name": "NSL-KDD",
        "X_train": X_train_cnn,
        "X_val": X_val_cnn,
        "X_test": X_test_cnn,
        "y_train": np.asarray(y_train_encoded, dtype=np.int64),
        "y_val": np.asarray(y_val_encoded, dtype=np.int64),
        "y_test": np.asarray(y_test_encoded, dtype=np.int64),
        "classes": class_names,
        "feature_count": X_train_processed.shape[1],
    }


# =============================================================================
# 5. UNSW-NB15 PREPROCESSING — EXACT NOTEBOOK LOGIC
# =============================================================================

UNSW_SELECTED_CLASSES = [
    "normal",
    "generic",
    "exploits",
    "dos",
    "fuzzers",
]

UNSW_DROP_FEATURES = [
    "ct_src_dport_ltm",
    "loss",
    "dwin",
    "ct_ftp_cmd",
    "label",
    "ct_srv_dst",
]


def read_unsw_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]
    return df


def preprocess_unsw_nb15() -> dict:
    print("\nLoading UNSW-NB15...")

    unsw_train_raw = read_unsw_file(UNSW_TRAIN_PATH)
    unsw_test_raw = read_unsw_file(UNSW_TEST_PATH)

    unsw_df = pd.concat(
        [unsw_train_raw, unsw_test_raw],
        ignore_index=True,
    )

    print("UNSW training-set shape:", unsw_train_raw.shape)
    print("UNSW testing-set shape :", unsw_test_raw.shape)
    print("Combined UNSW-NB15 shape:", unsw_df.shape)

    if "attack_cat" not in unsw_df.columns:
        raise ValueError("Expected an 'attack_cat' column.")

    unsw_df["attack_cat"] = (
        unsw_df["attack_cat"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unsw_selected = unsw_df[
        unsw_df["attack_cat"].isin(UNSW_SELECTED_CLASSES)
    ].copy()

    parts = []

    for class_name in UNSW_SELECTED_CLASSES:
        group = unsw_selected[
            unsw_selected["attack_cat"] == class_name
        ]

        # Exact notebook rule: cap any selected class above 50,000.
        if len(group) > 50_000:
            group = group.sample(
                n=50_000,
                random_state=DATA_SPLIT_SEED,
            )

        parts.append(group)

    unsw_selected = (
        pd.concat(parts)
        .sample(
            frac=1.0,
            random_state=DATA_SPLIT_SEED,
        )
        .reset_index(drop=True)
    )

    print("Selected class counts:")
    print(unsw_selected["attack_cat"].value_counts())

    columns_to_drop = ["attack_cat"] + [
        col for col in UNSW_DROP_FEATURES
        if col in unsw_selected.columns
    ]

    X = unsw_selected.drop(
        columns=columns_to_drop,
        errors="ignore",
    )
    y = unsw_selected["attack_cat"].copy()

    if "id" in X.columns:
        X = X.drop(columns=["id"])

    print("Features after paper-reported removals:", X.shape[1])

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=DATA_SPLIT_SEED,
        stratify=y,
    )

    val_fraction = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction,
        random_state=DATA_SPLIT_SEED,
        stratify=y_temp,
    )

    categorical_columns = [
        col for col in X_train.columns
        if X_train[col].dtype == "object"
    ]
    numerical_columns = [
        col for col in X_train.columns
        if col not in categorical_columns
    ]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    X_train_cat = encoder.fit_transform(
        X_train[categorical_columns]
    )
    X_val_cat = encoder.transform(
        X_val[categorical_columns]
    )
    X_test_cat = encoder.transform(
        X_test[categorical_columns]
    )

    X_train_num = numeric_part(X_train, numerical_columns)
    X_val_num = numeric_part(X_val, numerical_columns)
    X_test_num = numeric_part(X_test, numerical_columns)

    X_train_processed = np.hstack(
        [X_train_num.values, X_train_cat]
    )
    X_val_processed = np.hstack(
        [X_val_num.values, X_val_cat]
    )
    X_test_processed = np.hstack(
        [X_test_num.values, X_test_cat]
    )

    scaler = MinMaxScaler()
    X_train_processed = scaler.fit_transform(X_train_processed)
    X_val_processed = scaler.transform(X_val_processed)
    X_test_processed = scaler.transform(X_test_processed)

    class_names = [
        "Normal",
        "Generic",
        "Exploits",
        "DoS",
        "Fuzzers",
    ]

    class_to_id = {
        name.lower(): idx
        for idx, name in enumerate(class_names)
    }

    y_train_encoded = (
        y_train.map(class_to_id).astype(np.int64).values
    )
    y_val_encoded = (
        y_val.map(class_to_id).astype(np.int64).values
    )
    y_test_encoded = (
        y_test.map(class_to_id).astype(np.int64).values
    )

    X_train_cnn = np.asarray(
        X_train_processed[:, np.newaxis, :],
        dtype=np.float32,
    )
    X_val_cnn = np.asarray(
        X_val_processed[:, np.newaxis, :],
        dtype=np.float32,
    )
    X_test_cnn = np.asarray(
        X_test_processed[:, np.newaxis, :],
        dtype=np.float32,
    )

    return {
        "name": "UNSW-NB15",
        "X_train": X_train_cnn,
        "X_val": X_val_cnn,
        "X_test": X_test_cnn,
        "y_train": np.asarray(y_train_encoded, dtype=np.int64),
        "y_val": np.asarray(y_val_encoded, dtype=np.int64),
        "y_test": np.asarray(y_test_encoded, dtype=np.int64),
        "classes": class_names,
        "feature_count": X_train_processed.shape[1],
    }


# =============================================================================
# 6. 64 INDEPENDENT 1D-CNNS IN ONE VECTORIZED BANK
# =============================================================================

def pooled_length(length: int, pool_size: int = 2) -> int:
    # Keras MaxPooling1D(pool_size=2) defaults:
    # strides=pool_size, padding="valid".
    return ((length - pool_size) // pool_size) + 1


class CNNBank(nn.Module):
    """
    64 completely independent copies of the notebook's 1D-CNN.

    Tensor shapes:
        input: [models, batch, channels=1, length]
    """

    def __init__(
        self,
        input_length: int,
        seeds: list[int],
    ):
        super().__init__()

        self.input_length = int(input_length)
        self.seeds = list(seeds)
        self.n_models = len(seeds)

        c1, c2, c3 = CONV_FILTERS

        # [model, out_channels, in_channels, kernel]
        self.w1 = nn.Parameter(
            torch.empty(self.n_models, c1, 1, KERNEL_SIZE)
        )
        self.b1 = nn.Parameter(
            torch.zeros(self.n_models, c1)
        )

        self.w2 = nn.Parameter(
            torch.empty(self.n_models, c2, c1, KERNEL_SIZE)
        )
        self.b2 = nn.Parameter(
            torch.zeros(self.n_models, c2)
        )

        self.w3 = nn.Parameter(
            torch.empty(self.n_models, c3, c2, KERNEL_SIZE)
        )
        self.b3 = nn.Parameter(
            torch.zeros(self.n_models, c3)
        )

        l1 = pooled_length(self.input_length, POOL_SIZE)
        l2 = pooled_length(l1, POOL_SIZE)
        l3 = pooled_length(l2, POOL_SIZE)

        self.final_length = l3
        self.flatten_dim = c3 * l3

        # Store dense kernel Keras-style [model, in, out].
        self.w4 = nn.Parameter(
            torch.empty(
                self.n_models,
                self.flatten_dim,
                N_CLASSES,
            )
        )
        self.b4 = nn.Parameter(
            torch.zeros(self.n_models, 1, N_CLASSES)
        )

        self._initialize_per_seed()

    @staticmethod
    def _glorot_uniform_(
        tensor: torch.Tensor,
        generator: torch.Generator,
        fan_in: int,
        fan_out: int,
    ) -> None:
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        tensor.uniform_(
            -bound,
            bound,
            generator=generator,
        )

    def _initialize_per_seed(self) -> None:
        """
        Keras Conv1D/Dense defaults are GlorotUniform + zero bias.

        Each model gets one deterministic RNG seeded by its training seed,
        then its four layers are initialized sequentially.
        """
        c1, c2, c3 = CONV_FILTERS

        with torch.no_grad():
            for model_idx, seed in enumerate(self.seeds):
                g = torch.Generator(device="cpu")
                g.manual_seed(int(seed))

                # Conv1D fan_in/fan_out include receptive field size.
                self._glorot_uniform_(
                    self.w1[model_idx],
                    g,
                    fan_in=1 * KERNEL_SIZE,
                    fan_out=c1 * KERNEL_SIZE,
                )
                self._glorot_uniform_(
                    self.w2[model_idx],
                    g,
                    fan_in=c1 * KERNEL_SIZE,
                    fan_out=c2 * KERNEL_SIZE,
                )
                self._glorot_uniform_(
                    self.w3[model_idx],
                    g,
                    fan_in=c2 * KERNEL_SIZE,
                    fan_out=c3 * KERNEL_SIZE,
                )
                self._glorot_uniform_(
                    self.w4[model_idx],
                    g,
                    fan_in=self.flatten_dim,
                    fan_out=N_CLASSES,
                )

            self.b1.zero_()
            self.b2.zero_()
            self.b3.zero_()
            self.b4.zero_()

    def _grouped_conv1d(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply one different convolution kernel set per model using groups.

        x:
            [M, B, Cin, L]

        PyTorch grouped-conv input:
            [B, M*Cin, L]

        weight:
            [M*Cout, Cin, K]

        output:
            [M, B, Cout, L]
        """
        m, b, cin, length = x.shape
        cout = weight.shape[1]

        merged = (
            x.permute(1, 0, 2, 3)
            .contiguous()
            .view(b, m * cin, length)
        )

        merged_weight = weight.reshape(
            m * cout,
            cin,
            KERNEL_SIZE,
        )
        merged_bias = bias.reshape(m * cout)

        out = F.conv1d(
            merged,
            merged_weight,
            merged_bias,
            stride=1,
            padding=KERNEL_SIZE // 2,
            groups=m,
        )

        out_length = out.shape[-1]

        return (
            out.view(b, m, cout, out_length)
            .permute(1, 0, 2, 3)
            .contiguous()
        )

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        m, b, channels, length = x.shape

        merged = x.reshape(
            m * b,
            channels,
            length,
        )

        merged = F.max_pool1d(
            merged,
            kernel_size=POOL_SIZE,
            stride=POOL_SIZE,
            padding=0,
        )

        return merged.view(
            m,
            b,
            channels,
            merged.shape[-1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1: same padding -> ReLU -> pool2.
        x = self._grouped_conv1d(
            x,
            self.w1,
            self.b1,
        )
        x = F.relu(x)
        x = self._pool(x)

        # Conv2.
        x = self._grouped_conv1d(
            x,
            self.w2,
            self.b2,
        )
        x = F.relu(x)
        x = self._pool(x)

        # Conv3.
        x = self._grouped_conv1d(
            x,
            self.w3,
            self.b3,
        )
        x = F.relu(x)
        x = self._pool(x)

        # Keras Conv1D representation before Flatten is [B, L, C].
        # Preserve the same Flatten ordering rather than flattening PyTorch's
        # [B, C, L] memory order.
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(
            self.n_models,
            x.shape[1],
            self.flatten_dim,
        )

        logits = torch.bmm(x, self.w4) + self.b4
        return logits

    def seed64_state_dict(self) -> dict:
        idx = self.seeds.index(64)

        return {
            "conv1.weight": self.w1[idx].detach().cpu().contiguous(),
            "conv1.bias": self.b1[idx].detach().cpu().contiguous(),
            "conv2.weight": self.w2[idx].detach().cpu().contiguous(),
            "conv2.bias": self.b2[idx].detach().cpu().contiguous(),
            "conv3.weight": self.w3[idx].detach().cpu().contiguous(),
            "conv3.bias": self.b3[idx].detach().cpu().contiguous(),
            "output.weight": (
                self.w4[idx]
                .detach()
                .cpu()
                .T
                .contiguous()
            ),
            "output.bias": (
                self.b4[idx, 0]
                .detach()
                .cpu()
                .contiguous()
            ),
        }


class SingleCNN(nn.Module):
    """
    Standard single-model form corresponding to the bank.
    Provided so seed-64 checkpoints are easy to load later.
    """

    def __init__(self, input_length: int):
        super().__init__()

        c1, c2, c3 = CONV_FILTERS

        self.conv1 = nn.Conv1d(
            1,
            c1,
            KERNEL_SIZE,
            padding=KERNEL_SIZE // 2,
        )
        self.conv2 = nn.Conv1d(
            c1,
            c2,
            KERNEL_SIZE,
            padding=KERNEL_SIZE // 2,
        )
        self.conv3 = nn.Conv1d(
            c2,
            c3,
            KERNEL_SIZE,
            padding=KERNEL_SIZE // 2,
        )

        l1 = pooled_length(input_length, POOL_SIZE)
        l2 = pooled_length(l1, POOL_SIZE)
        l3 = pooled_length(l2, POOL_SIZE)

        self.final_length = l3
        self.output = nn.Linear(
            c3 * l3,
            N_CLASSES,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool1d(
            F.relu(self.conv1(x)),
            POOL_SIZE,
        )
        x = F.max_pool1d(
            F.relu(self.conv2(x)),
            POOL_SIZE,
        )
        x = F.max_pool1d(
            F.relu(self.conv3(x)),
            POOL_SIZE,
        )

        # Same Keras Flatten ordering [B, L, C].
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.shape[0], -1)

        return self.output(x)


# =============================================================================
# 7. OPTIMIZER
# =============================================================================

def make_optimizer(model: nn.Module):
    kwargs = dict(
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )

    # Fused AdamW changes only the execution kernel, not the optimizer
    # hyperparameters or mathematical update.
    try:
        return torch.optim.AdamW(
            model.parameters(),
            fused=True,
            **kwargs,
        )
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(
            model.parameters(),
            **kwargs,
        )


# =============================================================================
# 8. TRAIN / VALIDATE / TEST A 64-MODEL BANK
# =============================================================================

def build_per_seed_permutations(
    n_samples: int,
    rngs: list[np.random.Generator],
) -> torch.Tensor:
    indices = np.empty(
        (len(rngs), n_samples),
        dtype=np.int64,
    )

    for i, rng in enumerate(rngs):
        indices[i] = rng.permutation(n_samples)

    return torch.from_numpy(indices)


@torch.no_grad()
def evaluate_bank(
    bank: CNNBank,
    X: torch.Tensor,
    y: torch.Tensor,
    collect_predictions: bool = False,
):
    bank.eval()

    n_models = bank.n_models
    n_samples = X.shape[0]

    total_loss = torch.zeros(
        n_models,
        device=DEVICE,
        dtype=torch.float32,
    )
    total_correct = torch.zeros(
        n_models,
        device=DEVICE,
        dtype=torch.long,
    )

    predictions = [] if collect_predictions else None

    for start in range(0, n_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_samples)
        bs = end - start

        xb = X[start:end]
        yb = y[start:end]

        xb_bank = (
            xb.unsqueeze(0)
            .expand(n_models, -1, -1, -1)
        )

        logits = bank(xb_bank)

        targets = (
            yb.unsqueeze(0)
            .expand(n_models, -1)
        )

        element_loss = F.cross_entropy(
            logits.reshape(-1, N_CLASSES),
            targets.reshape(-1),
            reduction="none",
        ).view(n_models, bs)

        total_loss += element_loss.sum(dim=1)

        pred = logits.argmax(dim=-1)
        total_correct += (
            pred == targets
        ).sum(dim=1)

        if collect_predictions:
            predictions.append(
                pred.detach().cpu()
            )

    avg_loss = (
        total_loss / n_samples
    ).detach().cpu().numpy()

    accuracy = (
        total_correct.float() / n_samples
    ).detach().cpu().numpy()

    if collect_predictions:
        pred_array = torch.cat(
            predictions,
            dim=1,
        ).numpy()
    else:
        pred_array = None

    return avg_loss, accuracy, pred_array


def train_bank(dataset: dict) -> dict:
    torch.cuda.set_device(DEVICE)

    dataset_start = time.perf_counter()

    name = dataset["name"]

    X_train_np = dataset["X_train"]
    y_train_np = dataset["y_train"]

    X_val_np = dataset["X_val"]
    y_val_np = dataset["y_val"]

    X_test_np = dataset["X_test"]
    y_test_np = dataset["y_test"]

    classes = dataset["classes"]
    input_length = int(dataset["feature_count"])

    n_train = len(X_train_np)

    # One stream per dataset bank.
    stream = torch.cuda.Stream(device=DEVICE)

    with torch.cuda.stream(stream):
        X_train = torch.from_numpy(
            X_train_np
        ).to(DEVICE, non_blocking=True)
        y_train = torch.from_numpy(
            y_train_np
        ).to(DEVICE, non_blocking=True)

        X_val = torch.from_numpy(
            X_val_np
        ).to(DEVICE, non_blocking=True)
        y_val = torch.from_numpy(
            y_val_np
        ).to(DEVICE, non_blocking=True)

        X_test = torch.from_numpy(
            X_test_np
        ).to(DEVICE, non_blocking=True)
        y_test = torch.from_numpy(
            y_test_np
        ).to(DEVICE, non_blocking=True)

        bank = CNNBank(
            input_length=input_length,
            seeds=MODEL_SEEDS,
        ).to(DEVICE)

        optimizer = make_optimizer(bank)

        # Corresponding seed uses corresponding independent shuffle stream.
        shuffle_rngs = [
            np.random.default_rng(seed)
            for seed in MODEL_SEEDS
        ]

        epoch_history = []

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()

            bank.train()

            indices_cpu = build_per_seed_permutations(
                n_train,
                shuffle_rngs,
            )

            indices = indices_cpu.to(
                DEVICE,
                non_blocking=True,
            )
            del indices_cpu

            epoch_loss_sum = torch.zeros(
                bank.n_models,
                device=DEVICE,
                dtype=torch.float32,
            )
            epoch_correct = torch.zeros(
                bank.n_models,
                device=DEVICE,
                dtype=torch.long,
            )

            for start in range(0, n_train, BATCH_SIZE):
                end = min(
                    start + BATCH_SIZE,
                    n_train,
                )
                bs = end - start

                idx = indices[:, start:end]

                # Advanced indexing creates:
                # X: [64, batch, 1, feature_length]
                # y: [64, batch]
                xb = X_train[idx]
                yb = y_train[idx]

                optimizer.zero_grad(set_to_none=True)

                logits = bank(xb)

                per_element_loss = F.cross_entropy(
                    logits.reshape(-1, N_CLASSES),
                    yb.reshape(-1),
                    reduction="none",
                ).view(bank.n_models, bs)

                per_model_loss = per_element_loss.mean(dim=1)

                # Parameter slices are disjoint across models, so summing the
                # 64 scalar losses does not mix one model's gradients into
                # another model's weights.
                loss = per_model_loss.sum()

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    epoch_loss_sum += (
                        per_element_loss.sum(dim=1)
                    )
                    epoch_correct += (
                        logits.argmax(dim=-1) == yb
                    ).sum(dim=1)

            del indices

            train_loss = (
                epoch_loss_sum / n_train
            ).detach().cpu().numpy()

            train_acc = (
                epoch_correct.float() / n_train
            ).detach().cpu().numpy()

            val_loss, val_acc, _ = evaluate_bank(
                bank,
                X_val,
                y_val,
                collect_predictions=False,
            )

            # Only synchronize this dataset's stream.
            stream.synchronize()

            epoch_seconds = (
                time.perf_counter() - epoch_start
            )
            global_elapsed = (
                time.perf_counter() - experiment_start
            )

            epoch_rows = []

            for model_idx, seed in enumerate(MODEL_SEEDS):
                epoch_rows.append({
                    "seed": seed,
                    "dataset": name,
                    "epoch": epoch,
                    "train_loss": float(
                        train_loss[model_idx]
                    ),
                    "train_accuracy": float(
                        train_acc[model_idx]
                    ),
                    "val_loss": float(
                        val_loss[model_idx]
                    ),
                    "val_accuracy": float(
                        val_acc[model_idx]
                    ),
                    "epoch_seconds_bank": float(
                        epoch_seconds
                    ),
                    "elapsed_seconds_global": float(
                        global_elapsed
                    ),
                })

            append_epoch_rows(epoch_rows)
            epoch_history.extend(epoch_rows)

            update_progress(
                dataset_name=name,
                epoch=epoch,
                epoch_seconds=epoch_seconds,
                train_acc=train_acc,
                val_acc=val_acc,
                n_train=n_train,
            )

        # Final test inference for all 64 seeds.
        test_loss, test_acc, test_predictions = evaluate_bank(
            bank,
            X_test,
            y_test,
            collect_predictions=True,
        )

        stream.synchronize()

        bank_elapsed = (
            time.perf_counter() - dataset_start
        )

        final_epoch_frame = pd.DataFrame(
            epoch_history
        )
        final_epoch_frame = (
            final_epoch_frame[
                final_epoch_frame["epoch"] == EPOCHS
            ]
            .sort_values("seed")
            .reset_index(drop=True)
        )

        metric_rows = []

        for model_idx, seed in enumerate(MODEL_SEEDS):
            y_pred = test_predictions[model_idx]

            metric_rows.append({
                "seed": seed,
                "dataset": name,
                "accuracy": accuracy_score(
                    y_test_np,
                    y_pred,
                ),
                "precision_macro": precision_score(
                    y_test_np,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "precision_weighted": precision_score(
                    y_test_np,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "recall_macro": recall_score(
                    y_test_np,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "recall_weighted": recall_score(
                    y_test_np,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "f1_macro": f1_score(
                    y_test_np,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "f1_weighted": f1_score(
                    y_test_np,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "train_accuracy": float(
                    final_epoch_frame.iloc[
                        model_idx
                    ]["train_accuracy"]
                ),
                "val_accuracy": float(
                    final_epoch_frame.iloc[
                        model_idx
                    ]["val_accuracy"]
                ),
                "test_loss": float(
                    test_loss[model_idx]
                ),
                "elapsed_seconds": float(
                    bank_elapsed
                ),
            })

        # Seed 64 representative checkpoint and report.
        seed64_idx = MODEL_SEEDS.index(64)
        seed64_pred = test_predictions[seed64_idx]

        checkpoint_name = (
            "model_nsl_kdd_1dcnn_seed64.pt"
            if name == "NSL-KDD"
            else "model_unsw_nb15_1dcnn_seed64.pt"
        )

        torch.save(
            {
                "model_type": "Sharma_2024_1D_CNN",
                "seed": 64,
                "input_length": input_length,
                "conv_filters": CONV_FILTERS,
                "kernel_size": KERNEL_SIZE,
                "pool_size": POOL_SIZE,
                "dropout_rate": DROPOUT_RATE,
                "num_classes": N_CLASSES,
                "classes": classes,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "adam_betas": ADAM_BETAS,
                "adam_eps": ADAM_EPS,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "state_dict": bank.seed64_state_dict(),
            },
            OUTPUT_DIR / checkpoint_name,
        )

        cm = confusion_matrix(
            y_test_np,
            seed64_pred,
            labels=np.arange(len(classes)),
        )

        report = classification_report(
            y_test_np,
            seed64_pred,
            target_names=classes,
            digits=4,
            zero_division=0,
        )

        report_name = (
            "classification_report_nsl_1dcnn_seed64.txt"
            if name == "NSL-KDD"
            else "classification_report_unsw_1dcnn_seed64.txt"
        )

        (OUTPUT_DIR / report_name).write_text(
            report,
            encoding="utf-8",
        )

        # Free GPU memory after required data has been copied to CPU.
        del optimizer
        del bank
        del X_train, y_train
        del X_val, y_val
        del X_test, y_test

        torch.cuda.empty_cache()

        return {
            "dataset": name,
            "metrics": metric_rows,
            "confusion_matrix": cm,
            "classes": classes,
        }


# =============================================================================
# 9. SAVE SUMMARY / FIGURES
# =============================================================================

def save_outputs(results: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    metric_rows = []
    result_map = {}

    for result in results:
        metric_rows.extend(result["metrics"])
        result_map[result["dataset"]] = result

    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values(["dataset", "seed"])
        .reset_index(drop=True)
    )

    metrics_df.to_csv(
        PER_SEED_PATH,
        index=False,
    )

    metric_columns = [
        "accuracy",
        "precision_macro",
        "precision_weighted",
        "recall_macro",
        "recall_weighted",
        "f1_macro",
        "f1_weighted",
    ]

    summary_rows = []

    for dataset_name in ["NSL-KDD", "UNSW-NB15"]:
        subset = metrics_df[
            metrics_df["dataset"] == dataset_name
        ]

        for metric in metric_columns:
            summary_rows.append({
                "dataset": dataset_name,
                "metric": metric,
                "mean": subset[metric].mean(),
                "std": subset[metric].std(),
                "min": subset[metric].min(),
                "max": subset[metric].max(),
            })

    summary_df = pd.DataFrame(
        summary_rows
    ).round(6)

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    metrics_agg = (
        metrics_df
        .groupby("dataset")[metric_columns]
        .agg(["mean", "std"])
        .round(6)
    )

    metrics_agg.to_csv(
        OUTPUT_DIR / "metrics.csv"
    )

    # Seed-64 confusion matrices.
    for dataset_name, filename in [
        (
            "NSL-KDD",
            "confusion_matrix_nsl_1dcnn_seed64.png",
        ),
        (
            "UNSW-NB15",
            "confusion_matrix_unsw_1dcnn_seed64.png",
        ),
    ]:
        result = result_map[dataset_name]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            result["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=result["classes"],
            yticklabels=result["classes"],
        )
        plt.title(
            f"{dataset_name} 1D-CNN Confusion Matrix — Seed 64"
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Per-seed plots.
    for dataset_name in ["NSL-KDD", "UNSW-NB15"]:
        subset = (
            metrics_df[
                metrics_df["dataset"] == dataset_name
            ]
            .sort_values("seed")
        )

        slug = (
            dataset_name
            .lower()
            .replace("-", "_")
        )

        for metric in ["accuracy", "f1_macro"]:
            plt.figure(figsize=(10, 5))
            plt.plot(
                subset["seed"],
                subset[metric],
                marker="o",
                markersize=3,
            )
            plt.axhline(
                subset[metric].mean(),
                linestyle="--",
                label="Mean",
            )
            plt.xlabel("Seed")
            plt.ylabel(
                metric.replace("_", " ").title()
            )
            plt.title(
                f"{dataset_name} 1D-CNN — "
                f"{metric.replace('_', ' ').title()} "
                "Across 64 Seeds"
            )
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR
                / f"{slug}_1dcnn_{metric}_across_seeds.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    zip_base = SCRIPT_DIR / "sharma_1dcnn_results_gpu"

    zip_path = Path(
        shutil.make_archive(
            str(zip_base),
            "zip",
            root_dir=OUTPUT_DIR,
        )
    )

    print("\n" + "=" * 100)
    print("FINAL 1D-CNN MULTI-SEED SUMMARY")
    print("=" * 100)

    for dataset_name in ["NSL-KDD", "UNSW-NB15"]:
        subset = metrics_df[
            metrics_df["dataset"] == dataset_name
        ]

        print(
            f"{dataset_name:9s} | "
            f"accuracy "
            f"{subset['accuracy'].mean():.6f} "
            f"± {subset['accuracy'].std():.6f} | "
            f"macro-F1 "
            f"{subset['f1_macro'].mean():.6f} "
            f"± {subset['f1_macro'].std():.6f}"
        )

    print("\nOutputs:", OUTPUT_DIR)
    print("ZIP    :", zip_path)
    print("=" * 100)

    return zip_path


# =============================================================================
# 10. STARTUP VALIDATION
# =============================================================================

def print_header(
    nsl: dict,
    unsw: dict,
) -> None:
    print("\n" + "=" * 100)
    print(
        "SHARMA 2024 1D-CNN REPLICATION — "
        "LOCAL CUDA / 128 CONCURRENT MODELS"
    )
    print("=" * 100)

    print(f"Python              : {sys.version.split()[0]}")
    print(f"PyTorch             : {torch.__version__}")
    print(f"CUDA runtime        : {torch.version.cuda}")
    print(f"GPU                 : {GPU_NAME}")
    print(f"Dataset directory   : {DATA_DIR}")
    print(f"Output directory    : {OUTPUT_DIR}")

    print("-" * 100)

    print(f"Training seeds      : 1..64 ({len(MODEL_SEEDS)})")
    print(f"Fixed split seed    : {DATA_SPLIT_SEED}")
    print(f"Datasets            : 2")
    print(f"Active models       : {TOTAL_MODELS}")
    print(
        "Execution           : "
        "2 CUDA streams × 64 independent vectorized CNNs"
    )
    print(
        "Architecture        : "
        "Conv64(k3,same)→Pool2→"
        "Conv32(k3,same)→Pool2→"
        "Conv32(k3,same)→Pool2→Flatten→Dense5"
    )
    print(f"Activation          : ReLU")
    print(f"Dropout             : {DROPOUT_RATE}")
    print(f"Optimizer           : AdamW")
    print(f"Learning rate       : {LEARNING_RATE}")
    print(f"Weight decay        : {WEIGHT_DECAY}")
    print(f"Adam betas / eps    : {ADAM_BETAS} / {ADAM_EPS}")
    print(f"Epochs/model        : {EPOCHS}")
    print(f"Batch size/model    : {BATCH_SIZE}")
    print(
        f"Split               : "
        f"{TRAIN_SIZE:.0%}/{VAL_SIZE:.0%}/{TEST_SIZE:.0%}"
    )
    print(f"Precision           : FP32, TF32 disabled")
    print(f"Total model-epochs  : {TOTAL_MODEL_EPOCHS}")

    print("-" * 100)

    print(
        f"NSL-KDD             : "
        f"features={nsl['feature_count']} | "
        f"train={nsl['X_train'].shape} | "
        f"val={nsl['X_val'].shape} | "
        f"test={nsl['X_test'].shape}"
    )
    print(f"NSL classes         : {nsl['classes']}")

    print(
        f"UNSW-NB15           : "
        f"features={unsw['feature_count']} | "
        f"train={unsw['X_train'].shape} | "
        f"val={unsw['X_val'].shape} | "
        f"test={unsw['X_test'].shape}"
    )
    print(f"UNSW classes        : {unsw['classes']}")

    print("=" * 100)
    print()


def validate_expected_shapes(
    nsl: dict,
    unsw: dict,
) -> None:
    """
    These are the shapes printed by the supplied notebook for the same
    full datasets. Fail loudly if local preprocessing silently diverges.
    """
    expected = {
        "NSL-KDD": {
            "feature_count": 35,
            "train": (89109, 1, 35),
            "val": (22278, 1, 35),
            "test": (37130, 1, 35),
        },
        "UNSW-NB15": {
            "feature_count": 38,
            "train": (111074, 1, 38),
            "val": (27769, 1, 38),
            "test": (46281, 1, 38),
        },
    }

    for dataset in [nsl, unsw]:
        e = expected[dataset["name"]]

        actual = {
            "feature_count": dataset["feature_count"],
            "train": tuple(dataset["X_train"].shape),
            "val": tuple(dataset["X_val"].shape),
            "test": tuple(dataset["X_test"].shape),
        }

        if actual != e:
            raise RuntimeError(
                f"{dataset['name']} preprocessing does not match "
                f"the supplied notebook.\n"
                f"Expected: {e}\n"
                f"Actual  : {actual}\n"
                "Stopping instead of silently running a different experiment."
            )


# =============================================================================
# 11. MAIN
# =============================================================================

def main() -> None:
    global experiment_start
    global overall_bar
    global dataset_bars

    # Fresh result logs.
    for path in [
        EPOCH_LOG_PATH,
        PER_SEED_PATH,
        SUMMARY_PATH,
        ERROR_PATH,
    ]:
        if path.exists():
            path.unlink()

    nsl = preprocess_nsl_kdd()
    unsw = preprocess_unsw_nb15()

    validate_expected_shapes(
        nsl,
        unsw,
    )

    print_header(
        nsl,
        unsw,
    )

    experiment_start = time.perf_counter()

    tqdm.set_lock(PROGRESS_LOCK)

    overall_bar = tqdm(
        total=TOTAL_MODEL_EPOCHS,
        desc="ALL 128 CNN MODELS",
        unit="model-epoch",
        position=0,
        dynamic_ncols=True,
        leave=True,
    )

    dataset_bars["NSL-KDD"] = tqdm(
        total=EPOCHS,
        desc="NSL-KDD 64 CNNs",
        unit="epoch",
        position=1,
        dynamic_ncols=True,
        leave=True,
    )

    dataset_bars["UNSW-NB15"] = tqdm(
        total=EPOCHS,
        desc="UNSW-NB15 64 CNNs",
        unit="epoch",
        position=2,
        dynamic_ncols=True,
        leave=True,
    )

    results = []
    errors = []

    # The two 64-model banks submit work concurrently to two CUDA streams.
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="cnn-bank",
    ) as executor:

        futures = {
            executor.submit(
                train_bank,
                nsl,
            ): "NSL-KDD",
            executor.submit(
                train_bank,
                unsw,
            ): "UNSW-NB15",
        }

        for future in as_completed(futures):
            dataset_name = futures[future]

            try:
                result = future.result()
                results.append(result)

                with PROGRESS_LOCK:
                    tqdm.write(
                        f"[{dataset_name}] "
                        f"all 64 CNNs finished "
                        f"{EPOCHS} epochs + test evaluation."
                    )

            except Exception as exc:
                tb = traceback.format_exc()

                errors.append({
                    "dataset": dataset_name,
                    "error": repr(exc),
                    "traceback": tb,
                })

                with PROGRESS_LOCK:
                    tqdm.write(
                        f"\nFAILED DATASET BANK: "
                        f"{dataset_name}\n{tb}"
                    )

    for bar in dataset_bars.values():
        bar.close()

    overall_bar.close()

    elapsed = (
        time.perf_counter() - experiment_start
    )

    if errors:
        pd.DataFrame(errors).to_csv(
            ERROR_PATH,
            index=False,
        )

    if len(results) != 2:
        raise RuntimeError(
            f"Only {len(results)}/2 dataset banks completed. "
            f"See: {ERROR_PATH}"
        )

    zip_path = save_outputs(results)

    print(
        f"\nTotal wall time: {fmt_time(elapsed)}"
    )
    print(
        f"Final GPU state: {query_gpu_stats()}"
    )
    print(f"Results ZIP: {zip_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
