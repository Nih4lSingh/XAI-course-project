#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sharma et al. (2024) DNN replication — local Windows CUDA/PyTorch runner.

Purpose:
    Make the original Colab/TensorFlow experiment runnable on a native Windows
    NVIDIA GPU while preserving the experimental design:
      - 64 seeds (1..64)
      - 2 datasets: NSL-KDD and UNSW-NB15
      - fixed 60/15/25 split with seed 42
      - Dense(64) -> Dense(64) -> Dense(64) -> Dense(5, Softmax)
      - ReLU hidden activations
      - Dropout = 0.01
      - Adam-style optimizer, LR = 0.001, weight decay = 0.0001
      - 20 epochs
      - batch size 128
      - same preprocessing, class filtering/mapping, and metrics

GPU execution:
    All 128 independent model jobs are resident/active concurrently.
    They are represented as two vectorized banks of 64 independent models
    (one bank per dataset). The two banks execute on separate CUDA streams.
    This avoids launching 128 Python/TensorFlow workers for tiny MLPs while
    preserving independent weights, optimizer state, shuffled samples and
    per-seed metrics.

Notes:
    - This is a backend port, so bit-for-bit TensorFlow RNG equivalence is not
      expected. Experimental hyperparameters and data preparation are preserved.
    - No AMP / mixed precision / TF32 is used; training is FP32.
"""

from __future__ import annotations

import csv
import math
import os
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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# 0. EXACT EXPERIMENT SETTINGS FROM THE NOTEBOOK
# =============================================================================

SEEDS = list(range(1, 65))
DATA_SPLIT_SEED = 42

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-7          # Keras Adam default epsilon

EPOCHS = 20
BATCH_SIZE = 128
DROPOUT_RATE = 0.01
NUM_CLASSES = 5

# Two vectorized banks x 64 models = 128 active models.
MODELS_PER_DATASET = len(SEEDS)
TOTAL_MODELS = len(SEEDS) * 2
TOTAL_MODEL_EPOCHS = TOTAL_MODELS * EPOCHS

# Preserve FP32-style execution rather than silently enabling AMP/TF32.
USE_AMP = False


# =============================================================================
# 1. LOCAL PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Flexible dataset path resolution
def resolve_data_paths() -> tuple[Path, Path, Path, Path]:
    nsl_train, nsl_test = None, None
    unsw_train, unsw_test = None, None

    nsl_dirs = [
        SCRIPT_DIR / "data" / "raw" / "nsl_kdd",
        SCRIPT_DIR.parent.parent / "data" / "raw" / "nsl_kdd",
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
        SCRIPT_DIR.parent.parent / "data" / "raw" / "unsw_nb15",
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
            f"  - {SCRIPT_DIR.parent.parent / 'data' / 'raw'}\n"
            f"  - {SCRIPT_DIR / 'datasets'}\n"
            f"  - {SCRIPT_DIR}"
        )
    return nsl_train, nsl_test, unsw_train, unsw_test


NSL_TRAIN_PATH, NSL_TEST_PATH, UNSW_TRAIN_PATH, UNSW_TEST_PATH = resolve_data_paths()
DATA_DIR = NSL_TRAIN_PATH.parent

if (SCRIPT_DIR / "results").is_dir():
    OUTPUT_DIR = SCRIPT_DIR / "results" / "dnn_gpu"
elif (SCRIPT_DIR.parent.parent / "results").is_dir():
    OUTPUT_DIR = SCRIPT_DIR.parent.parent / "results" / "dnn_gpu"
else:
    OUTPUT_DIR = SCRIPT_DIR / "outputs_gpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PER_SEED_PATH = OUTPUT_DIR / "metrics_per_seed.csv"
EPOCH_LOG_PATH = OUTPUT_DIR / "epoch_log.csv"
SUMMARY_PATH = OUTPUT_DIR / "metrics_summary.csv"
ERROR_PATH = OUTPUT_DIR / "training_errors.csv"


# =============================================================================
# 2. CUDA / REPRODUCIBILITY SETUP
# =============================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
    GPU_NAME = torch.cuda.get_device_name(0)
    USE_CUDA = True

    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(DATA_SPLIT_SEED)
else:
    DEVICE = torch.device("cpu")
    GPU_NAME = "CPU (No CUDA available, running in CPU mode)"
    USE_CUDA = False

# Initial CPU-side deterministic state for preprocessing / helper operations.
np.random.seed(DATA_SPLIT_SEED)
torch.manual_seed(DATA_SPLIT_SEED)


# =============================================================================
# 3. SMALL UTILITIES
# =============================================================================

PRINT_LOCK = threading.RLock()
CSV_LOCK = threading.RLock()
PROGRESS_LOCK = threading.RLock()

experiment_start = None
overall_bar = None
dataset_bars = {}


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def query_nvidia_smi() -> str:
    """Compact GPU utilization line. Failure is harmless."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-i", "0",
        ]
        out = subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip().splitlines()[0]
        util, mem_used, mem_total, power, temp = [x.strip() for x in out.split(",")]
        return (
            f"GPU {util}% | VRAM {mem_used}/{mem_total} MiB | "
            f"{power} W | {temp} C"
        )
    except Exception:
        try:
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            return f"VRAM alloc/res {alloc:.0f}/{reserved:.0f} MiB"
        except Exception:
            return "GPU stats N/A"


def clean_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        df[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def append_epoch_rows(rows: list[dict]) -> None:
    if not rows:
        return
    with CSV_LOCK:
        df = pd.DataFrame(rows)
        header = not EPOCH_LOG_PATH.exists()
        df.to_csv(EPOCH_LOG_PATH, mode="a", header=header, index=False)


def progress_update(
    dataset_name: str,
    epoch: int,
    epoch_seconds: float,
    train_acc: np.ndarray,
    val_acc: np.ndarray,
    n_train: int,
) -> None:
    """Update three live progress bars and print a compact epoch report."""
    global overall_bar, dataset_bars

    elapsed = time.perf_counter() - experiment_start
    model_epochs_completed = 64  # one epoch completed by every model in bank
    model_epochs_per_s = model_epochs_completed / max(epoch_seconds, 1e-9)
    train_samples_per_s = (n_train * 64) / max(epoch_seconds, 1e-9)

    gpu_line = query_nvidia_smi()

    with PROGRESS_LOCK:
        dataset_bars[dataset_name].update(1)
        dataset_bars[dataset_name].set_postfix_str(
            f"train={train_acc.mean():.4f} val={val_acc.mean():.4f} "
            f"{epoch_seconds:.2f}s/epoch"
        )

        overall_bar.update(64)
        overall_rate = overall_bar.n / max(elapsed, 1e-9)
        overall_bar.set_postfix_str(
            f"{overall_rate:.2f} model-epochs/s | {gpu_line}"
        )

        tqdm.write(
            f"[{dataset_name:9s}] epoch {epoch:02d}/{EPOCHS} | "
            f"+64 model-epochs | "
            f"train acc mean/min/max "
            f"{train_acc.mean():.5f}/{train_acc.min():.5f}/{train_acc.max():.5f} | "
            f"val acc mean/min/max "
            f"{val_acc.mean():.5f}/{val_acc.min():.5f}/{val_acc.max():.5f} | "
            f"{epoch_seconds:.2f}s | "
            f"{model_epochs_per_s:.2f} model-epochs/s | "
            f"{train_samples_per_s/1e6:.3f}M train-samples/s | "
            f"elapsed {fmt_time(elapsed)} | {gpu_line}"
        )


# =============================================================================
# 4. PREPROCESS NSL-KDD — SAME LOGIC AS NOTEBOOK
# =============================================================================

NSL_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack",
    "difficulty",
]

NSL_ATTACK_MAP = {
    # Normal
    "normal": "Normal",

    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",

    # Probe
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",

    # R2L
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",

    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "httptunnel": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}


def preprocess_nsl():
    print("\nLoading NSL-KDD...")
    nsl_train = pd.read_csv(NSL_TRAIN_PATH, header=None, names=NSL_COLUMNS)
    nsl_test = pd.read_csv(NSL_TEST_PATH, header=None, names=NSL_COLUMNS)

    nsl = pd.concat([nsl_train, nsl_test], ignore_index=True)

    nsl["attack"] = (
        nsl["attack"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.rstrip(".")
    )

    nsl["target"] = nsl["attack"].map(NSL_ATTACK_MAP)

    unmapped = int(nsl["target"].isna().sum())
    if unmapped:
        print(f"NSL-KDD: dropping {unmapped:,} rows with unmapped attack labels.")

    nsl = nsl.dropna(subset=["target"]).reset_index(drop=True)

    NSL_DROP_COLUMNS = []

    X_nsl = nsl.drop(columns=["target", "attack", "difficulty"])
    y_nsl_text = nsl["target"].copy()
    X_nsl = X_nsl.drop(columns=NSL_DROP_COLUMNS, errors="ignore")

    X_train, X_temp, y_train_text, y_temp_text = train_test_split(
        X_nsl,
        y_nsl_text,
        test_size=0.40,
        random_state=DATA_SPLIT_SEED,
        stratify=y_nsl_text,
    )

    X_val, X_test, y_val_text, y_test_text = train_test_split(
        X_temp,
        y_temp_text,
        test_size=0.625,
        random_state=DATA_SPLIT_SEED,
        stratify=y_temp_text,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_val = label_encoder.transform(y_val_text)
    y_test = label_encoder.transform(y_test_text)

    categorical = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical = [c for c in X_train.columns if c not in categorical]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    X_train_cat = encoder.fit_transform(X_train[categorical])
    X_val_cat = encoder.transform(X_val[categorical])
    X_test_cat = encoder.transform(X_test[categorical])

    X_train_num = clean_numeric(X_train, numerical)
    X_val_num = clean_numeric(X_val, numerical)
    X_test_num = clean_numeric(X_test, numerical)

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

    return {
        "name": "NSL-KDD",
        "X_train": np.asarray(X_train_processed, dtype=np.float32),
        "y_train": np.asarray(y_train, dtype=np.int64),
        "X_val": np.asarray(X_val_processed, dtype=np.float32),
        "y_val": np.asarray(y_val, dtype=np.int64),
        "X_test": np.asarray(X_test_processed, dtype=np.float32),
        "y_test": np.asarray(y_test, dtype=np.int64),
        "classes": label_encoder.classes_.tolist(),
        "label_encoder": label_encoder,
    }


# =============================================================================
# 5. PREPROCESS UNSW-NB15 — SAME LOGIC AS NOTEBOOK
# =============================================================================

UNSW_CLASSES = ["Normal", "Generic", "Exploits", "DoS", "Fuzzers"]


def preprocess_unsw():
    print("\nLoading UNSW-NB15...")
    unsw_train = pd.read_csv(UNSW_TRAIN_PATH)
    unsw_test = pd.read_csv(UNSW_TEST_PATH)

    unsw = pd.concat([unsw_train, unsw_test], ignore_index=True)

    unsw["attack_cat"] = unsw["attack_cat"].astype(str).str.strip()
    unsw = unsw[unsw["attack_cat"].isin(UNSW_CLASSES)].copy()

    parts = []
    for class_name in UNSW_CLASSES:
        class_df = unsw[unsw["attack_cat"] == class_name].copy()
        if len(class_df) > 50000:
            class_df = class_df.sample(
                n=50000,
                random_state=DATA_SPLIT_SEED,
            )
        parts.append(class_df)

    unsw = pd.concat(parts, ignore_index=True)
    unsw = unsw.sample(
        frac=1,
        random_state=DATA_SPLIT_SEED,
    ).reset_index(drop=True)

    UNSW_DROP_COLUMNS = [
        "ct_src_dport_ltm",
        "loss",
        "dwin",
        "ct_ftp_cmd",
        "label",
        "ct_srv_dst",
    ]

    X_unsw = unsw.drop(columns=["attack_cat"], errors="ignore")
    y_unsw_text = unsw["attack_cat"].copy()
    X_unsw = X_unsw.drop(columns=UNSW_DROP_COLUMNS, errors="ignore")
    X_unsw = X_unsw.drop(columns=["id"], errors="ignore")

    X_train, X_temp, y_train_text, y_temp_text = train_test_split(
        X_unsw,
        y_unsw_text,
        test_size=0.40,
        random_state=DATA_SPLIT_SEED,
        stratify=y_unsw_text,
    )

    X_val, X_test, y_val_text, y_test_text = train_test_split(
        X_temp,
        y_temp_text,
        test_size=0.625,
        random_state=DATA_SPLIT_SEED,
        stratify=y_temp_text,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_val = label_encoder.transform(y_val_text)
    y_test = label_encoder.transform(y_test_text)

    categorical = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical = [c for c in X_train.columns if c not in categorical]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    X_train_cat = encoder.fit_transform(X_train[categorical])
    X_val_cat = encoder.transform(X_val[categorical])
    X_test_cat = encoder.transform(X_test[categorical])

    X_train_num = clean_numeric(X_train, numerical)
    X_val_num = clean_numeric(X_val, numerical)
    X_test_num = clean_numeric(X_test, numerical)

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

    return {
        "name": "UNSW-NB15",
        "X_train": np.asarray(X_train_processed, dtype=np.float32),
        "y_train": np.asarray(y_train, dtype=np.int64),
        "X_val": np.asarray(X_val_processed, dtype=np.float32),
        "y_val": np.asarray(y_val, dtype=np.int64),
        "X_test": np.asarray(X_test_processed, dtype=np.float32),
        "y_test": np.asarray(y_test, dtype=np.int64),
        "classes": label_encoder.classes_.tolist(),
        "label_encoder": label_encoder,
    }


# =============================================================================
# 6. 64-INDEPENDENT-MODEL GPU BANK
# =============================================================================

class DNNBank(nn.Module):
    """
    64 independent copies of:
        Dense(64) -> ReLU -> Dropout(0.01)
        Dense(64) -> ReLU -> Dropout(0.01)
        Dense(64) -> ReLU -> Dropout(0.01)
        Dense(5)

    Parameters carry a leading model dimension. No weights are shared between
    seeds. A single optimizer tensor holds element-wise independent Adam state
    for every model slice.
    """

    def __init__(
        self,
        input_dim: int,
        seeds: list[int],
        dropout_rate: float,
        init_dataset_offset: int,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.n_models = len(seeds)
        self.dropout_rate = float(dropout_rate)
        self.seeds = list(seeds)

        # Store kernels Keras-style: [model, input, output]
        self.w1 = nn.Parameter(torch.empty(self.n_models, input_dim, 64))
        self.b1 = nn.Parameter(torch.zeros(self.n_models, 1, 64))

        self.w2 = nn.Parameter(torch.empty(self.n_models, 64, 64))
        self.b2 = nn.Parameter(torch.zeros(self.n_models, 1, 64))

        self.w3 = nn.Parameter(torch.empty(self.n_models, 64, 64))
        self.b3 = nn.Parameter(torch.zeros(self.n_models, 1, 64))

        self.w4 = nn.Parameter(torch.empty(self.n_models, 64, NUM_CLASSES))
        self.b4 = nn.Parameter(torch.zeros(self.n_models, 1, NUM_CLASSES))

        self._keras_style_glorot_init(init_dataset_offset)

    @staticmethod
    def _glorot_tensor(
        n_models: int,
        fan_in: int,
        fan_out: int,
        seeds: list[int],
        seed_offset: int,
        dataset_offset: int,
    ) -> torch.Tensor:
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        out = torch.empty(n_models, fan_in, fan_out, dtype=torch.float32)

        # Each seed/model gets its own deterministic initializer stream.
        for i, seed in enumerate(seeds):
            g = torch.Generator(device="cpu")
            g.manual_seed(int(dataset_offset + seed + seed_offset))
            out[i].uniform_(-bound, bound, generator=g)

        return out

    def _keras_style_glorot_init(self, dataset_offset: int) -> None:
        with torch.no_grad():
            self.w1.copy_(
                self._glorot_tensor(
                    self.n_models, self.input_dim, 64,
                    self.seeds, 1, dataset_offset
                )
            )
            self.w2.copy_(
                self._glorot_tensor(
                    self.n_models, 64, 64,
                    self.seeds, 2, dataset_offset
                )
            )
            self.w3.copy_(
                self._glorot_tensor(
                    self.n_models, 64, 64,
                    self.seeds, 3, dataset_offset
                )
            )
            self.w4.copy_(
                self._glorot_tensor(
                    self.n_models, 64, NUM_CLASSES,
                    self.seeds, 4, dataset_offset
                )
            )
            self.b1.zero_()
            self.b2.zero_()
            self.b3.zero_()
            self.b4.zero_()

    def forward(
        self,
        x: torch.Tensor,
        dropout_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        # x: [models, batch, features]
        h = torch.bmm(x, self.w1) + self.b1
        h = F.relu(h)
        h = self._dropout(h, dropout_generator)

        h = torch.bmm(h, self.w2) + self.b2
        h = F.relu(h)
        h = self._dropout(h, dropout_generator)

        h = torch.bmm(h, self.w3) + self.b3
        h = F.relu(h)
        h = self._dropout(h, dropout_generator)

        logits = torch.bmm(h, self.w4) + self.b4
        return logits

    def _dropout(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if not self.training or self.dropout_rate <= 0.0:
            return x

        keep = 1.0 - self.dropout_rate
        mask = torch.rand(
            x.shape,
            device=x.device,
            dtype=torch.float32,
            generator=generator,
        ) < keep

        return x * mask.to(x.dtype) / keep

    def seed64_state_dict(self) -> dict:
        """
        Export model seed 64 in ordinary nn.Linear orientation so it can be
        loaded into a normal single-model PyTorch DNN later.
        """
        idx = self.seeds.index(64)
        return {
            "fc1.weight": self.w1[idx].detach().cpu().T.contiguous(),
            "fc1.bias": self.b1[idx, 0].detach().cpu().contiguous(),
            "fc2.weight": self.w2[idx].detach().cpu().T.contiguous(),
            "fc2.bias": self.b2[idx, 0].detach().cpu().contiguous(),
            "fc3.weight": self.w3[idx].detach().cpu().T.contiguous(),
            "fc3.bias": self.b3[idx, 0].detach().cpu().contiguous(),
            "fc4.weight": self.w4[idx].detach().cpu().T.contiguous(),
            "fc4.bias": self.b4[idx, 0].detach().cpu().contiguous(),
        }


class SingleDNN(nn.Module):
    """Ordinary seed-64 model loader for later inference/SHAP."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        return self.fc4(x)


# =============================================================================
# 7. OPTIMIZER
# =============================================================================

def make_optimizer(model: nn.Module):
    """
    Keras Adam(weight_decay=...) uses decoupled weight decay.
    AdamW is the corresponding PyTorch form.

    Explicit Keras defaults preserved:
        beta1=0.9, beta2=0.999, epsilon=1e-7
    """
    kwargs = dict(
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )

    # Fused optimizer is a backend execution detail. Use it when supported on
    # CUDA; fall back cleanly without changing hyperparameters.
    try:
        return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(model.parameters(), **kwargs)


# =============================================================================
# 8. VECTORIZED TRAIN / VALIDATION / TEST
# =============================================================================

def build_per_seed_epoch_permutations(
    n_samples: int,
    rngs: list[np.random.Generator],
) -> torch.Tensor:
    """
    Independent shuffled order for every seed/model.
    Returned shape: [64, n_samples].
    """
    arr = np.empty((len(rngs), n_samples), dtype=np.int64)
    for i, rng in enumerate(rngs):
        arr[i] = rng.permutation(n_samples)
    return torch.from_numpy(arr)


@torch.no_grad()
def evaluate_bank(
    bank: DNNBank,
    X: torch.Tensor,
    y: torch.Tensor,
    stream: torch.cuda.Stream,
    collect_predictions: bool = False,
):
    bank.eval()

    n_models = bank.n_models
    n_samples = X.shape[0]

    total_loss = torch.zeros(n_models, device=DEVICE, dtype=torch.float32)
    total_correct = torch.zeros(n_models, device=DEVICE, dtype=torch.long)

    all_preds = [] if collect_predictions else None

    for start in range(0, n_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_samples)
        xb = X[start:end]
        yb = y[start:end]
        bs = end - start

        xb_bank = xb.unsqueeze(0).expand(n_models, -1, -1)
        logits = bank(xb_bank, dropout_generator=None)

        targets = yb.unsqueeze(0).expand(n_models, -1)

        flat_loss = F.cross_entropy(
            logits.reshape(-1, NUM_CLASSES),
            targets.reshape(-1),
            reduction="none",
        ).view(n_models, bs)

        total_loss += flat_loss.sum(dim=1)
        pred = logits.argmax(dim=-1)
        total_correct += (pred == targets).sum(dim=1)

        if collect_predictions:
            all_preds.append(pred.detach().cpu())

    avg_loss = (total_loss / n_samples).detach().cpu().numpy()
    accuracy = (total_correct.float() / n_samples).detach().cpu().numpy()

    if collect_predictions:
        predictions = torch.cat(all_preds, dim=1).numpy()
    else:
        predictions = None

    return avg_loss, accuracy, predictions


def train_bank(dataset: dict) -> dict:
    """
    Train all 64 seeds for one dataset concurrently.
    When CUDA is available, executes on a dedicated CUDA stream.
    """
    if USE_CUDA:
        torch.cuda.set_device(DEVICE)
        stream = torch.cuda.Stream(device=DEVICE)
        import contextlib
        stream_ctx = torch.cuda.stream(stream)
    else:
        stream = None
        import contextlib
        stream_ctx = contextlib.nullcontext()

    bank_start = time.perf_counter()

    name = dataset["name"]
    X_train_np = dataset["X_train"]
    y_train_np = dataset["y_train"]
    X_val_np = dataset["X_val"]
    y_val_np = dataset["y_val"]
    X_test_np = dataset["X_test"]
    y_test_np = dataset["y_test"]
    classes = dataset["classes"]

    n_train = len(X_train_np)
    input_dim = X_train_np.shape[1]

    # Preserve the notebook's seed semantics across both datasets:
    # the corresponding NSL-KDD and UNSW-NB15 model use the same seed.
    init_dataset_offset = 0

    with stream_ctx:
        X_train = torch.from_numpy(X_train_np).to(DEVICE, non_blocking=USE_CUDA)
        y_train = torch.from_numpy(y_train_np).to(DEVICE, non_blocking=USE_CUDA)
        X_val = torch.from_numpy(X_val_np).to(DEVICE, non_blocking=USE_CUDA)
        y_val = torch.from_numpy(y_val_np).to(DEVICE, non_blocking=USE_CUDA)
        X_test = torch.from_numpy(X_test_np).to(DEVICE, non_blocking=USE_CUDA)
        y_test = torch.from_numpy(y_test_np).to(DEVICE, non_blocking=USE_CUDA)

        bank = DNNBank(
            input_dim=input_dim,
            seeds=SEEDS,
            dropout_rate=DROPOUT_RATE,
            init_dataset_offset=init_dataset_offset,
        ).to(DEVICE)

        optimizer = make_optimizer(bank)

        # Independent shuffle RNG per model/seed.
        shuffle_rngs = [
            np.random.default_rng(seed)
            for seed in SEEDS
        ]

        # Independent bank-level dropout stream. Model slices get
        # independent random values within each generated mask.
        dropout_gen = torch.Generator(device=DEVICE.type)
        dropout_gen.manual_seed(
            DATA_SPLIT_SEED + 1_000_000
        )

        epoch_history = []

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()

            bank.train()

            epoch_indices_cpu = build_per_seed_epoch_permutations(
                n_train,
                shuffle_rngs,
            )
            epoch_indices = epoch_indices_cpu.to(
                DEVICE,
                non_blocking=USE_CUDA,
            )
            del epoch_indices_cpu

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
                end = min(start + BATCH_SIZE, n_train)
                bs = end - start

                idx = epoch_indices[:, start:end]        # [64, B]
                xb = X_train[idx]                        # [64, B, D]
                yb = y_train[idx]                        # [64, B]

                optimizer.zero_grad(set_to_none=True)

                logits = bank(
                    xb,
                    dropout_generator=dropout_gen,
                )

                per_element_loss = F.cross_entropy(
                    logits.reshape(-1, NUM_CLASSES),
                    yb.reshape(-1),
                    reduction="none",
                ).view(bank.n_models, bs)

                # Each model's parameter slice receives exactly that model's
                # mean batch loss gradient. Summing across models does NOT
                # average their gradients together because parameters are
                # disjoint along the model dimension.
                per_model_loss = per_element_loss.mean(dim=1)
                loss = per_model_loss.sum()

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    epoch_loss_sum += per_element_loss.sum(dim=1)
                    epoch_correct += (
                        logits.argmax(dim=-1) == yb
                    ).sum(dim=1)

            del epoch_indices

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
                stream=stream,
                collect_predictions=False,
            )

            # Synchronize this bank's stream only. The other dataset's bank
            # can continue executing on its own stream.
            if stream is not None:
                stream.synchronize()
            epoch_seconds = time.perf_counter() - epoch_start

            elapsed = time.perf_counter() - experiment_start

            rows = []
            for i, seed in enumerate(SEEDS):
                rows.append({
                    "seed": seed,
                    "dataset": name,
                    "epoch": epoch,
                    "train_loss": float(train_loss[i]),
                    "train_accuracy": float(train_acc[i]),
                    "val_loss": float(val_loss[i]),
                    "val_accuracy": float(val_acc[i]),
                    "epoch_seconds_bank": float(epoch_seconds),
                    "elapsed_seconds_global": float(elapsed),
                })
            append_epoch_rows(rows)
            epoch_history.extend(rows)

            progress_update(
                dataset_name=name,
                epoch=epoch,
                epoch_seconds=epoch_seconds,
                train_acc=train_acc,
                val_acc=val_acc,
                n_train=n_train,
            )

        # Final test evaluation for all 64 models.
        test_loss, test_acc, test_predictions = evaluate_bank(
            bank,
            X_test,
            y_test,
            stream=stream,
            collect_predictions=True,
        )
        if stream is not None:
            stream.synchronize()

        final_epoch_rows = pd.DataFrame(epoch_history)
        final_epoch_rows = final_epoch_rows[
            final_epoch_rows["epoch"] == EPOCHS
        ].sort_values("seed")

        y_true_cpu = y_test_np

        bank_elapsed_seconds = time.perf_counter() - bank_start

        metric_rows = []
        for i, seed in enumerate(SEEDS):
            y_pred = test_predictions[i]

            metric_rows.append({
                "seed": seed,
                "dataset": name,
                "accuracy": accuracy_score(y_true_cpu, y_pred),
                "precision_macro": precision_score(
                    y_true_cpu, y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "precision_weighted": precision_score(
                    y_true_cpu, y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "recall_macro": recall_score(
                    y_true_cpu, y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "recall_weighted": recall_score(
                    y_true_cpu, y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "f1_macro": f1_score(
                    y_true_cpu, y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "f1_weighted": f1_score(
                    y_true_cpu, y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "train_accuracy": float(
                    final_epoch_rows.iloc[i]["train_accuracy"]
                ),
                "val_accuracy": float(
                    final_epoch_rows.iloc[i]["val_accuracy"]
                ),
                "elapsed_seconds": float(bank_elapsed_seconds),
                "test_loss": float(test_loss[i]),
            })

        seed64_idx = SEEDS.index(64)
        seed64_pred = test_predictions[seed64_idx]
        seed64_state = bank.seed64_state_dict()

        model_filename = (
            "model_nsl_kdd_seed64.pt"
            if name == "NSL-KDD"
            else "model_unsw_nb15_seed64.pt"
        )

        torch.save(
            {
                "architecture": "64-64-64-5",
                "input_dim": int(input_dim),
                "num_classes": NUM_CLASSES,
                "classes": classes,
                "dropout_rate": DROPOUT_RATE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "adam_betas": ADAM_BETAS,
                "adam_eps": ADAM_EPS,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": 64,
                "state_dict": seed64_state,
            },
            OUTPUT_DIR / model_filename,
        )

        cm = confusion_matrix(y_true_cpu, seed64_pred)

        report = classification_report(
            y_true_cpu,
            seed64_pred,
            target_names=classes,
            zero_division=0,
        )

        report_name = (
            "classification_report_nsl_seed64.txt"
            if name == "NSL-KDD"
            else "classification_report_unsw_seed64.txt"
        )
        (OUTPUT_DIR / report_name).write_text(
            report,
            encoding="utf-8",
        )

        # Free this bank after all required outputs are copied to CPU.
        del optimizer
        del bank
        del X_train, y_train, X_val, y_val, X_test, y_test
        if USE_CUDA:
            torch.cuda.empty_cache()

        return {
            "dataset": name,
            "metrics": metric_rows,
            "cm": cm,
            "classes": classes,
        }


# =============================================================================
# 9. OUTPUT PLOTS / SUMMARIES
# =============================================================================

def save_outputs(all_results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics_rows = []
    result_by_name = {}

    for result in all_results:
        metrics_rows.extend(result["metrics"])
        result_by_name[result["dataset"]] = result

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["dataset", "seed"]
    ).reset_index(drop=True)

    metrics_df.to_csv(PER_SEED_PATH, index=False)

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

    summary_df = pd.DataFrame(summary_rows).round(6)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    # Notebook-compatible mean/std table.
    metrics_agg = (
        metrics_df
        .groupby("dataset")[metric_columns]
        .agg(["mean", "std"])
        .round(6)
    )
    metrics_agg.to_csv(OUTPUT_DIR / "metrics.csv")

    # Confusion matrices (seed 64).
    for dataset_name, filename in [
        ("NSL-KDD", "confusion_matrix_nsl_seed64.png"),
        ("UNSW-NB15", "confusion_matrix_unsw_seed64.png"),
    ]:
        result = result_by_name[dataset_name]

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            result["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=result["classes"],
            yticklabels=result["classes"],
        )
        plt.title(f"{dataset_name} DNN Confusion Matrix — Seed 64")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Accuracy / macro-F1 vs seed plots.
    for dataset_name in ["NSL-KDD", "UNSW-NB15"]:
        subset = metrics_df[
            metrics_df["dataset"] == dataset_name
        ].sort_values("seed")

        slug = dataset_name.lower().replace("-", "_")

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
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(
                f"{dataset_name} - "
                f"{metric.replace('_', ' ').title()} Across 64 Seeds"
            )
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR / f"{slug}_{metric}_across_seeds.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # Package the complete result directory.
    zip_base = OUTPUT_DIR.parent / "sharma_dnn_results_gpu"
    zip_path = Path(
        shutil.make_archive(
            str(zip_base),
            "zip",
            root_dir=OUTPUT_DIR,
        )
    )

    print("\n" + "=" * 100)
    print("FINAL MULTI-SEED SUMMARY")
    print("=" * 100)

    for dataset_name in ["NSL-KDD", "UNSW-NB15"]:
        subset = metrics_df[
            metrics_df["dataset"] == dataset_name
        ]
        print(
            f"{dataset_name:9s} | "
            f"accuracy {subset['accuracy'].mean():.6f} "
            f"± {subset['accuracy'].std():.6f} | "
            f"macro-F1 {subset['f1_macro'].mean():.6f} "
            f"± {subset['f1_macro'].std():.6f}"
        )

    print("\nOutputs:", OUTPUT_DIR)
    print("ZIP    :", zip_path)
    print("=" * 100)


# =============================================================================
# 10. MAIN
# =============================================================================

def print_run_header(nsl: dict, unsw: dict) -> None:
    print("\n" + "=" * 100)
    print("SHARMA 2024 DNN REPLICATION — LOCAL CUDA / 128 CONCURRENT MODELS")
    print("=" * 100)
    print(f"Python              : {sys.version.split()[0]}")
    print(f"PyTorch             : {torch.__version__}")
    print(f"CUDA runtime        : {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"Device / GPU        : {GPU_NAME}")
    print(f"NSL-KDD Path        : {NSL_TRAIN_PATH}")
    print(f"UNSW-NB15 Path      : {UNSW_TRAIN_PATH}")
    print(f"Output directory    : {OUTPUT_DIR}")
    print("-" * 100)
    print(f"Seeds               : 1..64 ({len(SEEDS)})")
    print(f"Datasets            : 2")
    print(f"Active models       : {TOTAL_MODELS}")
    print(f"Execution           : 2 CUDA streams × 64-model vectorized banks")
    print(f"Architecture        : {nsl['X_train'].shape[1]}→64→64→64→5 (NSL-KDD)")
    print(f"                      {unsw['X_train'].shape[1]}→64→64→64→5 (UNSW-NB15)")
    print(f"Activation          : ReLU")
    print(f"Output              : Softmax-equivalent categorical objective")
    print(f"Dropout             : {DROPOUT_RATE}")
    print(f"Epochs/model        : {EPOCHS}")
    print(f"Batch size/model    : {BATCH_SIZE}")
    print(f"Learning rate       : {LEARNING_RATE}")
    print(f"Weight decay        : {WEIGHT_DECAY}")
    print(f"Adam betas / eps    : {ADAM_BETAS} / {ADAM_EPS}")
    print(f"Data split          : 60/15/25, seed {DATA_SPLIT_SEED}")
    print(f"Precision           : FP32 (AMP={USE_AMP}, TF32 disabled)")
    print(f"Total model-epochs  : {TOTAL_MODEL_EPOCHS}")
    print("-" * 100)
    print(
        f"NSL-KDD shapes      : train {nsl['X_train'].shape}, "
        f"val {nsl['X_val'].shape}, test {nsl['X_test'].shape}"
    )
    print(f"NSL-KDD classes     : {nsl['classes']}")
    print(
        f"UNSW-NB15 shapes    : train {unsw['X_train'].shape}, "
        f"val {unsw['X_val'].shape}, test {unsw['X_test'].shape}"
    )
    print(f"UNSW-NB15 classes   : {unsw['classes']}")
    print("=" * 100)
    print()


def main():
    global experiment_start, overall_bar, dataset_bars

    # Start clean logs for this run.
    for p in [PER_SEED_PATH, EPOCH_LOG_PATH, SUMMARY_PATH, ERROR_PATH]:
        if p.exists():
            p.unlink()

    # Data preprocessing stays on CPU and is done once.
    nsl = preprocess_nsl()
    unsw = preprocess_unsw()

    if len(nsl["classes"]) != NUM_CLASSES:
        raise RuntimeError(
            f"NSL-KDD has {len(nsl['classes'])} classes after preprocessing; "
            f"expected {NUM_CLASSES}: {nsl['classes']}"
        )
    if len(unsw["classes"]) != NUM_CLASSES:
        raise RuntimeError(
            f"UNSW-NB15 has {len(unsw['classes'])} classes after preprocessing; "
            f"expected {NUM_CLASSES}: {unsw['classes']}"
        )

    print_run_header(nsl, unsw)

    experiment_start = time.perf_counter()

    # Three live progress bars:
    #   1) total 2,560 model-epochs
    #   2) NSL-KDD bank epochs
    #   3) UNSW-NB15 bank epochs
    tqdm.set_lock(PROGRESS_LOCK)

    overall_bar = tqdm(
        total=TOTAL_MODEL_EPOCHS,
        desc="ALL 128 MODELS",
        unit="model-epoch",
        position=0,
        dynamic_ncols=True,
        leave=True,
    )

    dataset_bars["NSL-KDD"] = tqdm(
        total=EPOCHS,
        desc="NSL-KDD 64 models",
        unit="epoch",
        position=1,
        dynamic_ncols=True,
        leave=True,
    )

    dataset_bars["UNSW-NB15"] = tqdm(
        total=EPOCHS,
        desc="UNSW-NB15 64 models",
        unit="epoch",
        position=2,
        dynamic_ncols=True,
        leave=True,
    )

    all_results = []
    errors = []

    # Both dataset banks train concurrently on separate CUDA streams.
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="gpu-bank",
    ) as executor:
        futures = {
            executor.submit(train_bank, nsl): "NSL-KDD",
            executor.submit(train_bank, unsw): "UNSW-NB15",
        }

        for future in as_completed(futures):
            dataset_name = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                with PROGRESS_LOCK:
                    tqdm.write(
                        f"[{dataset_name}] all 64 models finished "
                        f"{EPOCHS} epochs and test evaluation."
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
                        f"\nFAILED DATASET BANK: {dataset_name}\n"
                        f"{tb}"
                    )

    for bar in dataset_bars.values():
        bar.close()
    overall_bar.close()

    elapsed = time.perf_counter() - experiment_start

    if errors:
        pd.DataFrame(errors).to_csv(ERROR_PATH, index=False)

    if len(all_results) != 2:
        raise RuntimeError(
            f"Only {len(all_results)}/2 dataset banks completed. "
            f"See {ERROR_PATH}"
        )

    save_outputs(all_results)

    print(f"\nTotal wall time: {fmt_time(elapsed)}")
    print(f"Final GPU state: {query_nvidia_smi()}")
    print("DONE.")


if __name__ == "__main__":
    main()
