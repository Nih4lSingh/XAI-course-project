"""Recreate the paper's 2D-CNN experiment on NSL-KDD.

Two evaluation protocols are available:

* paper: use KDDTrain+.txt only and make a stratified 60/15/25 split.
  The NSL-KDD difficulty field is retained so that removing the paper's six
  correlated predictors leaves exactly 36 inputs for a 6x6 image.
* official: train/validate on KDDTrain+.txt and test on KDDTest+.txt. The
  difficulty field is removed; the resulting 35 predictors are padded with
  one zero to obtain a 6x6 image.

TensorFlow is imported only when training begins, so --preprocess-only can be
used to audit the data pipeline before installing TensorFlow.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
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
]

ALL_COLUMNS = FEATURE_COLUMNS + ["label", "difficulty"]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# Section 4.2.1 of the paper names these six predictors explicitly.
PAPER_DROPPED_FEATURES = [
    "srv_serror_rate",
    "dst_host_srv_rerror_rate",
    "num_root",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "srv_rerror_rate",
]

CLASS_NAMES = ["DoS", "Normal", "Probe", "R2L", "U2R"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

ATTACK_TO_CLASS = {
    # Denial of service
    "apache2": "DoS",
    "back": "DoS",
    "land": "DoS",
    "mailbomb": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "processtable": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",
    # Probe
    "ipsweep": "Probe",
    "mscan": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "saint": "Probe",
    "satan": "Probe",
    # Remote to local
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "httptunnel": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "named": "R2L",
    "phf": "R2L",
    "sendmail": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    # User to root
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "ps": "U2R",
    "rootkit": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "normal": "Normal",
}


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any]


def read_nsl_kdd(path: Path) -> pd.DataFrame:
    """Read a headerless KDDTrain+/KDDTest+ file and validate its width."""
    frame = pd.read_csv(path, names=ALL_COLUMNS, header=None)
    if frame.shape[1] != len(ALL_COLUMNS):
        raise ValueError(
            f"Expected {len(ALL_COLUMNS)} columns in {path}, got {frame.shape[1]}."
        )
    if frame.isna().any().any():
        raise ValueError(f"Missing values found in {path}.")
    return frame


def map_attack_labels(labels: pd.Series) -> np.ndarray:
    mapped = labels.astype(str).str.rstrip(".").map(ATTACK_TO_CLASS)
    if mapped.isna().any():
        unknown = sorted(labels[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unmapped NSL-KDD attack labels: {unknown}")
    return mapped.map(CLASS_TO_ID).to_numpy(dtype=np.int64)


def fit_category_maps(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        column: sorted(frame[column].astype(str).unique().tolist())
        for column in CATEGORICAL_COLUMNS
    }


def encode_categories(
    frame: pd.DataFrame, category_maps: dict[str, list[str]]
) -> pd.DataFrame:
    encoded = frame.copy()
    for column, categories in category_maps.items():
        lookup = {value: index for index, value in enumerate(categories)}
        # Reserve the next integer for a category seen only in an official test set.
        unknown_id = len(categories)
        encoded[column] = (
            encoded[column].astype(str).map(lookup).fillna(unknown_id).astype(np.float32)
        )
    return encoded


def fit_minmax(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    minimum = frame.min(axis=0).astype(np.float64)
    maximum = frame.max(axis=0).astype(np.float64)
    return minimum, maximum


def apply_minmax(
    frame: pd.DataFrame,
    minimum: pd.Series,
    maximum: pd.Series,
    *,
    clip: bool,
) -> np.ndarray:
    denominator = (maximum - minimum).replace(0.0, 1.0)
    scaled = (frame.astype(np.float64) - minimum) / denominator
    if clip:
        scaled = scaled.clip(0.0, 1.0)
    return scaled.to_numpy(dtype=np.float32)


def stratified_partition(
    labels: np.ndarray,
    fractions: tuple[float, ...],
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Return deterministic, stratified indices for fractions summing to one."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {fractions}.")

    rng = np.random.default_rng(seed)
    partitions: list[list[np.ndarray]] = [[] for _ in fractions]
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        cut_points: list[int] = []
        used = 0
        for fraction in fractions[:-1]:
            used += int(round(len(indices) * fraction))
            cut_points.append(min(used, len(indices)))
        pieces = np.split(indices, cut_points)
        for partition, piece in zip(partitions, pieces):
            partition.append(piece)

    output = []
    for groups in partitions:
        indices = np.concatenate(groups)
        rng.shuffle(indices)
        output.append(indices)
    return tuple(output)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        name: int(np.sum(labels == class_id))
        for class_id, name in enumerate(CLASS_NAMES)
    }


def reshape_as_images(features: np.ndarray) -> np.ndarray:
    if features.shape[1] != 36:
        raise ValueError(f"A 6x6 input requires 36 features, got {features.shape[1]}.")
    return features.reshape((-1, 6, 6, 1)).astype(np.float32, copy=False)


def prepare_paper_protocol(train_path: Path, seed: int) -> PreparedData:
    """Approximate the paper's random 60/15/25 split of KDDTrain+."""
    frame = read_nsl_kdd(train_path)
    labels = map_attack_labels(frame["label"])

    # Retaining difficulty reconciles the raw 43-column files with the paper's
    # claimed 36 model inputs after its six feature removals.
    predictors = frame.drop(columns=["label"]).drop(columns=PAPER_DROPPED_FEATURES)
    category_maps = fit_category_maps(predictors)
    predictors = encode_categories(predictors, category_maps)

    if predictors.shape[1] != 36:
        raise AssertionError(f"Paper protocol produced {predictors.shape[1]} features.")

    minimum, maximum = fit_minmax(predictors)
    features = apply_minmax(predictors, minimum, maximum, clip=True)
    train_idx, val_idx, test_idx = stratified_partition(
        labels, (0.60, 0.15, 0.25), seed
    )

    metadata = {
        "protocol": "paper",
        "source_files": [str(train_path.resolve())],
        "feature_names": predictors.columns.tolist(),
        "paper_dropped_features": PAPER_DROPPED_FEATURES,
        "difficulty_retained": True,
        "zero_padding_features": 0,
        "category_maps": category_maps,
        "minima": minimum.to_dict(),
        "maxima": maximum.to_dict(),
        "class_names": CLASS_NAMES,
        "class_to_id": CLASS_TO_ID,
        "split_counts": {
            "train": class_counts(labels[train_idx]),
            "validation": class_counts(labels[val_idx]),
            "test": class_counts(labels[test_idx]),
        },
        "important_note": (
            "The paper does not explain how 36 predictors remain after dropping "
            "six of the 41 documented predictors. This reconstruction retains the "
            "raw NSL-KDD difficulty field to make the stated 6x6 input possible."
        ),
    }
    return PreparedData(
        reshape_as_images(features[train_idx]),
        labels[train_idx],
        reshape_as_images(features[val_idx]),
        labels[val_idx],
        reshape_as_images(features[test_idx]),
        labels[test_idx],
        metadata,
    )


def prepare_official_protocol(
    train_path: Path, test_path: Path, seed: int
) -> PreparedData:
    """Use the official NSL-KDD test split with leakage-safe preprocessing."""
    train_frame = read_nsl_kdd(train_path)
    test_frame = read_nsl_kdd(test_path)
    train_labels = map_attack_labels(train_frame["label"])
    test_labels = map_attack_labels(test_frame["label"])

    train_predictors = train_frame.drop(columns=["label", "difficulty"])
    test_predictors = test_frame.drop(columns=["label", "difficulty"])
    train_predictors = train_predictors.drop(columns=PAPER_DROPPED_FEATURES)
    test_predictors = test_predictors.drop(columns=PAPER_DROPPED_FEATURES)

    category_maps = fit_category_maps(train_predictors)
    train_predictors = encode_categories(train_predictors, category_maps)
    test_predictors = encode_categories(test_predictors, category_maps)
    minimum, maximum = fit_minmax(train_predictors)
    x_all_train = apply_minmax(
        train_predictors, minimum, maximum, clip=True
    )
    x_test = apply_minmax(test_predictors, minimum, maximum, clip=True)

    # The paper describes a 6x6 input but leaves only 35 legitimate predictors.
    # Zero-padding is explicit here rather than silently retaining difficulty.
    x_all_train = np.pad(x_all_train, ((0, 0), (0, 1)), constant_values=0.0)
    x_test = np.pad(x_test, ((0, 0), (0, 1)), constant_values=0.0)
    feature_names = train_predictors.columns.tolist() + ["__zero_padding__"]

    train_idx, val_idx = stratified_partition(train_labels, (0.80, 0.20), seed)
    metadata = {
        "protocol": "official",
        "source_files": [str(train_path.resolve()), str(test_path.resolve())],
        "feature_names": feature_names,
        "paper_dropped_features": PAPER_DROPPED_FEATURES,
        "difficulty_retained": False,
        "zero_padding_features": 1,
        "category_maps": category_maps,
        "minima": minimum.to_dict(),
        "maxima": maximum.to_dict(),
        "class_names": CLASS_NAMES,
        "class_to_id": CLASS_TO_ID,
        "split_counts": {
            "train": class_counts(train_labels[train_idx]),
            "validation": class_counts(train_labels[val_idx]),
            "test": class_counts(test_labels),
        },
        "important_note": (
            "This is a stricter benchmark than the paper: preprocessing is fit on "
            "KDDTrain+ only, difficulty is removed, and KDDTest+ remains unseen."
        ),
    }
    return PreparedData(
        reshape_as_images(x_all_train[train_idx]),
        train_labels[train_idx],
        reshape_as_images(x_all_train[val_idx]),
        train_labels[val_idx],
        reshape_as_images(x_test),
        test_labels,
        metadata,
    )


def build_model(learning_rate: float, weight_decay: float):
    """Build the 64-32-32 2D-CNN shown in Fig. 5 of the paper."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for training. Install requirements.txt, or "
            "run with --preprocess-only to audit the pipeline."
        ) from exc

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(6, 6, 1), name="nsl_kdd_6x6"),
            tf.keras.layers.Conv2D(
                64, (3, 3), padding="same", activation="relu", name="conv_64"
            ),
            tf.keras.layers.MaxPooling2D(
                (2, 2), padding="same", name="pool_1"
            ),
            tf.keras.layers.Conv2D(
                32, (3, 3), padding="same", activation="relu", name="conv_32_a"
            ),
            tf.keras.layers.MaxPooling2D(
                (2, 2), padding="same", name="pool_2"
            ),
            tf.keras.layers.Conv2D(
                32, (3, 3), padding="same", activation="relu", name="conv_32_b"
            ),
            tf.keras.layers.MaxPooling2D(
                (2, 2), padding="same", name="pool_3"
            ),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(5, activation="softmax", name="class_probabilities"),
        ],
        name="nsl_kdd_2dcnn",
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def classification_report(matrix: np.ndarray) -> pd.DataFrame:
    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    true_positives = np.diag(matrix).astype(np.float64)
    precision = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros_like(true_positives),
        where=predicted_counts != 0,
    )
    recall = np.divide(
        true_positives,
        true_counts,
        out=np.zeros_like(true_positives),
        where=true_counts != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return pd.DataFrame(
        {
            "class": CLASS_NAMES,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": true_counts,
        }
    )


def save_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def train_and_evaluate(
    data: PreparedData,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for training. Install requirements.txt, or "
            "run with --preprocess-only to audit the pipeline."
        ) from exc

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Deterministic kernels are not available on every TensorFlow/device pair.
        pass

    model = build_model(learning_rate, weight_decay)
    model.summary()
    history = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=2,
    )

    probabilities = model.predict(data.x_test, batch_size=batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(data.y_test, predictions)
    report = classification_report(matrix)
    accuracy = float(np.trace(matrix) / np.sum(matrix))

    model.save(output_dir / "nsl_kdd_2dcnn.keras")
    pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv", index_label="actual\\predicted"
    )
    report.to_csv(output_dir / "classification_report.csv", index=False)

    run_metadata = dict(data.metadata)
    run_metadata["training"] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "loss": "sparse_categorical_crossentropy",
        "seed": seed,
        "test_accuracy": accuracy,
    }
    save_json(output_dir / "run_metadata.json", run_metadata)

    print(f"Test accuracy: {accuracy:.6f}")
    print(report.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the paper's 2D-CNN on NSL-KDD."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Datasets/NSL-KDD"),
        help="Directory containing KDDTrain+.txt and KDDTest+.txt.",
    )
    parser.add_argument(
        "--protocol",
        choices=("paper", "official"),
        default="paper",
        help="paper: random 60/15/25 split; official: untouched KDDTest+ test set.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nsl_kdd"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The paper does not report batch size; Keras' conventional 32 is used.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Validate preprocessing and save metadata without importing TensorFlow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    train_path = args.data_dir / "KDDTrain+.txt"
    test_path = args.data_dir / "KDDTest+.txt"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    if args.protocol == "official" and not test_path.is_file():
        raise FileNotFoundError(test_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.protocol == "paper":
        data = prepare_paper_protocol(train_path, args.seed)
    else:
        data = prepare_official_protocol(train_path, test_path, args.seed)

    print(
        "Prepared arrays:",
        f"train={data.x_train.shape}",
        f"validation={data.x_val.shape}",
        f"test={data.x_test.shape}",
    )
    save_json(args.output_dir / "preprocessing.json", data.metadata)
    if args.preprocess_only:
        print(f"Saved preprocessing audit to {args.output_dir / 'preprocessing.json'}")
        return

    train_and_evaluate(
        data,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
