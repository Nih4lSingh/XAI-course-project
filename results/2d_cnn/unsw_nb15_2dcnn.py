"""Paper-like preprocessing for the UNSW-NB15 2D-CNN reproduction.

The paper combines the 257,673 records, keeps five classes, caps Normal and
Generic at 50,000 randomly selected records each, and retains every Exploits,
DoS, and Fuzzers record. After label encoding and min-max normalization, 38
predictors are padded with eleven zeros and reshaped to 7 x 7 x 1.

No package installation or network access is performed by this module.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"

CLASS_NAMES = ["DoS", "Exploits", "Fuzzers", "Generic", "Normal"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_LIMITS = {
    "DoS": None,
    "Exploits": None,
    "Fuzzers": None,
    "Generic": 50_000,
    "Normal": 50_000,
}

CATEGORICAL_COLUMNS = ["proto", "service", "state"]

# These are the four real predictor columns named by the paper. Its other two
# names are ``label`` (the binary label) and ``loss`` (not present in either
# supplied CSV). Keeping sloss and dloss is the only interpretation that gives
# the paper's stated 38 predictors and therefore exactly 11 padding values.
PAPER_DROPPED_PREDICTORS = [
    "ct_src_dport_ltm",
    "dwin",
    "ct_ftp_cmd",
    "ct_srv_dst",
]
PAPER_REPORTED_DROPS = [
    "ct_src_dport_ltm",
    "loss",
    "dwin",
    "ct_ftp_cmd",
    "label",
    "ct_srv_dst",
]


@dataclass
class FeatureData:
    features: np.ndarray
    labels: np.ndarray
    metadata: dict[str, Any]


def read_unsw_nb15(data_dir: Path) -> pd.DataFrame:
    """Read and combine the two supplied official-split CSV files."""
    paths = [data_dir / TRAIN_FILE, data_dir / TEST_FILE]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing UNSW-NB15 file(s): " + ", ".join(missing))

    frames = [pd.read_csv(path) for path in paths]
    if list(frames[0].columns) != list(frames[1].columns):
        raise ValueError("The UNSW-NB15 CSV files do not have identical columns.")
    frame = pd.concat(frames, ignore_index=True)
    if frame.shape != (257_673, 45):
        raise ValueError(
            f"Expected the supplied combined dataset to be 257673 x 45, got {frame.shape}."
        )
    if frame.isna().any().any():
        raise ValueError("Missing values were found in the supplied UNSW-NB15 files.")

    expected = {
        "id",
        "proto",
        "service",
        "state",
        "attack_cat",
        "label",
        *PAPER_DROPPED_PREDICTORS,
    }
    absent = sorted(expected - set(frame.columns))
    if absent:
        raise ValueError("Required UNSW-NB15 columns are missing: " + ", ".join(absent))

    normal_mask = frame["attack_cat"].eq("Normal")
    inconsistent = (normal_mask & frame["label"].ne(0)) | (
        ~normal_mask & frame["label"].ne(1)
    )
    if inconsistent.any():
        raise ValueError("attack_cat and binary label disagree in the supplied data.")
    return frame


def select_paper_classes(frame: pd.DataFrame, sample_seed: int) -> pd.DataFrame:
    """Apply the paper's five-class selection and 50K caps reproducibly."""
    rng = np.random.default_rng(sample_seed)
    selected_indices: list[np.ndarray] = []
    for class_name in CLASS_NAMES:
        indices = np.flatnonzero(frame["attack_cat"].to_numpy() == class_name)
        limit = CLASS_LIMITS[class_name]
        if limit is not None:
            if len(indices) < limit:
                raise ValueError(
                    f"Class {class_name} has only {len(indices)} rows; {limit} required."
                )
            indices = rng.choice(indices, size=limit, replace=False)
        selected_indices.append(indices)

    combined = np.concatenate(selected_indices)
    rng.shuffle(combined)
    selected = frame.iloc[combined].reset_index(drop=True)
    expected_counts = {
        "DoS": 16_353,
        "Exploits": 44_525,
        "Fuzzers": 24_246,
        "Generic": 50_000,
        "Normal": 50_000,
    }
    actual_counts = selected["attack_cat"].value_counts().to_dict()
    if actual_counts != expected_counts:
        raise AssertionError(
            f"Paper class sampling produced {actual_counts}, expected {expected_counts}."
        )
    return selected


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
        values = encoded[column].astype(str).map(lookup)
        if values.isna().any():
            raise ValueError(f"Unmapped categorical values found in {column}.")
        encoded[column] = values.astype(np.float32)
    return encoded


def fit_minmax(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return frame.min(axis=0).astype(np.float64), frame.max(axis=0).astype(np.float64)


def apply_minmax(
    frame: pd.DataFrame, minimum: pd.Series, maximum: pd.Series
) -> np.ndarray:
    denominator = (maximum - minimum).replace(0.0, 1.0)
    scaled = ((frame.astype(np.float64) - minimum) / denominator).clip(0.0, 1.0)
    return scaled.to_numpy(dtype=np.float32)


def stratified_partition(
    labels: np.ndarray, fractions: tuple[float, ...], seed: int
) -> tuple[np.ndarray, ...]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Split fractions must sum to one, got {fractions}.")
    rng = np.random.default_rng(seed)
    partitions: list[list[np.ndarray]] = [[] for _ in fractions]
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        cuts: list[int] = []
        used = 0
        for fraction in fractions[:-1]:
            used += int(round(len(indices) * fraction))
            cuts.append(min(used, len(indices)))
        for destination, piece in zip(partitions, np.split(indices, cuts)):
            destination.append(piece)

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


def prepare_paper_features(data_dir: Path, sample_seed: int) -> FeatureData:
    """Create the paper-like 185,124-row, 7x7 input tensor."""
    full = read_unsw_nb15(data_dir)
    selected = select_paper_classes(full, sample_seed)
    labels = (
        selected["attack_cat"].map(CLASS_TO_ID).to_numpy(dtype=np.int64)
    )

    predictors = selected.drop(columns=["id", "attack_cat", "label"])
    predictors = predictors.drop(columns=PAPER_DROPPED_PREDICTORS)
    if predictors.shape[1] != 38:
        raise AssertionError(f"Expected 38 retained predictors, got {predictors.shape[1]}.")

    category_maps = fit_category_maps(predictors)
    predictors = encode_categories(predictors, category_maps)
    minimum, maximum = fit_minmax(predictors)
    features = apply_minmax(predictors, minimum, maximum)
    padding_count = 49 - features.shape[1]
    if padding_count != 11:
        raise AssertionError(f"Expected 11 zero-padding values, got {padding_count}.")
    features = np.pad(features, ((0, 0), (0, padding_count)), constant_values=0.0)

    metadata = {
        "protocol": "paper_random_split",
        "source_files": [str((data_dir / TRAIN_FILE).resolve()), str((data_dir / TEST_FILE).resolve())],
        "combined_source_rows": len(full),
        "selected_rows": len(selected),
        "sample_seed": sample_seed,
        "class_names": CLASS_NAMES,
        "class_to_id": CLASS_TO_ID,
        "class_counts": class_counts(labels),
        "categorical_columns": CATEGORICAL_COLUMNS,
        "category_maps": category_maps,
        "paper_reported_drops": PAPER_REPORTED_DROPS,
        "actual_dropped_predictors": PAPER_DROPPED_PREDICTORS,
        "feature_names": predictors.columns.tolist() + [
            f"__zero_padding_{index + 1}__" for index in range(padding_count)
        ],
        "retained_predictor_count": predictors.shape[1],
        "zero_padding_count": padding_count,
        "minima": minimum.to_dict(),
        "maxima": maximum.to_dict(),
        "preprocessing_scope": "selected dataset before the paper's random split",
        "important_note": (
            "The paper names a dropped column called 'loss', but the supplied files "
            "contain sloss and dloss. Both are retained because this is the only "
            "interpretation consistent with 38 predictors and 11 padding values."
        ),
    }
    return FeatureData(
        features.reshape((-1, 1, 7, 7)).astype(np.float32, copy=False),
        labels,
        metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the paper-like UNSW-NB15 7x7 preprocessing pipeline."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Datasets/UNSW-NB15"))
    parser.add_argument("--sample-seed", type=int, default=20260909)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = prepare_paper_features(args.data_dir, args.sample_seed)
    summary = {
        "feature_shape": list(data.features.shape),
        "label_shape": list(data.labels.shape),
        "value_range": [float(data.features.min()), float(data.features.max())],
        "class_counts": data.metadata["class_counts"],
        "retained_predictor_count": data.metadata["retained_predictor_count"],
        "zero_padding_count": data.metadata["zero_padding_count"],
        "note": data.metadata["important_note"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
