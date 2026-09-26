"""Shared vectorized-GPU components for the UNSW-NB15 reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np

import nsl_kdd_gpu_sweep as shared
import unsw_nb15_2dcnn as base


DEFAULT_BATCH_SIZES = (32, 64, 128)
DEFAULT_OPTIMIZERS = ("adamw", "adam_l2")


@dataclass(frozen=True)
class SweepConfig:
    batch_size: int
    optimizer: str

    @property
    def config_id(self) -> str:
        return f"batch{self.batch_size:03d}-{self.optimizer}"


def experiment_matrix(
    batch_sizes: Iterable[int] = DEFAULT_BATCH_SIZES,
    optimizers: Iterable[str] = DEFAULT_OPTIMIZERS,
) -> list[SweepConfig]:
    return [
        SweepConfig(batch_size, optimizer)
        for batch_size, optimizer in product(batch_sizes, optimizers)
    ]


select_random_seeds = shared.select_random_seeds
VectorizedCNN = shared.VectorizedCNN
_torch = shared._torch


def prepare_features(data_dir: Path, sample_seed: int) -> base.FeatureData:
    return base.prepare_paper_features(data_dir, sample_seed)


def build_seed_splits(
    labels: np.ndarray, seeds: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_rows = []
    validation_rows = []
    test_rows = []
    for seed in seeds:
        train_idx, validation_idx, test_idx = base.stratified_partition(
            labels, (0.60, 0.15, 0.25), seed
        )
        train_rows.append(train_idx)
        validation_rows.append(validation_idx)
        test_rows.append(test_idx)
    return np.stack(train_rows), np.stack(validation_rows), np.stack(test_rows)


def metrics_from_confusions(confusions: np.ndarray) -> dict[str, np.ndarray]:
    true_positive = np.diagonal(confusions, axis1=1, axis2=2).astype(np.float64)
    predicted = confusions.sum(axis=1)
    actual = confusions.sum(axis=2)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros_like(true_positive),
        where=actual != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return {
        "macro_precision": precision.mean(axis=1),
        "macro_recall": recall.mean(axis=1),
        "macro_f1": f1.mean(axis=1),
        **{
            metric_name: values[:, class_id]
            for class_id, class_name in enumerate(base.CLASS_NAMES)
            for metric_name, values in (
                (f"{class_name.lower()}_precision", precision),
                (f"{class_name.lower()}_recall", recall),
                (f"{class_name.lower()}_f1", f1),
            )
        },
    }
