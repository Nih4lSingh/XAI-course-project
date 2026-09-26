"""
Feature-to-Grid 2D Mapping Module
Replication of Sharma et al. (2024)

Rules:
- Transforms 1D network feature vectors into 2D grid tensors for Conv2D.
- NSL-KDD: 6 x 6 x 1 (36 elements, with 1 zero-padding element if 35 features).
- UNSW-NB15: 7 x 7 x 1 (49 elements, zero-padded up to 49).
- Row-major deterministic mapping; never randomly shuffle features.
- Saves the exact mapping coordinates: feature # -> (row, col).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class FeatureGridMapper:
    """
    Deterministically maps 1D feature arrays into 2D spatial grids (H x W x 1).
    """
    def __init__(self, feature_names: List[str], grid_size: int, dataset_name: str = "dataset"):
        self.feature_names = feature_names
        self.grid_size = grid_size
        self.grid_capacity = grid_size * grid_size
        self.dataset_name = dataset_name
        self.mapping_: Dict[str, Tuple[int, int]] = {}
        self.padded_count = max(0, self.grid_capacity - len(feature_names))

        self._build_mapping()

    def _build_mapping(self):
        """Constructs deterministic row-major coordinate map."""
        for idx, feat in enumerate(self.feature_names):
            if idx >= self.grid_capacity:
                break
            row = idx // self.grid_size
            col = idx % self.grid_size
            self.mapping_[feat] = (row, col)

        for pad_idx in range(self.padded_count):
            idx = len(self.feature_names) + pad_idx
            row = idx // self.grid_size
            col = idx % self.grid_size
            self.mapping_[f"zero_pad_{pad_idx}"] = (row, col)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms 2D array of shape (N, D) into (N, H, W, 1).
        """
        N, D = X.shape
        if D < self.grid_capacity:
            padding = np.zeros((N, self.grid_capacity - D), dtype=X.dtype)
            X_padded = np.hstack([X, padding])
        elif D > self.grid_capacity:
            X_padded = X[:, :self.grid_capacity]
        else:
            X_padded = X

        return X_padded.reshape((N, self.grid_size, self.grid_size, 1))

    def save_mapping(self, filepath: Union[str, Path]):
        """Saves coordinate mapping to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dataset": self.dataset_name,
            "grid_size": f"{self.grid_size}x{self.grid_size}",
            "grid_capacity": self.grid_capacity,
            "num_features": len(self.feature_names),
            "num_padding": self.padded_count,
            "feature_coordinates": {
                feat: {"row": coords[0], "col": coords[1], "grid_index": coords[0] * self.grid_size + coords[1]}
                for feat, coords in self.mapping_.items()
            }
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[GRID] Saved {self.dataset_name} grid mapping to {filepath}")


def get_nsl_kdd_grid_mapper(selected_features: List[str]) -> FeatureGridMapper:
    """Returns 6x6 grid mapper for NSL-KDD."""
    return FeatureGridMapper(selected_features, grid_size=6, dataset_name="nsl_kdd")


def get_unsw_nb15_grid_mapper(selected_features: List[str]) -> FeatureGridMapper:
    """Returns 7x7 grid mapper for UNSW-NB15."""
    return FeatureGridMapper(selected_features, grid_size=7, dataset_name="unsw_nb15")
