"""
Min-Max Normalization Module
Replication of Sharma et al. (2024)

Formula:
    F_new = (F - F_min) / (F_max - F_min)

Maps all feature values strictly to [0, 1].
Handles zero-variance columns (F_max == F_min) by mapping to 0.0.
Saves scaling parameters to JSON for exact reproduction and production inference.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


class DeterministicMinMaxScaler:
    """
    Min-Max feature scaler adhering strictly to Sharma et al. Eq. (1).
    """
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names
        self.min_values_: Dict[str, float] = {}
        self.max_values_: Dict[str, float] = {}
        self.range_values_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "DeterministicMinMaxScaler":
        if self.feature_names is None:
            self.feature_names = list(df.columns)

        for col in self.feature_names:
            series = df[col].astype(np.float64)
            f_min = float(series.min())
            f_max = float(series.max())
            f_range = f_max - f_min
            
            self.min_values_[col] = f_min
            self.max_values_[col] = f_max
            self.range_values_[col] = f_range if f_range > 1e-12 else 1.0  # Avoid division by zero

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in self.feature_names:
            if col in df_out.columns:
                f_min = self.min_values_[col]
                f_range = self.range_values_[col]
                vals = df_out[col].astype(np.float64)
                scaled = (vals - f_min) / f_range
                # Clip strictly to [0.0, 1.0] to guard against numerical float precision
                df_out[col] = np.clip(scaled, 0.0, 1.0)
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "feature_names": self.feature_names,
            "min_values": self.min_values_,
            "max_values": self.max_values_,
            "range_values": self.range_values_
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DeterministicMinMaxScaler":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls(data["feature_names"])
        scaler.min_values_ = data["min_values"]
        scaler.max_values_ = data["max_values"]
        scaler.range_values_ = data["range_values"]
        return scaler
