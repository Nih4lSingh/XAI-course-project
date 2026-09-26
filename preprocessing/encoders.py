"""
Categorical Label Encoding & Target Encoding Module
Replication of Sharma et al. (2024)

Rules:
- Categorical features are converted to integer labels deterministically.
- Never use One-Hot Encoding for the primary replication.
- Target mappings match the paper's XAI section exactly.
- Encoders are fitted strictly on the data and saved for reproducible inference.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


# Exact 5-class mappings specified in Sharma et al. (2024) XAI section
NSL_KDD_CLASS_MAPPING = {
    0: "DoS",
    1: "Normal",
    2: "Probe",
    3: "R2L",
    4: "U2R"
}

NSL_KDD_ATTACK_TO_5CLASS = {
    # Normal
    "normal": 1,
    
    # DoS
    "apache2": 0, "back": 0, "land": 0, "neptune": 0, "mailbomb": 0,
    "pod": 0, "processtable": 0, "smurf": 0, "teardrop": 0, "udpstorm": 0,
    
    # Probe
    "ipsweep": 2, "mscan": 2, "nmap": 2, "portsweep": 2, "saint": 2, "satan": 2,
    
    # R2L
    "ftp_write": 3, "guess_passwd": 3, "httptunnel": 3, "imap": 3, "multihop": 3,
    "named": 3, "phf": 3, "sendmail": 3, "snmpgetattack": 3, "snmpguess": 3,
    "spy": 3, "warezclient": 3, "warezmaster": 3, "worm": 3, "xlock": 3, "xsnoop": 3,
    
    # U2R
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "ps": 4, "rootkit": 4,
    "sqlattack": 4, "xterm": 4
}

UNSW_NB15_CLASS_MAPPING = {
    0: "DoS",
    1: "Exploits",
    2: "Fuzzers",
    3: "Generic",
    4: "Normal"
}

UNSW_NB15_TARGET_MAP = {
    "dos": 0,
    "exploits": 1,
    "fuzzers": 2,
    "generic": 3,
    "normal": 4
}


class DeterministicLabelEncoder:
    """
    Encodes categorical string features into integer labels [0, K-1] based on sorted unique values.
    Supports serializing mapping to JSON.
    """
    def __init__(self, column_name: str):
        self.column_name = column_name
        self.classes_: List[str] = []
        self.mapping_: Dict[str, int] = {}
        self.inverse_mapping_: Dict[int, str] = {}

    def fit(self, series: pd.Series) -> "DeterministicLabelEncoder":
        unique_vals = sorted(series.dropna().astype(str).unique().tolist())
        self.classes_ = unique_vals
        self.mapping_ = {val: idx for idx, val in enumerate(unique_vals)}
        self.inverse_mapping_ = {idx: val for val, idx in self.mapping_.items()}
        return self

    def transform(self, series: pd.Series, unseen_value: int = -1) -> np.ndarray:
        str_series = series.astype(str)
        # Use get with default unseen_value for robust handling
        encoded = str_series.map(lambda x: self.mapping_.get(x, unseen_value)).values
        return encoded.astype(np.float64)

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        return self.fit(series).transform(series)

    def to_dict(self) -> Dict:
        return {
            "column_name": self.column_name,
            "classes": self.classes_,
            "mapping": self.mapping_
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DeterministicLabelEncoder":
        encoder = cls(data["column_name"])
        encoder.classes_ = data["classes"]
        encoder.mapping_ = data["mapping"]
        encoder.inverse_mapping_ = {int(v): k for k, v in data["mapping"].items()}
        return encoder


class CategoricalFeaturePipeline:
    """
    Manages deterministic label encoding across multiple categorical columns in a dataframe.
    """
    def __init__(self, categorical_columns: List[str]):
        self.categorical_columns = categorical_columns
        self.encoders: Dict[str, DeterministicLabelEncoder] = {}

    def fit(self, df: pd.DataFrame) -> "CategoricalFeaturePipeline":
        for col in self.categorical_columns:
            if col in df.columns:
                enc = DeterministicLabelEncoder(col)
                enc.fit(df[col])
                self.encoders[col] = enc
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col, enc in self.encoders.items():
            if col in df_out.columns:
                df_out[col] = enc.transform(df_out[col])
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {col: enc.to_dict() for col, enc in self.encoders.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CategoricalFeaturePipeline":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        pipeline = cls(list(data.keys()))
        for col, enc_dict in data.items():
            pipeline.encoders[col] = DeterministicLabelEncoder.from_dict(enc_dict)
        return pipeline
