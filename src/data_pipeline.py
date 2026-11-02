"""Dataset preparation following the published Sharma et al. workflow."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

SEED = 42

NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "class", "difficulty",
]

NSL_ATTACK_MAP = {
    **dict.fromkeys(
        ["apache2", "back", "land", "mailbomb", "neptune", "pod", "processtable",
         "smurf", "teardrop", "udpstorm", "worm"], "DoS"
    ),
    **dict.fromkeys(["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"], "Probe"),
    **dict.fromkeys(
        ["ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf",
         "sendmail", "snmpgetattack", "snmpguess", "spy", "warezclient", "warezmaster",
         "xlock", "xsnoop"], "R2L"
    ),
    **dict.fromkeys(
        ["buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm"], "U2R"
    ),
}

CLASS_ORDER = {
    "nsl": ["DoS", "Normal", "Probe", "R2L", "U2R"],
    "unsw": ["DoS", "Exploits", "Fuzzers", "Generic", "Normal"],
}

# The paper reports the final redundant-feature removals. Its text does not
# publish a complete fitted selector or preprocessing artifact.
PAPER_DROPS = {
    "nsl": [
        "srv_serror_rate", "dst_host_srv_rerror_rate", "num_root",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "srv_rerror_rate",
    ],
    "unsw": ["ct_src_dport_ltm", "loss", "dwin", "ct_ftp_cmd", "label", "ct_srv_dst"],
}


def _read_nsl(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=NSL_COLUMNS, skipinitialspace=True)
    attack = frame.pop("class").astype(str).str.strip().str.lower()
    frame.pop("difficulty")
    labels = attack.map(lambda x: "Normal" if x == "normal" else NSL_ATTACK_MAP.get(x))
    unknown = sorted(attack[labels.isna()].unique().tolist())
    if unknown:
        raise ValueError(f"Unmapped NSL-KDD attack labels: {unknown}")
    frame["__target__"] = labels
    return frame


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_unsw(train_path: str | Path, test_path: str | Path) -> pd.DataFrame:
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    frame = pd.concat([train, test], ignore_index=True)
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.drop(columns=[c for c in ["id"] if c in frame.columns])
    if "attack_cat" not in frame.columns:
        raise ValueError("UNSW files must contain the multiclass 'attack_cat' target.")
    target = frame["attack_cat"].fillna("").astype(str).str.strip().str.lower()
    if "label" in frame.columns:
        normal = pd.to_numeric(frame["label"], errors="coerce").fillna(1).eq(0)
        target = target.mask(target.eq("") & normal, "normal")
    frame["__target__"] = target.map(
        {"dos": "DoS", "exploits": "Exploits", "fuzzers": "Fuzzers", "generic": "Generic", "normal": "Normal"}
    )
    frame = frame.loc[frame["__target__"].notna()].copy()

    cap = 50_000
    pieces = []
    for label, group in frame.groupby("__target__", sort=False):
        if label in {"Normal", "Generic"} and len(group) > cap:
            group = group.sample(n=cap, random_state=SEED)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def prepare_dataset(
    dataset: str,
    nsl_train: str | Path | None = None,
    unsw_train: str | Path | None = None,
    unsw_test: str | Path | None = None,
) -> dict[str, Any]:
    """Return processed arrays plus an audit manifest for one benchmark."""
    dataset = dataset.lower()
    if dataset == "nsl":
        if nsl_train is None:
            raise ValueError("Pass --nsl-train with the KDDTrain+.txt file.")
        input_paths = [Path(nsl_train)]
        frame = _read_nsl(nsl_train)
    elif dataset == "unsw":
        if unsw_train is None or unsw_test is None:
            raise ValueError("Pass both --unsw-train and --unsw-test CSV files.")
        input_paths = [Path(unsw_train), Path(unsw_test)]
        frame = _read_unsw(unsw_train, unsw_test)
    else:
        raise ValueError("dataset must be 'nsl' or 'unsw'.")

    y_text = frame.pop("__target__").astype(str)
    class_names = CLASS_ORDER[dataset]
    unexpected = sorted(set(y_text) - set(class_names))
    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")
    y = np.asarray([class_names.index(label) for label in y_text], dtype=np.int64)

    # Replicate the paper's stated sequence: label encoding and min-max scaling
    # precede its 60/15/25 split. This can expose test-set distribution to the
    # unsupervised transformations; it is disclosed in the result report.
    feature_frame = frame.drop(columns=[c for c in ["attack_cat", "class", "difficulty"] if c in frame.columns])
    paper_drops = PAPER_DROPS[dataset]
    applied_drops = [c for c in paper_drops if c in feature_frame.columns]
    missing_reported_drops = [c for c in paper_drops if c not in feature_frame.columns]
    feature_frame = feature_frame.drop(columns=applied_drops)

    encoders: dict[str, dict[str, int]] = {}
    for column in feature_frame.columns:
        if not pd.api.types.is_numeric_dtype(feature_frame[column]):
            series = feature_frame[column].fillna("__missing__").astype(str).str.strip()
            categories = sorted(series.unique().tolist())
            mapping = {value: index for index, value in enumerate(categories)}
            feature_frame[column] = series.map(mapping).astype(np.float64)
            encoders[column] = mapping
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in feature_frame.columns:
        if feature_frame[column].isna().any():
            feature_frame[column] = feature_frame[column].fillna(feature_frame[column].median())

    scaler = MinMaxScaler()
    X = scaler.fit_transform(feature_frame.to_numpy(dtype=np.float64)).astype(np.float32)
    counts = {label: int(np.sum(y == index)) for index, label in enumerate(class_names)}
    return {
        "dataset": dataset,
        "X": X,
        "y": y,
        "feature_names": feature_frame.columns.astype(str).tolist(),
        "class_names": class_names,
        "class_counts": counts,
        "source_rows_after_filter_and_sampling": int(len(frame)),
        "source_columns_before_selection": int(frame.shape[1]),
        "selected_feature_count": int(X.shape[1]),
        "applied_paper_feature_drops": applied_drops,
        "reported_drops_missing_from_source_schema": missing_reported_drops,
        "categorical_encodings": encoders,
        "minmax_min": scaler.data_min_.tolist(),
        "minmax_max": scaler.data_max_.tolist(),
        "seed": SEED,
        "input_files": [
            {"name": path.name, "sha256": _sha256(path)} for path in input_paths
        ],
    }
