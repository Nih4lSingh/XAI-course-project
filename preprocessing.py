"""
Shared preprocessing pipeline for the Sharma et al. (2024) replication.

Used by:
    - DNN
    - 1D-CNN

The DNN receives:
    (samples, features)

The 1D-CNN receives:
    (samples, features, 1)

The exact feature-selection implementation is an assumption where
the paper does not provide the authors' selection code.
"""

import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


SEED = 42

NSL_TRAIN_PATH = "/content/KDDTrain+.txt"
NSL_TEST_PATH = "/content/KDDTest+.txt"

UNSW_TRAIN_PATH = "/content/UNSW_NB15_training-set.csv"
UNSW_TEST_PATH = "/content/UNSW_NB15_testing-set.csv"

NSL_NUM_FEATURES = 36
UNSW_NUM_FEATURES = 38

# Overall split: 60% train / 15% validation / 25% test.
TEST_SIZE = 0.25
VAL_SIZE_WITHIN_TRAINING = 0.15 / 0.75


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def select_top_k_by_pearson(X, y, k):
    """
    Select k features with the largest absolute Pearson correlation
    with the encoded target.

    This is an implementation assumption because the paper does not
    provide the exact feature-selection code.
    """
    X_df = pd.DataFrame(X)
    y = np.asarray(y, dtype=float)

    correlations = []

    for col in X_df.columns:
        x = pd.to_numeric(X_df[col], errors="coerce").fillna(0).to_numpy()

        if np.std(x) == 0 or np.std(y) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr):
                corr = 0.0

        correlations.append(abs(corr))

    return np.argsort(correlations)[::-1][:k].astype(int)


def load_nsl_kdd():
    """Load and preprocess NSL-KDD for DNN / 1D-CNN."""

    set_seed()

    columns = [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty"
    ]

    train = pd.read_csv(NSL_TRAIN_PATH, header=None, names=columns)
    test = pd.read_csv(NSL_TEST_PATH, header=None, names=columns)

    # Columns reported as removed in the paper.
    columns_to_drop = [
        "srv_serror_rate",
        "dst_host_srv_rerror_rate",
        "num_root",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "srv_rerror_rate",
        "difficulty"
    ]

    train = train.drop(columns=columns_to_drop, errors="ignore")
    test = test.drop(columns=columns_to_drop, errors="ignore")

    dos = {
        "back", "land", "neptune", "pod", "smurf", "teardrop",
        "apache2", "udpstorm", "processtable", "mailbomb"
    }

    probe = {
        "ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"
    }

    r2l = {
        "ftp_write", "guess_passwd", "imap", "multihop", "phf",
        "spy", "warezclient", "warezmaster", "named", "sendmail",
        "snmpgetattack", "snmpguess", "xlock", "xsnoop"
    }

    u2r = {
        "buffer_overflow", "loadmodule", "perl", "rootkit",
        "httptunnel", "ps", "sqlattack", "xterm"
    }

    def map_label(label):
        label = str(label).strip().lower()

        if label == "normal":
            return "normal"
        if label in dos:
            return "dos"
        if label in probe:
            return "probe"
        if label in r2l:
            return "r2l"
        if label in u2r:
            return "u2r"

        return "normal"

    train["target"] = train["label"].apply(map_label)
    test["target"] = test["label"].apply(map_label)

    train = train.drop(columns=["label"])
    test = test.drop(columns=["label"])

    train_features = train.drop(columns=["target"])
    test_features = test.drop(columns=["target"])

    combined = pd.concat(
        [train_features, test_features],
        ignore_index=True
    )

    categorical_cols = combined.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if categorical_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        combined[categorical_cols] = encoder.fit_transform(
            combined[categorical_cols].astype(str)
        )

    combined = combined.apply(pd.to_numeric, errors="coerce")
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train_raw = combined.iloc[:len(train)].to_numpy()
    X_test_raw = combined.iloc[len(train):].to_numpy()

    label_encoder = LabelEncoder()

    all_labels = pd.concat(
        [train["target"], test["target"]],
        ignore_index=True
    )

    label_encoder.fit(all_labels)

    y_train_all = label_encoder.transform(train["target"])
    y_test = label_encoder.transform(test["target"])

    scaler = MinMaxScaler()

    X_all = scaler.fit_transform(
        np.vstack([X_train_raw, X_test_raw])
    )

    X_train_all = X_all[:len(train)]
    X_test = X_all[len(train):]

    feature_indices = select_top_k_by_pearson(
        X_train_all,
        y_train_all,
        NSL_NUM_FEATURES
    )

    X_train_all = X_train_all[:, feature_indices]
    X_test = X_test[:, feature_indices]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=VAL_SIZE_WITHIN_TRAINING,
        random_state=SEED,
        stratify=y_train_all
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_indices, label_encoder
    )


def load_unsw_nb15():
    """Load and preprocess UNSW-NB15 for DNN / 1D-CNN."""

    set_seed()

    train = pd.read_csv(UNSW_TRAIN_PATH)
    test = pd.read_csv(UNSW_TEST_PATH)

    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()

    train["attack_cat"] = (
        train["attack_cat"].astype(str).str.strip().str.lower()
    )
    test["attack_cat"] = (
        test["attack_cat"].astype(str).str.strip().str.lower()
    )

    class_names = [
        "normal",
        "generic",
        "exploits",
        "dos",
        "fuzzers"
    ]

    train = train[train["attack_cat"].isin(class_names)].copy()
    test = test[test["attack_cat"].isin(class_names)].copy()

    label_mapping = {
        "normal": 0,
        "generic": 1,
        "exploits": 2,
        "dos": 3,
        "fuzzers": 4
    }

    train["target"] = train["attack_cat"].map(label_mapping)
    test["target"] = test["attack_cat"].map(label_mapping)

    columns_to_drop = [
        "ct_src_dport_ltm",
        "loss",
        "dwin",
        "ct_ftp_cmd",
        "label",
        "ct_srv_dst",
        "attack_cat"
    ]

    train_features = train.drop(columns=["target"])
    test_features = test.drop(columns=["target"])

    train_features = train_features.drop(
        columns=columns_to_drop,
        errors="ignore"
    )
    test_features = test_features.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    combined = pd.concat(
        [train_features, test_features],
        ignore_index=True
    )

    categorical_cols = combined.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if categorical_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        combined[categorical_cols] = encoder.fit_transform(
            combined[categorical_cols].astype(str)
        )

    combined = combined.apply(pd.to_numeric, errors="coerce")
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train_raw = combined.iloc[:len(train)].to_numpy()
    X_test_raw = combined.iloc[len(train):].to_numpy()

    y_train_all = train["target"].astype(int).to_numpy()
    y_test = test["target"].astype(int).to_numpy()

    scaler = MinMaxScaler()

    X_all = scaler.fit_transform(
        np.vstack([X_train_raw, X_test_raw])
    )

    X_train_all = X_all[:len(train)]
    X_test = X_all[len(train):]

    feature_indices = select_top_k_by_pearson(
        X_train_all,
        y_train_all,
        UNSW_NUM_FEATURES
    )

    X_train_all = X_train_all[:, feature_indices]
    X_test = X_test[:, feature_indices]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=VAL_SIZE_WITHIN_TRAINING,
        random_state=SEED,
        stratify=y_train_all
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_indices, class_names
    )


def add_channel_dimension(X_train, X_val, X_test):
    """
    Convert tabular arrays for 1D-CNN.

    (samples, features) -> (samples, features, 1)
    """
    return (
        X_train[..., np.newaxis],
        X_val[..., np.newaxis],
        X_test[..., np.newaxis]
    )


if __name__ == "__main__":

    print("Loading NSL-KDD...")
    nsl = load_nsl_kdd()

    print("NSL-KDD shapes:")
    print("  Train:", nsl[0].shape)
    print("  Validation:", nsl[1].shape)
    print("  Test:", nsl[2].shape)

    print("\nLoading UNSW-NB15...")
    unsw = load_unsw_nb15()

    print("UNSW-NB15 shapes:")
    print("  Train:", unsw[0].shape)
    print("  Validation:", unsw[1].shape)
    print("  Test:", unsw[2].shape)

    nsl_1d = add_channel_dimension(
        nsl[0], nsl[1], nsl[2]
    )

    print("\nNSL-KDD 1D-CNN shapes:")
    print("  Train:", nsl_1d[0].shape)
    print("  Validation:", nsl_1d[1].shape)
    print("  Test:", nsl_1d[2].shape)
