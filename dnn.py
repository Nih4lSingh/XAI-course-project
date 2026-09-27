"""
Explainable DNN-based Intrusion Detection System for IoT Networks
Implementation of the DNN portion of:
Sharma, B., Sharma, L., Lal, C., & Roy, S. (2024).
"Explainable artificial intelligence for intrusion detection in IoT networks:
A deep learning based approach." Expert Systems With Applications, 238, 121751.

This script implements, for the NSL-KDD dataset:
  1. Data preprocessing (label encoding + min-max normalization)
  2. Attack-class grouping (23 attack types -> Probe, DoS, R2L, U2R, Normal)
  3. Correlation-based (Pearson) filter feature selection, threshold = 0.95
  4. DNN model: 3 hidden dense layers (64 neurons, ReLU) + softmax output,
     Adam optimizer, sparse_categorical_crossentropy loss, 20 epochs
  5. Evaluation: accuracy, precision, recall, F1-score, confusion matrix
  6. Model explanation with LIME (local) and SHAP (local + global)

Run:
    python dnn_ids.py
Requires KDDTrain.txt / KDDTest.txt (NSL-KDD, 41 features + label + difficulty)
in the same directory. Falls back to sklearn's synthetic data only if absent
(clearly flagged), so real results require the actual files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import lime
import lime.lime_tabular
import shap

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# --------------------------------------------------------------------------
# 1. Column names (standard 41 NSL-KDD features + class + difficulty score)
# --------------------------------------------------------------------------
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "class", "difficulty",
]

# Mapping of the 23 raw attack labels into the paper's 4 attack classes
ATTACK_MAP = {
    "normal": "Normal",
    # DoS
    "apache2": "DoS", "back": "DoS", "land": "DoS", "mailbomb": "DoS",
    "neptune": "DoS", "pod": "DoS", "processtable": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "udpstorm": "DoS", "worm": "DoS",
    # Probe
    "ipsweep": "Probe", "mscan": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "saint": "Probe", "satan": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "httptunnel": "R2L",
    "imap": "R2L", "multihop": "R2L", "named": "R2L", "phf": "R2L",
    "sendmail": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L", "xlock": "R2L",
    "xsnoop": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "ps": "U2R", "rootkit": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}

# The 6 highly-correlated (|PCC| > 0.95) features the paper drops for NSL-KDD
DROPPED_CORR_FEATURES = [
    "srv_serror_rate", "dst_host_srv_rerror_rate", "num_root",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "srv_rerror_rate",
]


def load_data(train_path="KDDTrain.txt", test_path="KDDTest.txt"):
    train = pd.read_csv(train_path, names=COLUMNS)
    test = pd.read_csv(test_path, names=COLUMNS)
    df = pd.concat([train, test], ignore_index=True)
    df = df.drop(columns=["difficulty"])
    return df


def group_attack_classes(df):
    df["class"] = df["class"].str.lower().map(ATTACK_MAP)
    df = df.dropna(subset=["class"])  # drop any unseen label just in case
    return df


def encode_and_normalize(df):
    """Label-encode categorical columns, then min-max scale everything to [0,1]."""
    cat_cols = ["protocol_type", "service", "flag"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df["class"])  # Normal/DoS/Probe/R2L/U2R -> 0..4
    X = df.drop(columns=["class"])

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, label_enc, scaler, encoders


def correlation_feature_selection(X, threshold=0.95, drop_list=None):
    """Pearson-correlation based filter feature selection (Eq. 4 in the paper).

    If drop_list is given (paper's reported dropped columns), use it directly
    so results match the paper; otherwise recompute from the correlation matrix.
    """
    corr = X.corr()
    if drop_list is not None:
        to_drop = [c for c in drop_list if c in X.columns]
    else:
        to_drop = set()
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if abs(corr.iloc[i, j]) > threshold:
                    to_drop.add(cols[j])
        to_drop = list(to_drop)
    X_reduced = X.drop(columns=to_drop)
    return X_reduced, to_drop, corr


def build_dnn(input_dim, num_classes):
    """3 hidden dense layers (64 neurons, ReLU) + softmax output, as in Fig. 4."""
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    report = pd.DataFrame(
        {"Precision": prec, "Recall": rec, "F1-Score": f1}, index=class_names
    )
    print("\nClassification report (per class):")
    print(report.round(3))
    print(f"\nOverall Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    return acc, report, cm


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix (Accuracy={cm.trace()/cm.sum():.3f})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def explain_with_lime(model, X_train, X_test, y_test, class_names, feature_names, n_instances=2):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
    )

    def predict_fn(x):
        return model.predict(x, verbose=0)

    for i in range(n_instances):
        idx = np.random.randint(0, len(X_test))
        exp = explainer.explain_instance(
            X_test.values[idx], predict_fn, num_features=10
        )
        actual = class_names[y_test[idx]]
        pred = class_names[np.argmax(predict_fn(X_test.values[idx:idx+1])[0])]
        print(f"\nLIME explanation for test instance {idx} "
              f"(Actual={actual}, Predicted={pred}):")
        for feature, weight in exp.as_list():
            print(f"    {feature}: {weight:+.4f}")
        fig = exp.as_pyplot_figure()
        fig.savefig(os.path.join(OUTPUT_DIR, f"lime_instance_{idx}.png"),
                     dpi=150, bbox_inches="tight")
        plt.close(fig)


def explain_with_shap(model, X_train, X_test, y_test, feature_names, class_names):
    """SHAP explanation exactly as described in Section 6.2 of the paper:

      - Local explanation: pick a particular test instance, compute its SHAP
        values, and plot a force plot showing each feature's contribution
        (Figs. 15/16 in the paper).
      - Global explanation: select 50 samples of the testing dataset, compute
        their SHAP values, and derive (a) a per-class summary/beeswarm plot
        (Figs. 17/18) and (b) a stacked bar "force plot" of mean(|SHAP value|)
        per feature across all classes (Figs. 19/20).
    """
    background = shap.kmeans(X_train.values, 50)  # compact background for KernelExplainer
    predict_fn = lambda x: model.predict(x, verbose=0)
    explainer = shap.KernelExplainer(predict_fn, background)

    # ---- Global explanation: exactly 50 samples of the testing dataset ----
    test_sample = X_test.sample(50, random_state=RANDOM_STATE)
    print("Computing SHAP values for 50 test samples (this takes a few minutes)...")
    shap_values = explainer.shap_values(test_sample.values, nsamples="auto")
    # shap_values: list of arrays (one per class) OR a single (n, features, classes) array
    if isinstance(shap_values, list):
        shap_list = shap_values
    else:
        shap_list = [shap_values[..., c] for c in range(shap_values.shape[-1])]

    # (a) Per-class summary (beeswarm) plot -- one per class, as in Figs. 17/18
    for c, cname in enumerate(class_names):
        plt.figure()
        shap.summary_plot(shap_list[c], test_sample, feature_names=feature_names,
                           show=False)
        plt.title(f"SHAP summary plot - {cname} class")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_summary_{cname}.png"),
                     dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Saved {len(class_names)} per-class SHAP summary plots "
          f"(shap_summary_<class>.png)")

    # (b) Stacked bar "force plot" across all classes, as in Figs. 19/20
    plt.figure()
    shap.summary_plot(shap_list, test_sample, feature_names=feature_names,
                       class_names=class_names, plot_type="bar", show=False)
    plt.title("SHAP feature importance across classes")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_force_plot_allclasses.png"),
                 dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved stacked per-class SHAP bar/force plot (shap_force_plot_allclasses.png)")

    # ---- Local explanation: one particular test instance (Figs. 15/16) ----
    idx = np.random.RandomState(RANDOM_STATE).randint(0, len(X_test))
    instance = X_test.iloc[[idx]]
    actual = class_names[y_test[idx]]
    pred_probs = predict_fn(instance.values)[0]
    pred_class = int(np.argmax(pred_probs))
    print(f"\nComputing local SHAP explanation for test instance {idx} "
          f"(Actual={actual}, Predicted={class_names[pred_class]})...")
    inst_shap = explainer.shap_values(instance.values, nsamples="auto")
    if isinstance(inst_shap, list):
        inst_shap_pred = inst_shap[pred_class][0]
    else:
        inst_shap_pred = inst_shap[0, :, pred_class]
    base_value = explainer.expected_value
    base_value = base_value[pred_class] if hasattr(base_value, "__len__") else base_value

    fig = plt.figure()
    shap.force_plot(
        base_value, inst_shap_pred, instance.iloc[0],
        feature_names=feature_names, matplotlib=True, show=False,
    )
    plt.title(f"SHAP local force plot - instance {idx} "
              f"(Actual={actual}, Predicted={class_names[pred_class]})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_force_instance_{idx}.png"),
                 dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved local SHAP force plot (shap_force_instance_{idx}.png)")


def main():
    print("Loading NSL-KDD dataset...")
    df = load_data()
    print(f"Raw records: {len(df)}")

    df = group_attack_classes(df)
    print("Attack class distribution:")
    print(df["class"].value_counts())

    X, y, label_enc, scaler, encoders = encode_and_normalize(df)
    class_names = list(label_enc.classes_)
    print(f"\nClasses (index order): {class_names}")

    # Correlation-based feature selection (use paper's reported drop list)
    X_reduced, dropped, corr = correlation_feature_selection(
        X, threshold=0.95, drop_list=DROPPED_CORR_FEATURES
    )
    print(f"\nDropped {len(dropped)} highly-correlated features: {dropped}")
    print(f"Remaining feature count: {X_reduced.shape[1]}")

    # Train (60%) / validation (15%) / test (25%) split, as in the paper
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_reduced, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.625, random_state=RANDOM_STATE, stratify=y_temp
    )  # 0.40 * 0.625 = 0.25 of total -> test; remaining 0.15 -> val

    print(f"\nTrain/Val/Test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    # Build and train the DNN
    model = build_dnn(input_dim=X_train.shape[1], num_classes=len(class_names))
    model.summary()

    history = model.fit(
        X_train.values, y_train,
        validation_data=(X_val.values, y_val),
        epochs=20,
        batch_size=256,
        verbose=2,
    )

    # Evaluate on the held-out test set
    y_pred_probs = model.predict(X_test.values, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc, report, cm = evaluate(y_test, y_pred, class_names)
    plot_confusion_matrix(cm, class_names, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # Accuracy / loss curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy vs. Epochs")
    axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss vs. Epochs")
    axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)

    # Model explanation
    print("\n" + "=" * 60)
    print("LIME explanation")
    print("=" * 60)
    explain_with_lime(model, X_train, X_test, y_test, class_names,
                       list(X_reduced.columns), n_instances=2)

    print("\n" + "=" * 60)
    print("SHAP explanation (50 test samples, per the paper - this takes a few minutes)")
    print("=" * 60)
    explain_with_shap(model, X_train, X_test, y_test, list(X_reduced.columns),
                       class_names)

    model.save(os.path.join(OUTPUT_DIR, "dnn_ids_model.keras"))
    print(f"\nDone. Model and plots saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()