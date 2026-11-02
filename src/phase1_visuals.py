"""Generate local counterparts for Figures 1-20 in Sharma et al. (2024).

The article's numeric tables are transcribed in build_phase1_atlas.py. This
script creates fresh figures from the saved local Phase 1 models and data. It
does not copy or claim to reproduce the paper's original image files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from data_pipeline import _read_nsl, _read_unsw


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "outputs" / "source_code").is_dir()),
    SCRIPT_PATH.parents[1],
)
PHASE1_DIR = PROJECT_ROOT / "work" / "results" / "phase1"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "phase1_reconstruction" / "figures"
MODEL_ORDER = ["dnn", "cnn1d", "cnn2d"]
MODEL_LABELS = {"dnn": "DNN", "cnn1d": "1D-CNN", "cnn2d": "2D-CNN"}
MODEL_COLORS = {"dnn": "#214E75", "cnn1d": "#2A9D8F", "cnn2d": "#E9A23B"}
CLASS_COLORS = ["#247BA0", "#7353BA", "#E76F51", "#F4A261", "#43AA8B"]
DATASETS = {
    "nsl": {"name": "NSL-KDD", "paper_page": 5, "classes": ["DoS", "Normal", "Probe", "R2L", "U2R"]},
    "unsw": {"name": "UNSW-NB15", "paper_page": 6, "classes": ["DoS", "Exploits", "Fuzzers", "Generic", "Normal"]},
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataset(dataset: str):
    base = PHASE1_DIR / dataset
    bundle = np.load(base / "processed_data.npz")
    split = np.load(base / "split_indices.npz")
    manifest = joblib.load(base / "data_manifest.joblib")
    return base, bundle["X"], bundle["y"], split, manifest


def _preselection_feature_matrix(dataset: str):
    """Rebuild encoded predictors before the paper's correlation-based drops."""
    data_dir = PROJECT_ROOT / "work" / "datasets"
    if dataset == "nsl":
        source = data_dir / "KDDTrain+.txt"
        if not source.is_file():
            raise FileNotFoundError(f"Missing benchmark input required for Figure 2: {source}")
        frame = _read_nsl(source).drop(columns=["__target__"])
    else:
        train_path = data_dir / "UNSW_NB15_training-set.csv"
        test_path = data_dir / "UNSW_NB15_testing-set.csv"
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(f"Missing official UNSW split files required for Figure 3 under {data_dir}")
        frame = _read_unsw(train_path, test_path)
        frame = frame.drop(columns=[c for c in ["__target__", "attack_cat", "label", "id"] if c in frame.columns])

    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            series = frame[column].fillna("__missing__").astype(str).str.strip()
            mapping = {value: index for index, value in enumerate(sorted(series.unique().tolist()))}
            frame[column] = series.map(mapping).astype(np.float64)
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in frame.columns:
        if frame[column].isna().any():
            frame[column] = frame[column].fillna(frame[column].median())
    matrix = MinMaxScaler().fit_transform(frame.to_numpy(dtype=np.float64))
    return matrix.astype(np.float32), frame.columns.astype(str).tolist()


def _normalize_shap(values, n_rows: int, n_features: int, n_classes: int) -> np.ndarray:
    """Normalize SHAP 0.4x/0.5x return shapes to (rows, features, classes)."""
    if isinstance(values, list):
        parts = []
        for value in values:
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            parts.append(arr)
        arr = np.stack(parts, axis=-1)
    else:
        arr = np.asarray(values)
    if arr.ndim == 2 and n_classes == 1:
        arr = arr[..., None]
    if arr.shape == (n_classes, n_rows, n_features):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape == (n_rows, n_classes, n_features):
        arr = np.transpose(arr, (0, 2, 1))
    if arr.shape != (n_rows, n_features, n_classes):
        raise ValueError(f"Unexpected SHAP shape {arr.shape}; expected {(n_rows, n_features, n_classes)}")
    return arr.astype(np.float64)


def _save(fig, out: Path, number: int, title: str, paper_page: str, method: str):
    fig.tight_layout()
    path = out / f"figure_{number:02d}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"figure": number, "file": path.name, "title": title,
            "paper_page": paper_page, "local_method": method}


def figure_01(out: Path):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    labels = [
        (0.25, 1.68, 1.5, 1.35, "Benchmark data", "NSL-KDD\nUNSW-NB15"),
        (2.05, 1.68, 1.75, 1.35, "Preprocess", "Clean + encode\nMin-max scale"),
        (4.12, 1.68, 1.65, 1.35, "Select features", "Correlation filter\nReported drops"),
        (6.1, 1.68, 1.8, 1.35, "Train / test", "60 / 15 / 25\nDNN + CNNs"),
        (8.22, 1.68, 1.7, 1.35, "Predictions", "5-class output\nAccuracy + F1"),
        (10.25, 1.68, 2.25, 1.35, "Explain", "LIME local\nSHAP local + global"),
    ]
    for x, y, w, h, title, detail in labels:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1",
                             linewidth=1.4, edgecolor="#247BA0", facecolor="#EAF2F6")
        ax.add_patch(box)
        ax.text(x + w / 2, y + 0.91, title, ha="center", va="center", fontsize=10, weight="bold", color="#14324A")
        ax.text(x + w / 2, y + 0.43, detail, ha="center", va="center", fontsize=8.5, color="#25323B")
    for i in range(len(labels) - 1):
        x1 = labels[i][0] + labels[i][2] + 0.04
        x2 = labels[i + 1][0] - 0.05
        ax.add_patch(FancyArrowPatch((x1, 2.35), (x2, 2.35), arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.35, color="#5B6B75"))
    ax.text(6.5, 4.35, "Paper workflow reconstructed from the published method", ha="center", fontsize=14, weight="bold", color="#14324A")
    ax.text(6.5, 0.65, "Local run uses seed 42 because the paper does not publish a random seed or split indices.",
            ha="center", fontsize=9, color="#5B6B75")
    return _save(fig, out, 1, "Workflow", "p. 3", "Diagram reconstructed from the method description; details documented in the report.")


def figure_02_03(dataset: str, out: Path):
    X, features = _preselection_feature_matrix(dataset)
    corr = np.corrcoef(X, rowvar=False)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr = corr.copy()
    corr[mask] = np.nan
    fig, ax = plt.subplots(figsize=(14, 11))
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("white")
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=6.8)
    ax.set_yticks(range(len(features)), features, fontsize=6.8)
    ax.set_title(f"{DATASETS[dataset]['name']} feature correlation (local encoded/scaled data)", fontsize=13, weight="bold", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Pearson correlation")
    ax.set_xlabel("Upper triangle hidden to match the paper's lower-triangle presentation")
    number = 2 if dataset == "nsl" else 3
    page = "p. 7"
    return _save(fig, out, number, f"{DATASETS[dataset]['name']} correlation heatmap", page,
                 "Computed from the local data before redundant-feature removals, using the same class filtering, sampling, label encoding, and min-max scaling as Phase 1.")


def figure_04(out: Path):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    layers = [("Input", "35 / 38 selected features"), ("Dense + ReLU", "64 units"), ("Dense + ReLU", "64 units"),
              ("Dense + ReLU", "64 units"), ("Softmax output", "5 classes")]
    y_positions = [6.6, 5.25, 3.9, 2.55, 1.2]
    for idx, ((title, detail), y) in enumerate(zip(layers, y_positions)):
        w = 5.2 if idx else 6.6
        x = (10 - w) / 2
        ax.add_patch(FancyBboxPatch((x, y), w, 0.85, boxstyle="round,pad=0.03", facecolor="#EAF2F6",
                                    edgecolor="#247BA0", linewidth=1.3))
        ax.text(5, y + 0.55, title, ha="center", fontsize=10, weight="bold", color="#14324A")
        ax.text(5, y + 0.22, detail, ha="center", fontsize=8.5, color="#25323B")
        if idx < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((5, y - 0.05), (5, y_positions[idx + 1] + 0.9), arrowstyle="-|>", mutation_scale=13,
                                         linewidth=1.2, color="#5B6B75"))
    ax.text(5, 7.7, "DNN architecture described by Sharma et al.", ha="center", fontsize=13, weight="bold", color="#14324A")
    ax.text(5, 0.35, "Adam | sparse categorical cross-entropy | 20 epochs", ha="center", fontsize=9, color="#5B6B75")
    fig.tight_layout()
    return _save(fig, out, 4, "DNN architecture", "p. 7", "Reconstructed from the architecture and training details in the paper.")


def figure_05(out: Path):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")
    nodes = [
        ("Input feature grid", "6 x 6 x 1 (paper example)"),
        ("Conv2D + ReLU", "64 filters, 3 x 3"),
        ("Max pool", "2 x 2"),
        ("Conv2D + ReLU", "32 filters, 3 x 3"),
        ("Max pool", "2 x 2"),
        ("Conv2D + ReLU", "32 filters, 3 x 3"),
        ("Max pool + flatten", "2 x 2"),
        ("Softmax", "5 classes"),
    ]
    ys = np.linspace(7.05, 0.9, len(nodes))
    for i, ((title, detail), y) in enumerate(zip(nodes, ys)):
        ax.add_patch(FancyBboxPatch((2.7, y), 5.6, 0.62, boxstyle="round,pad=0.025", facecolor="#EAF2F6",
                                    edgecolor="#247BA0", linewidth=1.1))
        ax.text(5.5, y + 0.4, title, ha="center", fontsize=9.5, weight="bold", color="#14324A")
        ax.text(5.5, y + 0.14, detail, ha="center", fontsize=7.7, color="#25323B")
        if i < len(nodes) - 1:
            ax.add_patch(FancyArrowPatch((5.5, y - 0.02), (5.5, ys[i + 1] + 0.65), arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.0, color="#5B6B75"))
    ax.text(5.5, 7.78, "CNN architecture described by Sharma et al.", ha="center", fontsize=13, weight="bold", color="#14324A")
    ax.text(5.5, 0.28, "For this reconstruction, 35 or 38 selected features are zero-padded to the nearest square.", ha="center", fontsize=8.5, color="#5B6B75")
    return _save(fig, out, 5, "CNN architecture", "p. 8", "Architecture reconstructed from the paper; local tensor dimensions are disclosed.")


def figure_06(out: Path):
    fig, ax = plt.subplots(figsize=(6, 5.3))
    ax.axis("off")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.text(1.1, 1.91, "Predicted class", ha="center", fontsize=10, weight="bold")
    ax.text(0.06, 1.0, "True\nclass", ha="center", va="center", fontsize=9, weight="bold")
    cells = [((0.38, 1.02), "True negative\n(TN)", "#E76F51"), ((1.18, 1.02), "False positive\n(FP)", "#A9D6E5"),
             ((0.38, 0.22), "False negative\n(FN)", "#A9D6E5"), ((1.18, 0.22), "True positive\n(TP)", "#E76F51")]
    for (x, y), label, color in cells:
        ax.add_patch(plt.Rectangle((x, y), .76, .68, facecolor=color, edgecolor="white"))
        ax.text(x + .38, y + .34, label, ha="center", va="center", fontsize=9, weight="bold")
    ax.text(.76, .9, "C1", ha="center", va="top", fontsize=8)
    ax.text(1.56, .9, "C2", ha="center", va="top", fontsize=8)
    ax.text(.31, 1.36, "C1", ha="right", va="center", fontsize=8)
    ax.text(.31, .56, "C2", ha="right", va="center", fontsize=8)
    ax.set_title("Generic binary confusion-matrix key", fontsize=12, weight="bold", color="#14324A")
    return _save(fig, out, 6, "Confusion matrix definition", "p. 8", "Recreated schematic; multiclass results follow in Figures 7 and 9.")


def _metrics_and_confusion(dataset: str, out: Path, number: int):
    class_names = DATASETS[dataset]["classes"]
    metrics_by_model = {}
    matrices = {}
    for model in MODEL_ORDER:
        result = _read_json(PHASE1_DIR / dataset / model / "metrics.json")
        metrics_by_model[model] = result["classification_report"]
        matrices[model] = np.asarray(result["confusion_matrix"], dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, model in zip(axes, MODEL_ORDER):
        cm = matrices[model]
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        pct = cm / row_sums
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.0%}", ha="center", va="center",
                        fontsize=6.5, color="white" if pct[i, j] > .56 else "#14324A")
        ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(class_names)), class_names, fontsize=7)
        acc = _read_json(PHASE1_DIR / dataset / model / "metrics.json")["test_accuracy"]
        ax.set_title(f"{MODEL_LABELS[model]} | accuracy {acc:.3f}", fontsize=9, weight="bold")
        ax.set_xlabel("Predicted label", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("True label", fontsize=7)
    fig.suptitle(f"{DATASETS[dataset]['name']} local held-out confusion matrices", fontsize=13, weight="bold")
    conf_record = _save(fig, out, number, f"{DATASETS[dataset]['name']} confusion matrices", "p. 9", "Counts and row-normalized percentages computed from saved held-out predictions.")

    metric_names = [("precision", "Precision"), ("recall", "Recall"), ("f1-score", "F1-score")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    x = np.arange(len(class_names))
    width = .24
    for ax, (key, label) in zip(axes, metric_names):
        for i, model in enumerate(MODEL_ORDER):
            report = metrics_by_model[model]
            values = [report[c][key] for c in class_names]
            ax.bar(x + (i - 1) * width, values, width, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
        ax.set_xticks(x, class_names, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_title(label, fontsize=10, weight="bold")
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Score", fontsize=9)
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"{DATASETS[dataset]['name']} local class metrics", fontsize=13, weight="bold")
    metrics_record = _save(fig, out, number + 1, f"{DATASETS[dataset]['name']} precision, recall, F1", "p. 9", "Class metrics calculated from the local held-out predictions.")
    return [conf_record, metrics_record]


def figure_11(out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, dataset in enumerate(("nsl", "unsw")):
        for col, metric in enumerate(("accuracy", "loss")):
            ax = axes[row, col]
            for model in MODEL_ORDER:
                result = _read_json(PHASE1_DIR / dataset / model / "metrics.json")
                hist = result.get("history", {})
                train_values = hist.get(metric, [])
                val_values = hist.get("val_" + metric, [])
                if train_values:
                    epochs = np.arange(1, len(train_values) + 1)
                    color = MODEL_COLORS[model]
                    ax.plot(epochs, train_values, color=color, linewidth=1.8, label=MODEL_LABELS[model] + " train")
                    if val_values:
                        ax.plot(epochs, val_values, color=color, linewidth=1.25, linestyle="--", alpha=.8, label=MODEL_LABELS[model] + " validation")
            ax.set_title(f"{DATASETS[dataset]['name']} {metric}", fontsize=10, weight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.grid(alpha=.22)
            ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.suptitle("Training and validation curves from local 20-epoch runs", fontsize=13, weight="bold")
    return _save(fig, out, 11, "Accuracy and loss over epochs", "p. 11", "Saved Keras training histories for all three models and both datasets.")


def figure_12(out: Path):
    paper_fig = {"nsl": [0.992, 0.990, 0.985, 0.990, 0.989, 0.992],
                 "unsw": [0.777, 0.761, 0.770, 0.800, 0.799, 0.810]}
    paper_text = {"nsl": {"dnn": .993, "cnn1d": .992, "cnn2d": .994},
                  "unsw": {"dnn": .800, "cnn1d": .800, "cnn2d": .810}}
    names = ["KNN", "DT", "SVM", "DNN", "1D-CNN", "2D-CNN"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, dataset in zip(axes, ("nsl", "unsw")):
        vals = paper_fig[dataset]
        x = np.arange(len(names))
        ax.bar(x, vals, color="#E9A23B", alpha=.82, label="Published Fig. 12")
        ax.scatter(x[:3], vals[:3], marker="D", color="#14324A", s=26, zorder=3, label="Published classical models")
        local_names = ["dnn", "cnn1d", "cnn2d"]
        local_vals = [float(_read_json(PHASE1_DIR / dataset / m / "metrics.json")["test_accuracy"]) for m in local_names]
        ax.scatter(x[3:], local_vals, marker="s", s=42, color="#247BA0", edgecolor="white", linewidth=.5,
                   zorder=4, label="Local test result")
        for i, model in enumerate(local_names, start=3):
            ax.scatter(i, paper_text[dataset][model], marker="o", facecolor="none", edgecolor="#E76F51", s=62,
                       linewidth=1.5, zorder=5, label="Published result text" if i == 3 and dataset == "nsl" else None)
        ax.set_xticks(x, names, rotation=20, ha="right")
        ax.set_ylim(.72, 1.04)
        ax.set_title(DATASETS[dataset]["name"], weight="bold")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Published Figure 12 and local held-out accuracy", fontsize=13, weight="bold")
    return _save(fig, out, 12, "Accuracy comparison", "p. 11", "Published bars/markers are transcribed from the paper; local squares use this project's held-out predictions.")


def _pick_example(dataset: str, target_id: int, X: np.ndarray, y: np.ndarray, test_idx: np.ndarray, model):
    probabilities = model.predict(X[test_idx], batch_size=1024, verbose=0)
    predictions = probabilities.argmax(axis=1)
    candidates = np.flatnonzero((y[test_idx] == target_id) & (predictions == target_id))
    if len(candidates) == 0:
        candidates = np.flatnonzero(predictions == target_id)
    if len(candidates) == 0:
        raise ValueError(f"No local example predicted as {DATASETS[dataset]['classes'][target_id]} for {dataset}.")
    pos = int(candidates[0])
    return test_idx[pos], probabilities[pos]


def _lime_examples(dataset: str, out: Path, figure_number: int, target_classes: list[int]):
    base, X, y, split, manifest = _load_dataset(dataset)
    model = keras.models.load_model(base / "dnn" / "model.keras")
    train_idx, test_idx = split["train"], split["test"]
    names = manifest["feature_names"]
    class_names = manifest["class_names"]
    explainer = LimeTabularExplainer(X[train_idx], feature_names=names, class_names=class_names,
                                     mode="classification", discretize_continuous=False, random_state=42)
    predict = lambda z: model.predict(np.asarray(z, dtype=np.float32), batch_size=4096, verbose=0)
    examples = []
    for target in target_classes:
        row_idx, probs = _pick_example(dataset, target, X, y, test_idx, model)
        explanation = explainer.explain_instance(X[row_idx].astype(np.float64), predict,
                                                 num_features=10, num_samples=1000, labels=[target])
        weights = explanation.as_map()[target]
        weights = sorted(weights, key=lambda pair: abs(pair[1]), reverse=True)[:10]
        examples.append((row_idx, target, probs, [(names[i], float(w)) for i, w in weights]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [1, 1.35]})
    for row, (idx, target, probs, weights) in enumerate(examples):
        axp = axes[row, 0]
        axw = axes[row, 1]
        class_colors = ["#E76F51" if i == target else "#A9C5D3" for i in range(len(class_names))]
        axp.barh(class_names, probs, color=class_colors)
        axp.invert_yaxis()
        axp.set_xlim(0, 1.02)
        axp.set_xlabel("Predicted probability")
        axp.set_title(f"Local example {row + 1}: actual {class_names[y[idx]]}; predicted {class_names[target]}", fontsize=9, weight="bold")
        for i, p in enumerate(probs):
            axp.text(min(p + .015, .95), i, f"{p:.2f}", va="center", fontsize=7)
        feats = [item[0] for item in weights][::-1]
        vals = [item[1] for item in weights][::-1]
        colors = ["#E76F51" if v > 0 else "#247BA0" for v in vals]
        axw.barh(feats, vals, color=colors)
        axw.axvline(0, color="#25323B", linewidth=.8)
        axw.set_title("LIME local feature weights", fontsize=9, weight="bold")
        axw.tick_params(axis="y", labelsize=7)
        axw.set_xlabel("Weight toward predicted class  (red + / blue -)")
        axw.grid(axis="x", alpha=.2)
    fig.suptitle(f"{DATASETS[dataset]['name']} local LIME examples from the saved DNN", fontsize=13, weight="bold")
    return _save(fig, out, figure_number, f"{DATASETS[dataset]['name']} LIME examples", "pp. 11-12", "Two local, correctly classified test examples; 1,000 LIME perturbations per example.")


def _shap_for_class(dataset: str, X: np.ndarray, split, manifest, model):
    train_idx, test_idx = split["train"], split["test"]
    rng = np.random.default_rng(42)
    # Match the article's stated 50 test records while making the local selection reproducible.
    positions = np.sort(rng.choice(len(test_idx), size=min(50, len(test_idx)), replace=False))
    rows = X[test_idx[positions]].astype(np.float32)
    bg_idx = train_idx[np.linspace(0, len(train_idx) - 1, min(100, len(train_idx)), dtype=int)]
    explainer = shap.DeepExplainer(model, X[bg_idx].astype(np.float32))
    values = explainer.shap_values(rows, check_additivity=False)
    predictions = model.predict(rows, batch_size=256, verbose=0)
    arr = _normalize_shap(values, len(rows), X.shape[1], len(manifest["class_names"]))
    return rows, arr, predictions


def _shap_summary_plot(dataset: str, out: Path, number: int, row_values: np.ndarray, shap_values: np.ndarray,
                       manifest, class_id: int):
    phi = shap_values[:, :, class_id]
    X = row_values
    importance = np.mean(np.abs(phi), axis=0)
    order = np.argsort(importance)[-20:]
    names = np.asarray(manifest["feature_names"])
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    rng = np.random.default_rng(42)
    cmap = plt.get_cmap("coolwarm")
    for yi, feature_id in enumerate(order):
        vals = phi[:, feature_id]
        feat = X[:, feature_id]
        low, high = np.min(feat), np.max(feat)
        scaled = np.zeros_like(feat) if high <= low else (feat - low) / (high - low)
        yj = yi + rng.normal(0, .13, size=len(vals))
        ax.scatter(vals, yj, c=scaled, cmap=cmap, s=16, alpha=.8, linewidths=0, vmin=0, vmax=1)
    ax.axvline(0, color="#64727A", linewidth=.8)
    ax.set_yticks(range(len(order)), names[order], fontsize=8)
    ax.set_xlabel("SHAP value for selected class (impact on model output)")
    ax.set_title(f"{DATASETS[dataset]['name']} SHAP summary: {manifest['class_names'][class_id]} class, 50 local test rows",
                 fontsize=12, weight="bold")
    ax.grid(axis="x", alpha=.18)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), ax=ax, fraction=.03, pad=.02,
                 label="Feature value (low to high)")
    return _save(fig, out, number, f"{DATASETS[dataset]['name']} SHAP summary", "p. 13", "DeepExplainer values for the selected class on 50 seeded local test rows.")


def _shap_global_class_bars(dataset: str, out: Path, number: int, values: np.ndarray, manifest):
    importances = np.mean(np.abs(values), axis=0)  # features x classes
    order = np.argsort(importances.mean(axis=1))[-20:]
    names = np.asarray(manifest["feature_names"])
    classes = manifest["class_names"]
    fig, ax = plt.subplots(figsize=(13, 8))
    left = np.zeros(len(order))
    for class_id, class_name in enumerate(classes):
        v = importances[order, class_id]
        ax.barh(np.arange(len(order)), v, left=left, color=CLASS_COLORS[class_id], label=class_name, height=.78)
        left += v
    ax.set_yticks(np.arange(len(order)), names[order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute SHAP value summed across classes")
    ax.set_title(f"{DATASETS[dataset]['name']} global feature impact by class (50 test rows)", fontsize=12, weight="bold")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=.18)
    return _save(fig, out, number, f"{DATASETS[dataset]['name']} global SHAP feature impact", "p. 13", "Per-class stacked mean absolute DeepExplainer values over the same 50 test rows.")


def _shap_local_plot(dataset: str, out: Path, number: int, X: np.ndarray, y: np.ndarray, split,
                     manifest, model, shap_rows: np.ndarray, shap_values: np.ndarray, probs: np.ndarray,
                     class_id: int):
    test_idx = split["test"]
    example_idx, example_probs = _pick_example(dataset, class_id, X, y, test_idx, model)
    distances = np.sum((shap_rows - X[example_idx]) ** 2, axis=1)
    pos = int(np.argmin(distances))
    # If the sample is not part of the 50-row summary, explain it directly.
    if distances[pos] > 1e-12:
        bg_idx = split["train"][np.linspace(0, len(split["train"]) - 1, min(100, len(split["train"])), dtype=int)]
        explainer = shap.DeepExplainer(model, X[bg_idx].astype(np.float32))
        one = explainer.shap_values(X[example_idx:example_idx + 1].astype(np.float32), check_additivity=False)
        one = _normalize_shap(one, 1, X.shape[1], len(manifest["class_names"]))
        phi = one[0, :, class_id]
    else:
        phi = shap_values[pos, :, class_id]
    names = np.asarray(manifest["feature_names"])
    order = np.argsort(np.abs(phi))[-12:]
    values = phi[order]
    labels = names[order]
    colors = ["#E76F51" if value > 0 else "#247BA0" for value in values]
    fig, ax = plt.subplots(figsize=(11, 6.3))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#25323B", linewidth=.8)
    ax.set_xlabel("SHAP value contribution to selected class output")
    ax.set_title(f"Local SHAP: {DATASETS[dataset]['classes'][class_id]} example | probability {example_probs[class_id]:.3f}",
                 fontsize=12, weight="bold")
    ax.grid(axis="x", alpha=.18)
    return _save(fig, out, number, f"{DATASETS[dataset]['name']} local SHAP example", "p. 13", "DeepExplainer attribution for one local test record; signs indicate direction on the selected model output.")


def generate(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    records = []
    records.append(figure_01(out))
    records.append(figure_02_03("nsl", out))
    records.append(figure_02_03("unsw", out))
    records.append(figure_04(out))
    records.append(figure_05(out))
    records.append(figure_06(out))
    records.extend(_metrics_and_confusion("nsl", out, 7))
    records.extend(_metrics_and_confusion("unsw", out, 9))
    records.append(figure_11(out))
    records.append(figure_12(out))
    records.append(_lime_examples("nsl", out, 13, [1, 0]))
    records.append(_lime_examples("unsw", out, 14, [4, 1]))

    for dataset, local_number, summary_number, global_number, target_id in (
        ("nsl", 15, 17, 19, 0), ("unsw", 16, 18, 20, 4)
    ):
        base, X, y, split, manifest = _load_dataset(dataset)
        model = keras.models.load_model(base / "dnn" / "model.keras")
        rows, values, predictions = _shap_for_class(dataset, X, split, manifest, model)
        records.append(_shap_local_plot(dataset, out, local_number, X, y, split, manifest, model, rows, values, predictions, target_id))
        records.append(_shap_summary_plot(dataset, out, summary_number, rows, values, manifest, target_id))
        records.append(_shap_global_class_bars(dataset, out, global_number, values, manifest))

    with (out / "figure_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["figure", "file", "title", "paper_page", "local_method"])
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda record: record["figure"]))
    print(f"Generated {len(records)} figure counterparts under {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
