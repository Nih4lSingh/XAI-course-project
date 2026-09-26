"""Reproduce the paper's LIME and SHAP analysis for the replicated 2D-CNNs.

The paper applied LIME and SHAP to its DNN. This script adapts the same
instance-level explanation protocol to the replicated 2D-CNN checkpoints.
It deliberately implements the model-agnostic algorithms with NumPy instead
of installing the external ``lime`` or ``shap`` packages.

Outputs are written under ``outputs/xai_2dcnn/<dataset>``:

* LIME plots and a CSV for the paper's example classes;
* local Kernel-SHAP plots and a CSV;
* a global mean-absolute SHAP plot and CSV;
* a JSON audit record containing the selected run, instance indices, fidelity,
  model verification, sampling settings, and top features.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import nsl_kdd_gpu_sweep as nsl_sweep
import unsw_nb15_gpu_sweep as unsw_sweep


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "xai_2dcnn"


Predictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DatasetRequest:
    slug: str
    display_name: str
    output_source: Path
    checkpoint: Path
    comparison: Path
    result_csv: Path
    class_names: tuple[str, ...]
    lime_classes: tuple[str, ...]
    shap_class: str


@dataclass
class LoadedExperiment:
    request: DatasetRequest
    model: Any
    device: Any
    feature_names: list[str]
    real_feature_indices: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    config_id: str
    seed: int
    stored_accuracy: float
    comparison_summary: dict[str, Any]


@dataclass
class LimeResult:
    intercept: float
    coefficients: np.ndarray
    local_prediction: float
    model_prediction: float
    weighted_r2: float
    weighted_rmse: float
    weighted_mae: float
    samples: int


@dataclass
class ShapResult:
    base_value: float
    values: np.ndarray
    model_prediction: float
    reconstructed_prediction: float
    efficiency_error: float
    samples: int


REQUESTS = {
    "nsl": DatasetRequest(
        slug="nsl_kdd",
        display_name="NSL-KDD",
        output_source=ROOT / "outputs" / "nsl_kdd_all_768_concurrent",
        checkpoint=ROOT / "outputs" / "nsl_kdd_all_768_concurrent" / "checkpoint_latest.pt",
        comparison=ROOT / "outputs" / "nsl_kdd_all_768_concurrent" / "paper_comparison" / "comparison_summary.json",
        result_csv=ROOT / "outputs" / "nsl_kdd_all_768_concurrent" / "all_seed_results.csv",
        class_names=("DoS", "Normal", "Probe", "R2L", "U2R"),
        lime_classes=("Normal", "DoS"),
        shap_class="DoS",
    ),
    "unsw": DatasetRequest(
        slug="unsw_nb15",
        display_name="UNSW-NB15",
        output_source=ROOT / "outputs" / "unsw_nb15_all_384_concurrent",
        checkpoint=ROOT / "outputs" / "unsw_nb15_all_384_concurrent" / "checkpoint_latest.pt",
        comparison=ROOT / "outputs" / "unsw_nb15_all_384_concurrent" / "paper_comparison" / "comparison_summary.json",
        result_csv=ROOT / "outputs" / "unsw_nb15_all_384_concurrent" / "all_seed_results.csv",
        class_names=("DoS", "Exploits", "Fuzzers", "Generic", "Normal"),
        lime_classes=("Normal", "Exploits"),
        shap_class="Normal",
    ),
}


def _torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "A working PyTorch environment is required. Run this script with the "
            "same existing Python environment used for GPU training."
        ) from exc
    return torch


def choose_device(requested: str):
    torch = _torch()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot access a CUDA device.")
    return device


def checkpoint_config_id(config: dict[str, Any], dataset_slug: str) -> str:
    if "config_id" in config:
        return str(config["config_id"])
    batch = int(config["batch_size"])
    optimizer = str(config["optimizer"])
    if dataset_slug == "nsl_kdd":
        return f"batch{batch:03d}-{config['feature_mode']}-{optimizer}"
    return f"batch{batch:03d}-{optimizer}"


class SingleReplicaCNN:
    """Factory for a single seed/configuration extracted from a grouped checkpoint."""

    @staticmethod
    def create(state: dict[str, Any], config_index: int, seed_index: int, device):
        torch = _torch()
        nn = torch.nn
        functional = torch.nn.functional

        class _SingleReplicaCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                prefix = f"config_models.{config_index}."
                self.conv1_weight = nn.Parameter(
                    state[prefix + "conv1_weight"][seed_index].detach().clone(),
                    requires_grad=False,
                )
                self.conv1_bias = nn.Parameter(
                    state[prefix + "conv1_bias"][seed_index].detach().clone(),
                    requires_grad=False,
                )
                self.conv2_weight = nn.Parameter(
                    state[prefix + "conv2_weight"][seed_index].detach().clone(),
                    requires_grad=False,
                )
                self.conv2_bias = nn.Parameter(
                    state[prefix + "conv2_bias"][seed_index].detach().clone(),
                    requires_grad=False,
                )
                self.conv3_weight = nn.Parameter(
                    state[prefix + "conv3_weight"][seed_index].detach().clone(),
                    requires_grad=False,
                )
                self.conv3_bias = nn.Parameter(
                    state[prefix + "conv3_bias"][seed_index].detach().clone(),
                    requires_grad=False,
                )
                self.dense_weight = nn.Parameter(
                    state[prefix + "dense_weight"][seed_index].detach().clone(),
                    requires_grad=False,
                )
                self.dense_bias = nn.Parameter(
                    state[prefix + "dense_bias"][seed_index].detach().clone(),
                    requires_grad=False,
                )

            def forward(self, values):
                values = functional.conv2d(
                    values, self.conv1_weight, self.conv1_bias, padding=1
                )
                values = functional.max_pool2d(
                    functional.relu(values), 2, stride=2, ceil_mode=True
                )
                values = functional.conv2d(
                    values, self.conv2_weight, self.conv2_bias, padding=1
                )
                values = functional.max_pool2d(
                    functional.relu(values), 2, stride=2, ceil_mode=True
                )
                values = functional.conv2d(
                    values, self.conv3_weight, self.conv3_bias, padding=1
                )
                values = functional.max_pool2d(
                    functional.relu(values), 2, stride=2, ceil_mode=True
                )
                values = values.flatten(start_dim=1)
                return functional.linear(values, self.dense_weight, self.dense_bias)

        return _SingleReplicaCNN().to(device).eval()


def load_experiment(request: DatasetRequest, device) -> LoadedExperiment:
    torch = _torch()
    for path in (request.checkpoint, request.comparison, request.result_csv):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment artifact not found: {path}")

    comparison = json.loads(request.comparison.read_text(encoding="utf-8"))
    best = comparison["best_individual_run"]
    config_id = str(best["config_id"])
    seed = int(best["seed"])

    checkpoint = torch.load(request.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 20:
        raise ValueError(f"Expected an epoch-20 checkpoint, got {checkpoint.get('epoch')}.")
    config_index = next(
        index
        for index, config in enumerate(checkpoint["configs"])
        if checkpoint_config_id(config, request.slug) == config_id
    )
    seed_index = checkpoint["seeds"].index(seed)
    config = checkpoint["configs"][config_index]
    model = SingleReplicaCNN.create(
        checkpoint["model_state_dict"], config_index, seed_index, device
    )

    if request.slug == "nsl_kdd":
        feature_data = nsl_sweep.prepare_features(
            ROOT / "Datasets" / "NSL-KDD" / "KDDTrain+.txt",
            str(config["feature_mode"]),
        )
        train, validation, test = nsl_sweep.build_seed_splits(
            feature_data.labels, [seed]
        )
    else:
        manifest = json.loads(
            (request.output_source / "manifest.json").read_text(encoding="utf-8")
        )
        feature_data = unsw_sweep.prepare_features(
            ROOT / "Datasets" / "UNSW-NB15", int(manifest["sample_seed"])
        )
        train, validation, test = unsw_sweep.build_seed_splits(
            feature_data.labels, [seed]
        )

    results = pd.read_csv(request.result_csv)
    selected = results.loc[
        (results["config_id"] == config_id) & (results["seed"] == seed)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one stored result for {config_id}/{seed}, found {len(selected)}."
        )
    stored_accuracy = float(selected.iloc[0]["test_accuracy"])
    feature_names = list(feature_data.metadata["feature_names"])
    real_feature_indices = np.asarray(
        [
            index
            for index, name in enumerate(feature_names)
            if not str(name).startswith("__zero_padding")
        ],
        dtype=np.int64,
    )
    del checkpoint
    gc.collect()
    return LoadedExperiment(
        request=request,
        model=model,
        device=device,
        feature_names=feature_names,
        real_feature_indices=real_feature_indices,
        features=feature_data.features,
        labels=feature_data.labels,
        train_indices=train[0],
        validation_indices=validation[0],
        test_indices=test[0],
        config_id=config_id,
        seed=seed,
        stored_accuracy=stored_accuracy,
        comparison_summary=comparison,
    )


def make_predictor(experiment: LoadedExperiment, batch_size: int) -> Predictor:
    torch = _torch()
    model = experiment.model
    device = experiment.device
    side = int(round(math.sqrt(experiment.features.shape[2] * experiment.features.shape[3])))

    def predict(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != side * side:
            raise ValueError(f"Expected [rows, {side * side}] flat inputs, got {values.shape}.")
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                tensor = torch.as_tensor(
                    values[start : start + batch_size].reshape(-1, 1, side, side),
                    device=device,
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(tensor)
                    probabilities = torch.softmax(logits.float(), dim=1)
                outputs.append(probabilities.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    return predict


def weighted_ridge(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    matrix = np.column_stack([np.ones(len(design)), design])
    lhs = matrix.T @ (weights[:, None] * matrix)
    regularizer = np.eye(matrix.shape[1], dtype=np.float64) * alpha
    regularizer[0, 0] = 0.0
    rhs = matrix.T @ (weights * target)
    beta = np.linalg.solve(lhs + regularizer, rhs)
    fitted = matrix @ beta
    return float(beta[0]), beta[1:], fitted


def weighted_r2(target: np.ndarray, fitted: np.ndarray, weights: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    fitted = np.asarray(fitted, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = float(np.average(target, weights=weights))
    total = float(np.sum(weights * (target - mean) ** 2))
    residual = float(np.sum(weights * (target - fitted) ** 2))
    return 1.0 - residual / total if total > 0 else 1.0


def weighted_errors(
    target: np.ndarray, fitted: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(
        fitted, dtype=np.float64
    )
    weights = np.asarray(weights, dtype=np.float64)
    rmse = math.sqrt(float(np.average(residual**2, weights=weights)))
    mae = float(np.average(np.abs(residual), weights=weights))
    return rmse, mae


def lime_explain(
    predictor: Predictor,
    instance: np.ndarray,
    background: np.ndarray,
    real_indices: np.ndarray,
    target_class: int,
    *,
    samples: int,
    rng: np.random.Generator,
    alpha: float = 1e-3,
) -> LimeResult:
    if samples < len(real_indices) + 2:
        raise ValueError("LIME requires more perturbations than interpretable features.")
    feature_count = len(real_indices)
    # Bias masks toward the original instance so the surrogate remains local.
    masks = (rng.random((samples, feature_count)) < 0.85).astype(np.int8)
    masks[0] = 1
    masks[1] = 0
    for index in range(min(feature_count, samples - 2)):
        masks[index + 2] = 1
        masks[index + 2, index] = 0

    baseline = background.mean(axis=0, dtype=np.float64).astype(np.float32)
    source_rows = np.repeat(baseline[None, :], samples, axis=0)
    source_rows[:, real_indices] = np.where(
        masks.astype(bool), instance[real_indices], baseline[real_indices]
    )
    probabilities = predictor(source_rows)[:, target_class].astype(np.float64)
    distances = np.sqrt(np.sum(1 - masks, axis=1, dtype=np.float64))
    kernel_width = 0.50 * math.sqrt(feature_count)
    weights = np.exp(-(distances**2) / (kernel_width**2))
    intercept, coefficients, fitted = weighted_ridge(
        masks, probabilities, weights, alpha
    )
    local_prediction = float(intercept + coefficients.sum())
    model_prediction = float(predictor(instance[None, :])[0, target_class])
    rmse, mae = weighted_errors(probabilities, fitted, weights)
    return LimeResult(
        intercept=intercept,
        coefficients=coefficients,
        local_prediction=local_prediction,
        model_prediction=model_prediction,
        weighted_r2=weighted_r2(probabilities, fitted, weights),
        weighted_rmse=rmse,
        weighted_mae=mae,
        samples=samples,
    )


def sample_shap_masks(
    feature_count: int, samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if samples < feature_count + 2:
        raise ValueError("Kernel SHAP requires more coalitions than features.")
    masks = np.zeros((samples, feature_count), dtype=np.int8)
    masks[1] = 1
    sizes = np.arange(1, feature_count, dtype=np.int64)
    size_probabilities = 1.0 / (sizes * (feature_count - sizes))
    size_probabilities /= size_probabilities.sum()
    for row in range(2, samples):
        size = int(rng.choice(sizes, p=size_probabilities))
        chosen = rng.choice(feature_count, size=size, replace=False)
        masks[row, chosen] = 1
    weights = np.ones(samples, dtype=np.float64)
    weights[:2] = max(1_000.0, samples * 10.0)
    return masks, weights


def kernel_shap_explain(
    predictor: Predictor,
    instance: np.ndarray,
    baseline: np.ndarray,
    real_indices: np.ndarray,
    target_class: int,
    *,
    samples: int,
    rng: np.random.Generator,
    alpha: float = 1e-8,
) -> ShapResult:
    masks, weights = sample_shap_masks(len(real_indices), samples, rng)
    perturbations = np.repeat(baseline[None, :], samples, axis=0)
    perturbations[:, real_indices] = np.where(
        masks.astype(bool), instance[real_indices], baseline[real_indices]
    )
    predictions = predictor(perturbations)[:, target_class].astype(np.float64)
    base_value = float(predictor(baseline[None, :])[0, target_class])

    design = masks.astype(np.float64)
    centered = predictions - base_value
    model_prediction = float(predictor(instance[None, :])[0, target_class])
    lhs = design.T @ (weights[:, None] * design)
    lhs += np.eye(design.shape[1], dtype=np.float64) * alpha
    rhs = design.T @ (weights * centered)
    # Kernel SHAP requires efficiency: sum(phi) = f(x) - E[f(x)].
    # Solve the weighted least-squares system with that equality constraint.
    feature_count = design.shape[1]
    kkt = np.zeros((feature_count + 1, feature_count + 1), dtype=np.float64)
    kkt[:feature_count, :feature_count] = lhs
    kkt[:feature_count, feature_count] = 1.0
    kkt[feature_count, :feature_count] = 1.0
    constrained_rhs = np.append(rhs, model_prediction - base_value)
    values = np.linalg.solve(kkt, constrained_rhs)[:feature_count]
    reconstructed = float(base_value + values.sum())
    return ShapResult(
        base_value=base_value,
        values=values,
        model_prediction=model_prediction,
        reconstructed_prediction=reconstructed,
        efficiency_error=float(reconstructed - model_prediction),
        samples=samples,
    )


def short_name(value: str, limit: int = 28) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
        }
    )
    return plt


def plot_lime(
    output_path: Path,
    experiment: LoadedExperiment,
    instance: np.ndarray,
    probabilities: np.ndarray,
    actual_class: int,
    predicted_class: int,
    explained_class: int,
    result: LimeResult,
    top_features: int,
) -> None:
    plt = setup_matplotlib()
    names = np.asarray(experiment.feature_names, dtype=object)[
        experiment.real_feature_indices
    ]
    values = instance[experiment.real_feature_indices]
    order = np.argsort(np.abs(result.coefficients))[-top_features:]
    order = order[np.argsort(np.abs(result.coefficients[order]))]
    shown_names = [short_name(str(names[index])) for index in order]
    shown_weights = result.coefficients[order]
    colors = ["#e76f51" if value >= 0 else "#277da1" for value in shown_weights]

    figure, axes = plt.subplots(
        1, 3, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.0, 1.8, 1.35]}
    )
    y_classes = np.arange(len(experiment.request.class_names))
    axes[0].barh(y_classes, probabilities, color="#355c9a")
    axes[0].set_yticks(y_classes, experiment.request.class_names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_title("Class probabilities")
    for index, value in enumerate(probabilities):
        axes[0].text(min(value + 0.015, 0.94), index, f"{value:.3f}", va="center")

    axes[1].barh(np.arange(len(order)), shown_weights, color=colors)
    axes[1].axvline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(order)), shown_names)
    axes[1].set_xlabel(f"LIME weight for {experiment.request.class_names[explained_class]}")
    axes[1].set_title("Local feature effects")

    axes[2].axis("off")
    cell_text = [[short_name(str(names[index]), 22), f"{values[index]:.5f}"] for index in order[::-1]]
    table = axes[2].table(
        cellText=cell_text,
        colLabels=["Feature", "Scaled value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.70, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    axes[2].set_title("Instance values")

    figure.suptitle(
        f"{experiment.request.display_name} 2D-CNN LIME explanation\n"
        f"actual={experiment.request.class_names[actual_class]}  "
        f"predicted={experiment.request.class_names[predicted_class]}  "
        f"explained={experiment.request.class_names[explained_class]}  "
        f"probability={result.model_prediction:.4f}  weighted R2={result.weighted_r2:.4f}  "
        f"RMSE={result.weighted_rmse:.4f}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_local_shap(
    output_path: Path,
    experiment: LoadedExperiment,
    instance: np.ndarray,
    probabilities: np.ndarray,
    actual_class: int,
    predicted_class: int,
    explained_class: int,
    result: ShapResult,
    top_features: int,
) -> None:
    plt = setup_matplotlib()
    names = np.asarray(experiment.feature_names, dtype=object)[
        experiment.real_feature_indices
    ]
    values = instance[experiment.real_feature_indices]
    order = np.argsort(np.abs(result.values))[-top_features:]
    order = order[np.argsort(np.abs(result.values[order]))]
    labels = [
        f"{short_name(str(names[index]), 23)} = {values[index]:.4f}"
        for index in order
    ]
    shown = result.values[order]
    colors = ["#ef476f" if value >= 0 else "#277da1" for value in shown]

    figure, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [1.0, 2.5]}
    )
    y_classes = np.arange(len(experiment.request.class_names))
    axes[0].barh(y_classes, probabilities, color="#5c4d9d")
    axes[0].set_yticks(y_classes, experiment.request.class_names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_title("Class probabilities")
    for index, value in enumerate(probabilities):
        axes[0].text(min(value + 0.015, 0.94), index, f"{value:.3f}", va="center")

    axes[1].barh(np.arange(len(order)), shown, color=colors)
    axes[1].axvline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(order)), labels)
    axes[1].set_xlabel(f"SHAP contribution to P({experiment.request.class_names[explained_class]})")
    axes[1].set_title(
        f"Local Kernel SHAP  base={result.base_value:.4f}  "
        f"prediction={result.model_prediction:.4f}  "
        f"sum={result.reconstructed_prediction:.4f}"
    )
    figure.suptitle(
        f"{experiment.request.display_name} 2D-CNN SHAP explanation\n"
        f"actual={experiment.request.class_names[actual_class]}  "
        f"predicted={experiment.request.class_names[predicted_class]}  "
        f"explained={experiment.request.class_names[explained_class]}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_global_shap(
    output_path: Path,
    display_name: str,
    feature_names: list[str],
    mean_absolute_values: np.ndarray,
    top_features: int,
    instance_count: int,
) -> None:
    plt = setup_matplotlib()
    order = np.argsort(mean_absolute_values)[-top_features:]
    order = order[np.argsort(mean_absolute_values[order])]
    figure, axis = plt.subplots(figsize=(8.8, 6.0))
    axis.barh(
        np.arange(len(order)), mean_absolute_values[order], color="#2a9d8f"
    )
    axis.set_yticks(
        np.arange(len(order)), [short_name(feature_names[index], 34) for index in order]
    )
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_title(
        f"{display_name} 2D-CNN global SHAP importance\n"
        f"Predicted-class explanations across {instance_count} test instances"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def select_correct_instance(
    labels: np.ndarray,
    predictions: np.ndarray,
    target_class: int,
) -> int:
    matches = np.flatnonzero((labels == target_class) & (predictions == target_class))
    if not len(matches):
        raise ValueError(f"No correctly predicted instance found for class {target_class}.")
    return int(matches[0])


def stratified_correct_positions(
    labels: np.ndarray,
    predictions: np.ndarray,
    count: int,
    class_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected: list[int] = []
    base = count // class_count
    remainder = count % class_count
    for class_id in range(class_count):
        candidates = np.flatnonzero(
            (labels == class_id) & (predictions == class_id)
        )
        take = base + (1 if class_id < remainder else 0)
        if len(candidates) < take:
            candidates = np.flatnonzero(labels == class_id)
        if len(candidates):
            selected.extend(
                rng.choice(candidates, size=min(take, len(candidates)), replace=False).tolist()
            )
    if len(selected) < count:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected))
        selected.extend(
            rng.choice(remaining, size=min(count - len(selected), len(remaining)), replace=False).tolist()
        )
    return np.asarray(selected[:count], dtype=np.int64)


def explain_dataset(
    request: DatasetRequest,
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    output_dir = args.output_dir / request.slug
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"[{request.display_name}] loading closest full-metric 2D-CNN run on {device}")
    experiment = load_experiment(request, device)
    predictor = make_predictor(experiment, args.prediction_batch_size)

    flat = experiment.features.reshape(len(experiment.features), -1)
    x_train = flat[experiment.train_indices]
    x_test = flat[experiment.test_indices]
    y_test = experiment.labels[experiment.test_indices]
    test_probabilities = predictor(x_test)
    test_predictions = test_probabilities.argmax(axis=1)
    observed_accuracy = float(np.mean(test_predictions == y_test))
    accuracy_difference = abs(observed_accuracy - experiment.stored_accuracy)
    # The checkpoint was evaluated inside one large grouped-convolution graph.
    # Extracting one replica changes the CUDA kernel shape, so bfloat16 rounding
    # can move a few boundary cases without changing the learned parameters.
    tolerance = max(5e-4, 10.0 / len(y_test))
    if accuracy_difference > tolerance:
        raise AssertionError(
            f"Extracted model accuracy {observed_accuracy:.9f} does not match stored "
            f"accuracy {experiment.stored_accuracy:.9f}."
        )
    print(
        f"[{request.display_name}] verified accuracy {observed_accuracy:.9f} "
        f"for {experiment.config_id} / seed {experiment.seed}"
    )

    rng = np.random.default_rng(args.random_seed + seed_offset)
    background_limit = min(args.background_rows, len(x_train))
    background_positions = rng.choice(
        len(x_train), size=background_limit, replace=False
    )
    background = x_train[background_positions]
    baseline = background.mean(axis=0, dtype=np.float64).astype(np.float32)

    lime_rows: list[dict[str, Any]] = []
    lime_instances: list[dict[str, Any]] = []
    local_instance_positions: dict[str, int] = {}
    for target_name in request.lime_classes:
        target_class = request.class_names.index(target_name)
        test_position = select_correct_instance(y_test, test_predictions, target_class)
        local_instance_positions[target_name] = test_position
        instance = x_test[test_position]
        probabilities = test_probabilities[test_position]
        result = lime_explain(
            predictor,
            instance,
            background,
            experiment.real_feature_indices,
            target_class,
            samples=args.lime_samples,
            rng=rng,
        )
        names = np.asarray(experiment.feature_names, dtype=object)[
            experiment.real_feature_indices
        ]
        values = instance[experiment.real_feature_indices]
        ranks = np.argsort(-np.abs(result.coefficients))
        for rank, index in enumerate(ranks, 1):
            lime_rows.append(
                {
                    "dataset": request.display_name,
                    "config_id": experiment.config_id,
                    "seed": experiment.seed,
                    "test_position": test_position,
                    "source_row_index": int(experiment.test_indices[test_position]),
                    "actual_class": request.class_names[int(y_test[test_position])],
                    "predicted_class": request.class_names[int(test_predictions[test_position])],
                    "explained_class": target_name,
                    "model_probability": result.model_prediction,
                    "local_surrogate_prediction": result.local_prediction,
                    "weighted_r2": result.weighted_r2,
                    "weighted_rmse": result.weighted_rmse,
                    "weighted_mae": result.weighted_mae,
                    "rank": rank,
                    "feature": str(names[index]),
                    "scaled_value": float(values[index]),
                    "lime_weight": float(result.coefficients[index]),
                }
            )
        plot_path = output_dir / f"lime_{target_name.lower()}.png"
        plot_lime(
            plot_path,
            experiment,
            instance,
            probabilities,
            int(y_test[test_position]),
            int(test_predictions[test_position]),
            target_class,
            result,
            args.top_features,
        )
        lime_instances.append(
            {
                "class": target_name,
                "test_position": test_position,
                "source_row_index": int(experiment.test_indices[test_position]),
                "probability": result.model_prediction,
                "weighted_r2": result.weighted_r2,
                "weighted_rmse": result.weighted_rmse,
                "weighted_mae": result.weighted_mae,
                "plot": plot_path.name,
            }
        )
        print(
            f"[{request.display_name}] LIME {target_name}: "
            f"p={result.model_prediction:.4f}, weighted_R2={result.weighted_r2:.4f}, "
            f"RMSE={result.weighted_rmse:.4f}"
        )
    pd.DataFrame(lime_rows).to_csv(output_dir / "lime_explanations.csv", index=False)

    shap_target_name = request.shap_class
    shap_target_class = request.class_names.index(shap_target_name)
    shap_position = local_instance_positions.get(shap_target_name)
    if shap_position is None:
        shap_position = select_correct_instance(
            y_test, test_predictions, shap_target_class
        )
    shap_instance = x_test[shap_position]
    shap_result = kernel_shap_explain(
        predictor,
        shap_instance,
        baseline,
        experiment.real_feature_indices,
        shap_target_class,
        samples=args.shap_samples,
        rng=rng,
    )
    real_names = np.asarray(experiment.feature_names, dtype=object)[
        experiment.real_feature_indices
    ]
    real_values = shap_instance[experiment.real_feature_indices]
    shap_ranks = np.argsort(-np.abs(shap_result.values))
    shap_rows = [
        {
            "dataset": request.display_name,
            "config_id": experiment.config_id,
            "seed": experiment.seed,
            "test_position": shap_position,
            "source_row_index": int(experiment.test_indices[shap_position]),
            "actual_class": request.class_names[int(y_test[shap_position])],
            "predicted_class": request.class_names[int(test_predictions[shap_position])],
            "explained_class": shap_target_name,
            "model_probability": shap_result.model_prediction,
            "base_value": shap_result.base_value,
            "reconstructed_probability": shap_result.reconstructed_prediction,
            "efficiency_error": shap_result.efficiency_error,
            "rank": rank,
            "feature": str(real_names[index]),
            "scaled_value": float(real_values[index]),
            "shap_value": float(shap_result.values[index]),
        }
        for rank, index in enumerate(shap_ranks, 1)
    ]
    pd.DataFrame(shap_rows).to_csv(output_dir / "shap_local_explanation.csv", index=False)
    local_shap_plot = output_dir / f"shap_local_{shap_target_name.lower()}.png"
    plot_local_shap(
        local_shap_plot,
        experiment,
        shap_instance,
        test_probabilities[shap_position],
        int(y_test[shap_position]),
        int(test_predictions[shap_position]),
        shap_target_class,
        shap_result,
        args.top_features,
    )
    print(
        f"[{request.display_name}] SHAP {shap_target_name}: "
        f"base={shap_result.base_value:.4f}, prediction={shap_result.model_prediction:.4f}, "
        f"efficiency_error={shap_result.efficiency_error:.6g}"
    )

    global_records: list[dict[str, Any]] = []
    global_summary: dict[str, Any] | None = None
    if not args.skip_global:
        positions = stratified_correct_positions(
            y_test,
            test_predictions,
            args.global_instances,
            len(request.class_names),
            rng,
        )
        global_values = []
        for number, position in enumerate(positions, 1):
            target_class = int(test_predictions[position])
            result = kernel_shap_explain(
                predictor,
                x_test[position],
                baseline,
                experiment.real_feature_indices,
                target_class,
                samples=args.global_shap_samples,
                rng=rng,
            )
            global_values.append(result.values)
            for feature_index, feature_name in enumerate(real_names):
                global_records.append(
                    {
                        "dataset": request.display_name,
                        "test_position": int(position),
                        "source_row_index": int(experiment.test_indices[position]),
                        "actual_class": request.class_names[int(y_test[position])],
                        "predicted_class": request.class_names[target_class],
                        "feature": str(feature_name),
                        "shap_value": float(result.values[feature_index]),
                        "absolute_shap_value": float(abs(result.values[feature_index])),
                    }
                )
            print(
                f"[{request.display_name}] global SHAP instance {number}/{len(positions)}",
                end="\r" if number < len(positions) else "\n",
                flush=True,
            )
        values_array = np.asarray(global_values)
        mean_absolute = np.mean(np.abs(values_array), axis=0)
        global_frame = pd.DataFrame(global_records)
        global_frame.to_csv(output_dir / "shap_global_instances.csv", index=False)
        ranking = pd.DataFrame(
            {
                "feature": real_names.astype(str),
                "mean_absolute_shap": mean_absolute,
            }
        ).sort_values("mean_absolute_shap", ascending=False)
        ranking.to_csv(output_dir / "shap_global_importance.csv", index=False)
        plot_global_shap(
            output_dir / "shap_global_importance.png",
            request.display_name,
            real_names.astype(str).tolist(),
            mean_absolute,
            args.top_features,
            len(positions),
        )
        global_summary = {
            "instances": int(len(positions)),
            "coalitions_per_instance": int(args.global_shap_samples),
            "test_positions": positions.tolist(),
            "top_features": ranking.head(args.top_features).to_dict(orient="records"),
        }

    summary = {
        "dataset": request.display_name,
        "paper_scope_note": (
            "The paper applied LIME and SHAP to its DNN. This reproduction applies "
            "the same local explanation class choices to the replicated 2D-CNN."
        ),
        "selected_run_rule": "closest individual 17-metric profile to the paper",
        "config_id": experiment.config_id,
        "seed": experiment.seed,
        "checkpoint": str(request.checkpoint.resolve()),
        "checkpoint_epoch": 20,
        "stored_test_accuracy": experiment.stored_accuracy,
        "verified_test_accuracy": observed_accuracy,
        "accuracy_difference": accuracy_difference,
        "accuracy_verification_tolerance": tolerance,
        "accuracy_verification_note": (
            "The saved weights are extracted from the concurrent checkpoint. A small "
            "tolerance allows mixed-precision CUDA kernel-order differences between "
            "the original grouped graph and single-replica inference."
        ),
        "test_rows": int(len(y_test)),
        "input_features": int(flat.shape[1]),
        "explained_real_features": int(len(experiment.real_feature_indices)),
        "excluded_constant_padding_features": int(
            flat.shape[1] - len(experiment.real_feature_indices)
        ),
        "device": str(experiment.device),
        "lime_method": "binary keep-or-training-mean perturbations biased toward the instance, with an exponential locality kernel and weighted ridge surrogate",
        "lime_samples": int(args.lime_samples),
        "lime_instances": lime_instances,
        "shap_method": "model-agnostic Kernel SHAP coalition regression using a training-mean baseline",
        "shap_samples": int(args.shap_samples),
        "shap_local": {
            "class": shap_target_name,
            "test_position": shap_position,
            "source_row_index": int(experiment.test_indices[shap_position]),
            "base_value": shap_result.base_value,
            "model_probability": shap_result.model_prediction,
            "reconstructed_probability": shap_result.reconstructed_prediction,
            "efficiency_error": shap_result.efficiency_error,
            "plot": local_shap_plot.name,
            "top_features": [
                {
                    "feature": str(real_names[index]),
                    "scaled_value": float(real_values[index]),
                    "shap_value": float(shap_result.values[index]),
                }
                for index in shap_ranks[: args.top_features]
            ],
        },
        "global_shap": global_summary,
        "random_seed": int(args.random_seed + seed_offset),
        "background_rows": int(background_limit),
    }
    (output_dir / "xai_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    del experiment.model
    gc.collect()
    if device.type == "cuda":
        _torch().cuda.empty_cache()
    return summary


def write_readme(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# 2D-CNN LIME and SHAP reproduction",
        "",
        "The source paper used LIME and SHAP on its DNN. These outputs apply the same",
        "paper example classes to the replicated 2D-CNN runs selected by the existing",
        "17-metric paper-comparison scripts.",
        "",
        "No external LIME or SHAP package is required. The implementation uses weighted",
        "local keep-or-mean surrogate regression for LIME and model-agnostic Kernel SHAP coalition",
        "regression for SHAP.",
        "",
        "## Verified runs",
        "",
        "| Dataset | Configuration | Seed | Stored accuracy | Verified accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['dataset']} | {summary['config_id']} | {summary['seed']} | "
            f"{summary['stored_test_accuracy']:.9f} | {summary['verified_test_accuracy']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Files per dataset",
            "",
            "- `lime_<class>.png`: class probabilities, local LIME weights, and instance values.",
            "- `lime_explanations.csv`: all LIME feature weights and fidelity values.",
            "- `shap_local_<class>.png`: local Kernel-SHAP contributions.",
            "- `shap_local_explanation.csv`: all local SHAP values.",
            "- `shap_global_importance.png`: mean absolute SHAP importance across a stratified test sample.",
            "- `shap_global_importance.csv`: ranked global SHAP importance.",
            "- `xai_summary.json`: full audit metadata and sampling settings.",
            "",
            "Constant zero-padding cells are excluded from the interpretable feature set",
            "because they cannot explain a prediction. They remain fixed in every model input.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LIME and SHAP explanations for the replicated 2D-CNNs."
    )
    parser.add_argument("--dataset", choices=("both", "nsl", "unsw"), default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--prediction-batch-size", type=int, default=4096)
    parser.add_argument("--background-rows", type=int, default=2048)
    parser.add_argument("--lime-samples", type=int, default=5000)
    parser.add_argument("--shap-samples", type=int, default=4096)
    parser.add_argument("--global-instances", type=int, default=20)
    parser.add_argument("--global-shap-samples", type=int, default=768)
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260925)
    parser.add_argument("--skip-global", action="store_true")
    args = parser.parse_args()
    for name in (
        "prediction_batch_size",
        "background_rows",
        "lime_samples",
        "shap_samples",
        "global_instances",
        "global_shap_samples",
        "top_features",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = ("nsl", "unsw") if args.dataset == "both" else (args.dataset,)
    summaries = [
        explain_dataset(REQUESTS[key], args, index * 10_000)
        for index, key in enumerate(keys)
    ]
    write_readme(args.output_dir, summaries)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(f"LIME and SHAP outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
