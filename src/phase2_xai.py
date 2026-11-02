"""Compare SHAP, LIME, and neighborhood-averaged SHAP on the frozen DNN."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

# Resolve the enclosing project when the code lives in outputs/source_code;
# when cloned as a standalone repository, use the repository root instead.
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "outputs" / "source_code").is_dir()),
    SCRIPT_PATH.parents[1],
)
# Keep framework temporary files within the project on managed Windows hosts.
TEMP_DIRECTORY = PROJECT_ROOT / "work" / "tmp"
TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TEMP_DIRECTORY)
os.environ["TMP"] = str(TEMP_DIRECTORY)
tempfile.tempdir = str(TEMP_DIRECTORY)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from tensorflow import keras

from models import SEED, split_indices

PHASE1_DIR = PROJECT_ROOT / "work" / "results" / "phase1"
PHASE2_DIR = PROJECT_ROOT / "work" / "results" / "phase2"


def _as_nfc(values: Any, n_features: int, n_classes: int) -> np.ndarray:
    """Normalize old/new SHAP return formats to (rows, features, classes)."""
    if isinstance(values, list):
        per_class = []
        for value in values:
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            per_class.append(arr)
        result = np.stack(per_class, axis=-1)
    else:
        result = np.asarray(values)
    if result.ndim == 2 and n_classes == 1:
        result = result[..., np.newaxis]
    if result.ndim == 3 and result.shape[1] == n_classes and result.shape[2] == n_features:
        result = np.transpose(result, (0, 2, 1))
    if result.ndim != 3 or result.shape[1:] != (n_features, n_classes):
        raise ValueError(f"Unexpected SHAP array shape {result.shape}; expected (N,{n_features},{n_classes})")
    return result.astype(np.float64)


def _stratified_by_prediction(indices: np.ndarray, predicted: np.ndarray, cap_per_class: int) -> np.ndarray:
    selected = []
    for class_id in np.unique(predicted[indices]):
        candidates = indices[predicted[indices] == class_id]
        selected.extend(candidates[:cap_per_class].tolist())
    return np.asarray(selected, dtype=np.int64)


def _top_indices(values: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(np.abs(values))[-k:]


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = set(a.tolist()), set(b.tolist())
    return len(aa & bb) / len(aa | bb) if aa | bb else 1.0


def _paired_summary(a: np.ndarray, b: np.ndarray, seed: int = SEED) -> dict[str, float]:
    """Paired percentile bootstrap CI for mean(b-a) and a signed-rank test."""
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    delta = delta[np.isfinite(delta)]
    if not len(delta):
        return {"mean_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "wilcoxon_p": float("nan")}
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(delta, size=(2000, len(delta)), replace=True), axis=1)
    try:
        pvalue = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
    except ValueError:
        pvalue = 1.0
    return {
        "mean_delta": float(np.mean(delta)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "wilcoxon_p": pvalue,
    }


def run(dataset: str, output_dir: Path, n_queries: int, n_neighbors: int, lime_samples: int) -> None:
    base = PHASE1_DIR / dataset
    if not (base / "dnn" / "model.keras").exists():
        raise FileNotFoundError(f"Missing Phase 1 DNN artifact: {base / 'dnn' / 'model.keras'}")
    bundle = np.load(base / "processed_data.npz")
    X, y = bundle["X"], bundle["y"]
    manifest = joblib.load(base / "data_manifest.joblib")
    class_names = manifest["class_names"]
    feature_names = manifest["feature_names"]
    splits = np.load(base / "split_indices.npz")
    train_idx, test_idx = splits["train"], splits["test"]
    model = keras.models.load_model(base / "dnn" / "model.keras")

    train_prob = model.predict(X[train_idx], batch_size=1024, verbose=0)
    test_prob_all = model.predict(X[test_idx], batch_size=1024, verbose=0)
    train_pred, test_pred_all = train_prob.argmax(1), test_prob_all.argmax(1)
    # Allocate queries across predicted classes where possible, so every selected
    # query has a same-predicted-class training neighborhood.
    test_positions = np.arange(len(test_idx))
    test_positions = _stratified_by_prediction(test_positions, test_pred_all, max(1, n_queries // len(class_names)))
    if len(test_positions) < n_queries:
        # Preserve equal predicted-class quotas where possible, then fill any
        # shortfall from unused test rows with a deterministic random sample.
        remaining = np.setdiff1d(np.arange(len(test_idx)), test_positions, assume_unique=False)
        fill_count = min(n_queries - len(test_positions), len(remaining))
        rng = np.random.default_rng(SEED)
        fill = rng.choice(remaining, size=fill_count, replace=False)
        test_positions = np.concatenate([test_positions, fill])
    test_positions = test_positions[:n_queries]
    query_idx = test_idx[test_positions]
    Xq, pq = X[query_idx], test_prob_all[test_positions]
    pred_q = pq.argmax(1)

    # A fixed, class-stratified subset of training rows keeps the cost bounded;
    # all neighbors still come from the training partition.
    train_positions = _stratified_by_prediction(np.arange(len(train_idx)), train_pred, 100)
    ref_idx = train_idx[train_positions]
    Xr, pred_r = X[ref_idx], train_pred[train_positions]
    background = X[train_idx[np.linspace(0, len(train_idx) - 1, min(100, len(train_idx)), dtype=int)]]

    explainer = shap.DeepExplainer(model, background.astype(np.float32))
    start = time.perf_counter()
    raw_values = _as_nfc(explainer.shap_values(Xq.astype(np.float32), check_additivity=False), X.shape[1], len(class_names))
    shap_runtime = (time.perf_counter() - start) / max(1, len(Xq))
    start = time.perf_counter()
    ref_values = _as_nfc(explainer.shap_values(Xr.astype(np.float32), check_additivity=False), X.shape[1], len(class_names))
    reference_shap_runtime = (time.perf_counter() - start) / max(1, len(Xr))

    # Split up to 100 explanation references per predicted class into disjoint
    # aggregation and evaluation neighborhoods. This keeps the stability
    # comparison independent of the references used to build SC-SHAP.
    agg_pools: dict[int, np.ndarray] = {}
    eval_pools: dict[int, np.ndarray] = {}
    for c in range(len(class_names)):
        pool = np.flatnonzero(pred_r == c)
        midpoint = len(pool) // 2
        agg_pools[c], eval_pools[c] = pool[:midpoint], pool[midpoint:]
    neighbor_models = {
        c: NearestNeighbors(n_neighbors=min(n_neighbors, len(pool)), metric="euclidean").fit(Xr[pool])
        for c, pool in agg_pools.items() if len(pool) > 0
    }
    evaluation_neighbor_models = {
        c: NearestNeighbors(n_neighbors=min(n_neighbors, len(pool)), metric="euclidean").fit(Xr[pool])
        for c, pool in eval_pools.items() if len(pool) > 0
    }

    lime = LimeTabularExplainer(
        X[train_idx], feature_names=feature_names, class_names=class_names,
        mode="classification", discretize_continuous=False, random_state=SEED,
        sample_around_instance=False,
    )
    model_predict = lambda z: model.predict(np.asarray(z, dtype=np.float32), batch_size=4096, verbose=0)
    lime_values = np.zeros((len(Xq), X.shape[1]), dtype=np.float64)
    lime_seconds = np.zeros(len(Xq), dtype=np.float64)
    for i, row in enumerate(Xq):
        started = time.perf_counter()
        explanation = lime.explain_instance(
            row.astype(np.float64), model_predict, num_features=X.shape[1],
            num_samples=lime_samples, labels=[int(pred_q[i])],
        )
        for feature_id, weight in explanation.as_map()[int(pred_q[i])]:
            lime_values[i, feature_id] = weight
        lime_seconds[i] = time.perf_counter() - started

    raw_local = np.stack([raw_values[i, :, pred_q[i]] for i in range(len(Xq))])
    ref_local = np.stack([ref_values[i, :, pred_r[i]] for i in range(len(Xr))])
    sc_values = np.zeros_like(raw_local)
    reliability_rows = []
    neighbor_positions = []
    evaluation_neighbor_positions = []
    aggregation_started = time.perf_counter()
    for i, class_id in enumerate(pred_q):
        class_key = int(class_id)
        pool = agg_pools.get(class_key, np.asarray([], dtype=int))
        eval_pool = eval_pools.get(class_key, np.asarray([], dtype=int))
        if len(pool) == 0 or len(eval_pool) == 0:
            raise ValueError(f"Insufficient disjoint training references for predicted class {class_names[class_id]}.")
        k = min(n_neighbors, len(pool))
        distances, local_pos = neighbor_models[class_key].kneighbors(Xq[i:i+1])
        global_pos = pool[local_pos[0]]
        neighbor_positions.append(global_pos)
        _, eval_local_pos = evaluation_neighbor_models[class_key].kneighbors(Xq[i:i+1])
        evaluation_neighbor_positions.append(eval_pool[eval_local_pos[0]])
        weights = 1.0 / np.maximum(distances[0], 1e-8)
        weights = weights / weights.sum()
        local_ref_phi = ref_local[global_pos]
        sc_values[i] = np.average(local_ref_phi, axis=0, weights=weights)
        top5_counts = np.zeros(X.shape[1], dtype=int)
        for phi in local_ref_phi:
            top5_counts[_top_indices(phi, 5)] += 1
        # Ignore zero-valued neighbors when assessing direction agreement.
        positive = (local_ref_phi > 1e-9).sum(axis=0)
        negative = (local_ref_phi < -1e-9).sum(axis=0)
        nonzero = positive + negative
        direction = np.divide(np.maximum(positive, negative), nonzero, out=np.ones_like(positive, dtype=float), where=nonzero > 0)
        selected_frequency = float(np.mean(top5_counts[_top_indices(sc_values[i], 5)]) / k)
        reliability_rows.append({
            "query_position": i,
            "predicted_class": class_names[int(class_id)],
            "neighbors": int(k),
            "mean_sc_top5_neighbor_frequency": selected_frequency,
            "mean_direction_consistency": float(np.mean(direction)),
            "mean_neighbor_distance": float(np.mean(distances[0])),
        })
    aggregation_seconds_per_query = (time.perf_counter() - aggregation_started) / max(1, len(Xq))

    methods = {"Original_SHAP": raw_local, "Original_LIME": lime_values, "SC_SHAP": sc_values}
    metrics_rows = []
    for i, class_id in enumerate(pred_q):
        validation_phi = ref_local[evaluation_neighbor_positions[i]]
        top5_neighbor = [_top_indices(phi, 5) for phi in validation_phi]
        row = {"query_position": i, "predicted_class": class_names[int(class_id)]}
        row["evaluation_neighbors"] = int(len(validation_phi))
        for method, values in methods.items():
            phi = values[i]
            top5 = _top_indices(phi, 5)
            row[f"{method}_top5_jaccard"] = float(np.mean([_jaccard(top5, n) for n in top5_neighbor]))
            row[f"{method}_neighbor_spearman"] = float(np.nanmean([
                spearmanr(phi, ref_local[j]).statistic for j in evaluation_neighbor_positions[i]
            ]))
            # Faithfulness proxy: replace top-ranked inputs with the median of
            # same-predicted-class training neighbors; compare to random top-k.
            neighbors = Xr[neighbor_positions[i]]
            local_median = np.median(neighbors, axis=0)
            target_probability = float(pq[i, class_id])
            top_mask = Xq[i].copy()
            top_mask[top5] = local_median[top5]
            top_prob = float(model(top_mask[None, :], training=False).numpy()[0, class_id])
            rng = np.random.default_rng(SEED + i)
            random_idx = rng.choice(X.shape[1], size=len(top5), replace=False)
            random_mask = Xq[i].copy()
            random_mask[random_idx] = local_median[random_idx]
            random_prob = float(model(random_mask[None, :], training=False).numpy()[0, class_id])
            row[f"{method}_top5_confidence_drop"] = target_probability - top_prob
            row[f"{method}_random5_confidence_drop"] = target_probability - random_prob
            row[f"{method}_faithfulness_gap"] = (target_probability - top_prob) - (target_probability - random_prob)
        row["SHAP_vs_LIME_top5_jaccard"] = _jaccard(_top_indices(raw_local[i], 5), _top_indices(lime_values[i], 5))
        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows)
    reliability = pd.DataFrame(reliability_rows)
    # Prediction parity is exact by construction because all explanations use
    # the same frozen DNN; keep the scores in the output for auditability.
    yq = y[query_idx]
    prediction_summary = {
        "accuracy_on_selected_queries": float(accuracy_score(yq, pred_q)),
        "macro_f1_on_selected_queries": float(f1_score(yq, pred_q, average="macro", zero_division=0)),
        "model_file": str(base / "dnn" / "model.keras"),
        "query_count": int(len(Xq)),
        "reference_count": int(len(Xr)),
        "prediction_hash": __import__("hashlib").sha256(pq.tobytes()).hexdigest(),
        "model_weights_hash": __import__("hashlib").sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest(),
    }
    runtime_summary = {
        "original_shap_seconds_per_query": float(shap_runtime),
        "reference_shap_seconds_per_training_reference": float(reference_shap_runtime),
        "original_lime_seconds_per_query_mean": float(lime_seconds.mean()),
        "sc_shap_posthoc_seconds_per_query_mean": float(aggregation_seconds_per_query),
        "note": "SC-SHAP profiles reuse precomputed SHAP values from a disjoint aggregation-reference pool; this runtime excludes the one-time explanation cost for both aggregation and evaluation reference pools.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "per_query_metrics.csv", index=False)
    reliability.to_csv(output_dir / "sc_shap_reliability.csv", index=False)
    feature_summary = []
    for method, values in methods.items():
        importance = np.mean(np.abs(values), axis=0)
        order = np.argsort(importance)[::-1]
        for rank, feature_id in enumerate(order, start=1):
            feature_summary.append({"method": method, "rank": rank, "feature": feature_names[feature_id], "mean_abs_attribution": float(importance[feature_id])})
    pd.DataFrame(feature_summary).to_csv(output_dir / "feature_ranking.csv", index=False)

    comparison = {
        "dataset": dataset,
        "model": "frozen DNN; no retraining during explanation evaluation",
        "prediction_and_model": prediction_summary,
        "runtime": runtime_summary,
        "mean_metrics": metrics.mean(numeric_only=True).to_dict(),
        "paired_tests_sc_shap_vs_original_shap": {
            "top5_jaccard": _paired_summary(metrics["Original_SHAP_top5_jaccard"].to_numpy(), metrics["SC_SHAP_top5_jaccard"].to_numpy()),
            "neighbor_spearman": _paired_summary(metrics["Original_SHAP_neighbor_spearman"].to_numpy(), metrics["SC_SHAP_neighbor_spearman"].to_numpy()),
            "faithfulness_gap": _paired_summary(metrics["Original_SHAP_faithfulness_gap"].to_numpy(), metrics["SC_SHAP_faithfulness_gap"].to_numpy()),
        },
        "interpretation_caveat": "SC-SHAP is a distance-weighted neighborhood attribution profile; because it aggregates explanations for neighboring records, it is not guaranteed to satisfy SHAP local accuracy for the query record. Stability metrics are evaluated against disjoint same-class training references not used to construct the profile. Faithfulness and stability results must be interpreted together.",
        "parameters": {"seed": SEED, "neighbors": n_neighbors, "test_explanations": len(Xq), "lime_samples": lime_samples},
    }
    (output_dir / "summary.json").write_text(json.dumps(comparison, indent=2, allow_nan=True), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = ["Original_SHAP_top5_jaccard", "Original_LIME_top5_jaccard", "SC_SHAP_top5_jaccard"]
    ax.boxplot([metrics[c].dropna() for c in cols], tick_labels=["SHAP", "LIME", "SC-SHAP"], showmeans=True)
    ax.set_ylabel("Mean top-5 overlap with same-class neighbors")
    ax.set_title(f"Explanation neighborhood consistency — {dataset.upper()}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "neighborhood_consistency.png", dpi=180)
    plt.close(fig)

    print(f"Saved Phase 2 XAI results to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["nsl", "unsw"], required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--lime-samples", type=int, default=1000)
    args = parser.parse_args()
    out = args.output or (PHASE2_DIR / args.dataset)
    run(args.dataset, out, args.queries, args.neighbors, args.lime_samples)


if __name__ == "__main__":
    main()
