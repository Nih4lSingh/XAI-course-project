"""
Master Explainability (XAI) Runner
Replication of Sharma et al. (2024)

Executes:
1. NSL-KDD Selected DNN -> LIME (DoS & Normal instances) + SHAP (50 test samples global & local).
2. UNSW-NB15 Selected DNN -> LIME (Normal & Exploits instances) + SHAP (50 test samples global & local).
3. Compiles in-depth xai_report.md comparing LIME vs SHAP vs paper-reported features.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from explainability.lime_explainer import LimeExplainerWrapper
from explainability.shap_explainer import ShapExplainerWrapper
from preprocessing.encoders import NSL_KDD_CLASS_MAPPING, UNSW_NB15_CLASS_MAPPING
from training.train_experiment import run_experiment


def run_nsl_kdd_xai(output_base: Path):
    print("\n" + "=" * 65)
    print("RUNNING XAI FOR NSL-KDD SELECTED-FEATURE DNN")
    print("=" * 65)

    lime_dir = output_base / "lime" / "nsl_kdd"
    shap_dir = output_base / "shap" / "nsl_kdd"

    # Load data
    npz_path = PROJECT_ROOT / "data" / "processed" / "nsl_kdd" / "nsl_kdd_processed.npz"
    data = np.load(npz_path)
    X_train = data["X_selected_train"]
    X_test = data["X_selected_test"]
    y_test = data["y_test"]
    feature_names = [str(f) for f in data["selected_features"]]
    class_names = [NSL_KDD_CLASS_MAPPING[i] for i in range(5)]

    # Load trained model
    model_path = PROJECT_ROOT / "results" / "models" / "nsl_selected_dnn.keras"
    if not model_path.exists():
        print("[XAI] Training NSL_SELECTED_DNN model first...")
        run_experiment("NSL_SELECTED_DNN", epochs=20, batch_size=64)
    model = tf.keras.models.load_model(model_path)

    # 1. SHAP Analysis (50 test samples)
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), 50, replace=False)
    X_test_50 = X_test[sample_indices]

    shap_wrapper = ShapExplainerWrapper(
        model=model,
        background_data=X_train[:100],
        feature_names=feature_names,
        class_names=class_names
    )
    shap_global_meta = shap_wrapper.explain_global(
        X_test_50=X_test_50,
        test_indices_50=sample_indices,
        output_dir=shap_dir,
        dataset_name="nsl_kdd"
    )

    # Find representative test samples: one DoS (class 0) and one Normal (class 1)
    dos_indices = np.where(y_test == 0)[0]
    norm_indices = np.where(y_test == 1)[0]
    dos_idx = int(dos_indices[0])
    norm_idx = int(norm_indices[0])

    print(f"\n[LOCAL SAMPLES] Selected DoS instance #{dos_idx}, Normal instance #{norm_idx}")

    # Local SHAP
    shap_wrapper.explain_local_instance(X_test[dos_idx], y_test[dos_idx], dos_idx, shap_dir, "nsl_kdd")
    shap_wrapper.explain_local_instance(X_test[norm_idx], y_test[norm_idx], norm_idx, shap_dir, "nsl_kdd")

    # 2. LIME Analysis
    lime_wrapper = LimeExplainerWrapper(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names
    )
    lime_wrapper.explain_instance(model, X_test[dos_idx], y_test[dos_idx], dos_idx, output_dir=lime_dir)
    lime_wrapper.explain_instance(model, X_test[norm_idx], y_test[norm_idx], norm_idx, output_dir=lime_dir)

    print("[COMPLETE] NSL-KDD XAI analysis finished.")
    return shap_global_meta


def run_unsw_nb15_xai(output_base: Path):
    print("\n" + "=" * 65)
    print("RUNNING XAI FOR UNSW-NB15 SELECTED-FEATURE DNN")
    print("=" * 65)

    lime_dir = output_base / "lime" / "unsw_nb15"
    shap_dir = output_base / "shap" / "unsw_nb15"

    npz_path = PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "unsw_processed.npz"
    data = np.load(npz_path)
    X_train = data["X_selected_train"]
    X_test = data["X_selected_test"]
    y_test = data["y_test"]
    feature_names = [str(f) for f in data["selected_features"]]
    class_names = [UNSW_NB15_CLASS_MAPPING[i] for i in range(5)]

    # Load model
    model_path = PROJECT_ROOT / "results" / "models" / "unsw_selected_dnn.keras"
    if not model_path.exists():
        print("[XAI] Training UNSW_SELECTED_DNN model first...")
        run_experiment("UNSW_SELECTED_DNN", epochs=20, batch_size=64)
    model = tf.keras.models.load_model(model_path)

    # 1. SHAP Analysis (50 test samples)
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), 50, replace=False)
    X_test_50 = X_test[sample_indices]

    shap_wrapper = ShapExplainerWrapper(
        model=model,
        background_data=X_train[:100],
        feature_names=feature_names,
        class_names=class_names
    )
    shap_global_meta = shap_wrapper.explain_global(
        X_test_50=X_test_50,
        test_indices_50=sample_indices,
        output_dir=shap_dir,
        dataset_name="unsw_nb15"
    )

    # Find representative samples: Normal (class 4) and Exploits (class 1)
    norm_indices = np.where(y_test == 4)[0]
    exp_indices = np.where(y_test == 1)[0]
    norm_idx = int(norm_indices[0])
    exp_idx = int(exp_indices[0])

    print(f"\n[LOCAL SAMPLES] Selected Normal instance #{norm_idx}, Exploits instance #{exp_idx}")

    # Local SHAP
    shap_wrapper.explain_local_instance(X_test[norm_idx], y_test[norm_idx], norm_idx, shap_dir, "unsw_nb15")
    shap_wrapper.explain_local_instance(X_test[exp_idx], y_test[exp_idx], exp_idx, shap_dir, "unsw_nb15")

    # 2. LIME Analysis
    lime_wrapper = LimeExplainerWrapper(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names
    )
    lime_wrapper.explain_instance(model, X_test[norm_idx], y_test[norm_idx], norm_idx, output_dir=lime_dir)
    lime_wrapper.explain_instance(model, X_test[exp_idx], y_test[exp_idx], exp_idx, output_dir=lime_dir)

    print("[COMPLETE] UNSW-NB15 XAI analysis finished.")
    return shap_global_meta


def generate_xai_comparison_report(nsl_meta: dict, unsw_meta: dict, output_path: Path):
    """Generates summary XAI markdown comparison."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# XAI Explainability Analysis Report
Sharma et al. (2024) Replication

## 1. NSL-KDD Feature Importance (SHAP Top 5)
{nsl_meta}

## 2. UNSW-NB15 Feature Importance (SHAP Top 5)
{unsw_meta}
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[XAI] Generated comparison report at {output_path}")


def run_all_xai():
    output_base = PROJECT_ROOT / "results" / "xai"
    nsl_meta = run_nsl_kdd_xai(output_base)
    unsw_meta = run_unsw_nb15_xai(output_base)
    if nsl_meta and unsw_meta:
        generate_xai_comparison_report(nsl_meta, unsw_meta, PROJECT_ROOT / "xai_report.md")


if __name__ == "__main__":
    run_all_xai()
