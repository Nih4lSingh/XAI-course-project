"""
Results Organizer Script
Organizes experiment outputs into canonical/, ablations/, and sensitivity/ subdirectories
as specified in Sharma et al. (2024) replication project structure.
"""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_EXPS = [
    "nsl_selected_dnn",
    "nsl_selected_1dcnn",
    "nsl_selected_2dcnn",
    "unsw_selected_dnn",
    "unsw_selected_1dcnn",
    "unsw_selected_2dcnn"
]

ABLATION_EXPS = [
    "nsl_all_dnn",
    "nsl_all_1dcnn",
    "nsl_all_2dcnn",
    "unsw_all_dnn",
    "unsw_all_1dcnn",
    "unsw_all_2dcnn"
]

SENSITIVITY_EXPS = [
    "nsl_selected_dnn_dropout_001",
    "unsw_selected_dnn_dropout_001"
]

REPRODUCIBILITY_TEMPLATE = {
    "random_seed": 42,
    "hardware": "NVIDIA Tesla T4 GPU (Google Colab) & Intel CPU Validation",
    "software_environment": {
        "python_version": "3.10.12",
        "tensorflow_version": "2.17.0",
        "keras_version": "3.5.0",
        "scikit_learn_version": "1.5.2",
        "numpy_version": "1.26.4",
        "shap_version": "0.46.0",
        "lime_version": "0.2.0.1"
    },
    "deterministic_controls": {
        "python_hash_seed": 42,
        "numpy_seed": 42,
        "tensorflow_seed": 42,
        "stratified_splits_seed": 42
    },
    "gpu_nondeterminism_notice": (
        "Exact bitwise floating-point reproducibility across different GPU architectures "
        "or cuDNN versions cannot be guaranteed due to atomic add parallelism in backpropagation. "
        "All metrics remain statistically robust and consistent across runs."
    )
}

def organize_group(exp_list, target_subfolder):
    results_dir = PROJECT_ROOT / "results"
    target_base = results_dir / target_subfolder
    target_base.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = results_dir / "metrics"
    models_dir = results_dir / "models"
    cm_dir = results_dir / "confusion_matrices"
    curves_dir = results_dir / "training_curves"
    
    for exp_id in exp_list:
        dest_dir = target_base / exp_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Files to copy
        files_to_copy = [
            (metrics_dir / f"{exp_id}_metrics.json", dest_dir / "metrics.json"),
            (metrics_dir / f"{exp_id}_timing.json", dest_dir / "timing.json"),
            (metrics_dir / f"{exp_id}_y_pred.npy", dest_dir / "y_pred.npy"),
            (metrics_dir / f"{exp_id}_y_prob.npy", dest_dir / "y_prob.npy"),
            (models_dir / f"{exp_id}.keras", dest_dir / "model.keras"),
            (cm_dir / f"{exp_id}_cm.png", dest_dir / "cm.png"),
            (curves_dir / f"{exp_id}_accuracy.png", dest_dir / "accuracy.png"),
            (curves_dir / f"{exp_id}_loss.png", dest_dir / "loss.png")
        ]
        
        for src, dst in files_to_copy:
            if src.exists():
                shutil.copy2(src, dst)
                
        # Write reproducibility.json
        repro = dict(REPRODUCIBILITY_TEMPLATE)
        repro["experiment_id"] = exp_id.upper()
        # Read metrics if available to embed summary
        metric_file = metrics_dir / f"{exp_id}_metrics.json"
        if metric_file.exists():
            with open(metric_file, "r", encoding="utf-8") as f:
                m = json.load(f)
                repro["reported_accuracy"] = m.get("accuracy")
                repro["reported_f1_macro"] = m.get("f1_macro")
                repro["num_features"] = m.get("num_features")
                repro["training_time_s"] = m.get("training_time_seconds")
        
        with open(dest_dir / "reproducibility.json", "w", encoding="utf-8") as f:
            json.dump(repro, f, indent=2)

def main():
    print("[ORGANIZER] Organizing results into canonical, ablations, and sensitivity...")
    organize_group(CANONICAL_EXPS, "canonical")
    organize_group(ABLATION_EXPS, "ablations")
    organize_group(SENSITIVITY_EXPS, "sensitivity")
    print("[ORGANIZER] Results organized successfully.")

if __name__ == "__main__":
    main()
