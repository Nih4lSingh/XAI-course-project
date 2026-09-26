"""
Single Experiment Runner
Replication of Sharma et al. (2024)

Runs any configured experiment ID from the 12-model matrix:
  [DATASET]_[FEATURE_MODE]_[MODEL]
  Example: NSL_SELECTED_DNN, UNSW_ALL_2DCNN, etc.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from models.dnn import build_dnn_model
from models.cnn1d import build_cnn1d_model
from models.cnn2d import build_cnn2d_model
from training.train_utils import train_and_evaluate_model
from preprocessing.encoders import NSL_KDD_CLASS_MAPPING, UNSW_NB15_CLASS_MAPPING


def set_deterministic_seeds(seed: int = 42):
    """Sets random seeds across Python, NumPy, and TensorFlow."""
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_experiment(
    experiment_id: str,
    epochs: int = 20,
    batch_size: int = 64,
    dropout_rate: float = 0.0,
    verbose: int = 1
) -> Dict:
    set_deterministic_seeds(42)

    parts = experiment_id.upper().split("_")
    dataset_name = "nsl_kdd" if "NSL" in parts[0] else "unsw_nb15"
    feature_mode = "selected" if "SELECTED" in parts[1] else "all"
    
    # Model type parsing
    if "2DCNN" in experiment_id.upper():
        model_type = "2DCNN"
    elif "1DCNN" in experiment_id.upper():
        model_type = "1DCNN"
    elif "DNN" in experiment_id.upper():
        model_type = "DNN"
    else:
        raise ValueError(f"Unknown model type in {experiment_id}")

    # Load preprocessed arrays
    if dataset_name == "nsl_kdd":
        npz_path = PROJECT_ROOT / "data" / "processed" / "nsl_kdd" / "nsl_kdd_processed.npz"
        class_names = [NSL_KDD_CLASS_MAPPING[i] for i in range(5)]
    else:
        npz_path = PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "unsw_processed.npz"
        class_names = [UNSW_NB15_CLASS_MAPPING[i] for i in range(5)]

    data = np.load(npz_path)
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    # Select input representations
    if model_type == "2DCNN":
        from models.feature_to_grid import FeatureGridMapper

        if dataset_name == "nsl_kdd":
            if feature_mode == "selected":
                if "X_selected_36_train" in data and data["X_selected_36_train"].shape[1] == 36:
                    X_train = data["X_selected_36_train"].reshape((-1, 6, 6, 1))
                    X_val = data["X_selected_36_val"].reshape((-1, 6, 6, 1))
                    X_test = data["X_selected_36_test"].reshape((-1, 6, 6, 1))
                else:
                    mapper = FeatureGridMapper([str(i) for i in range(data["X_selected_train"].shape[1])], grid_size=6)
                    X_train = mapper.transform(data["X_selected_train"])
                    X_val = mapper.transform(data["X_selected_val"])
                    X_test = mapper.transform(data["X_selected_test"])
                grid_shape = (6, 6, 1)
            else:
                mapper = FeatureGridMapper([str(i) for i in range(data["X_all_train"].shape[1])], grid_size=7)
                X_train = mapper.transform(data["X_all_train"])
                X_val = mapper.transform(data["X_all_val"])
                X_test = mapper.transform(data["X_all_test"])
                grid_shape = (7, 7, 1)
        else:
            # UNSW-NB15
            if feature_mode == "selected":
                if "X_selected_49_train" in data and data["X_selected_49_train"].shape[1] == 49:
                    X_train = data["X_selected_49_train"].reshape((-1, 7, 7, 1))
                    X_val = data["X_selected_49_val"].reshape((-1, 7, 7, 1))
                    X_test = data["X_selected_49_test"].reshape((-1, 7, 7, 1))
                else:
                    mapper = FeatureGridMapper([str(i) for i in range(data["X_selected_train"].shape[1])], grid_size=7)
                    X_train = mapper.transform(data["X_selected_train"])
                    X_val = mapper.transform(data["X_selected_val"])
                    X_test = mapper.transform(data["X_selected_test"])
            else:
                if "X_all_49_train" in data and data["X_all_49_train"].shape[1] == 49:
                    X_train = data["X_all_49_train"].reshape((-1, 7, 7, 1))
                    X_val = data["X_all_49_val"].reshape((-1, 7, 7, 1))
                    X_test = data["X_all_49_test"].reshape((-1, 7, 7, 1))
                else:
                    mapper = FeatureGridMapper([str(i) for i in range(data["X_all_train"].shape[1])], grid_size=7)
                    X_train = mapper.transform(data["X_all_train"])
                    X_val = mapper.transform(data["X_all_val"])
                    X_test = mapper.transform(data["X_all_test"])
            grid_shape = (7, 7, 1)

        model = build_cnn2d_model(
            input_shape=grid_shape,
            num_classes=5,
            learning_rate=0.001,
            weight_decay=0.0001,
            name=experiment_id
        )

    elif model_type == "1DCNN":
        if feature_mode == "selected":
            X_train = np.expand_dims(data["X_selected_train"], -1)
            X_val = np.expand_dims(data["X_selected_val"], -1)
            X_test = np.expand_dims(data["X_selected_test"], -1)
        else:
            X_train = np.expand_dims(data["X_all_train"], -1)
            X_val = np.expand_dims(data["X_all_val"], -1)
            X_test = np.expand_dims(data["X_all_test"], -1)

        input_dim = X_train.shape[1]
        model = build_cnn1d_model(
            input_dim=input_dim,
            num_classes=5,
            learning_rate=0.001,
            weight_decay=0.0001,
            name=experiment_id
        )

    else:
        # DNN
        if feature_mode == "selected":
            X_train = data["X_selected_train"]
            X_val = data["X_selected_val"]
            X_test = data["X_selected_test"]
        else:
            X_train = data["X_all_train"]
            X_val = data["X_all_val"]
            X_test = data["X_all_test"]

        input_dim = X_train.shape[1]
        model = build_dnn_model(
            input_dim=input_dim,
            num_classes=5,
            dropout_rate=dropout_rate,
            learning_rate=0.001,
            weight_decay=0.0001,
            name=experiment_id
        )

    # Train and evaluate
    metrics = train_and_evaluate_model(
        experiment_id=experiment_id,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default="NSL_SELECTED_DNN")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    run_experiment(
        experiment_id=args.exp_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_rate=args.dropout
    )
