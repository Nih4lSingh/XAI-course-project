"""Command-line orchestration for one model-dataset run."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from time import perf_counter
import numpy as np
import tensorflow as tf
from src.cnn_1d import build_cnn_1d, reshape_for_1d
from src.cnn_2d import build_cnn_2d, reshape_for_2d
from src.data_preprocessing import prepare_dataset
from src.dnn import build_dnn
from src.evaluation import evaluate_model, save_evaluation

MODEL_DIR = {"dnn": "dnn", "cnn_1d": "1d_cnn", "cnn_2d": "2d_cnn"}

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as stream: return json.load(stream)

def set_seed(seed):
    if seed is None: return
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_DIR)
    parser.add_argument("--dataset", required=True, choices=("nsl_kdd", "unsw_nb15"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config); set_seed(config.get("seed"))
    bundle = prepare_dataset(args.dataset, config)
    model_config = config["models"][args.model]
    if args.model == "dnn":
        train_x, val_x, test_x = bundle.X_train, bundle.X_val, bundle.X_test
        model = build_dnn(train_x.shape[1], len(bundle.class_names), model_config)
    elif args.model == "cnn_1d":
        train_x, val_x, test_x = reshape_for_1d(bundle.X_train), reshape_for_1d(bundle.X_val), reshape_for_1d(bundle.X_test)
        model = build_cnn_1d(bundle.X_train.shape[1], len(bundle.class_names), model_config)
    else:
        train_x, side = reshape_for_2d(bundle.X_train)
        val_x, val_side = reshape_for_2d(bundle.X_val)
        test_x, test_side = reshape_for_2d(bundle.X_test)
        if side != val_side or side != test_side: raise RuntimeError("2D reshape produced inconsistent shapes.")
        model = build_cnn_2d(side, len(bundle.class_names), model_config)
    training = config["training"]
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    start = perf_counter()
    history = model.fit(train_x, bundle.y_train, epochs=training["epochs"], batch_size=training["batch_size"], validation_data=(val_x, bundle.y_val), verbose=training["verbose"], callbacks=callbacks)
    train_seconds = perf_counter() - start
    metrics, matrix = evaluate_model(model, test_x, bundle.y_test, bundle.class_names, train_seconds, len(bundle.selected_features))
    result_dir = Path("results") / MODEL_DIR[args.model]
    save_evaluation(result_dir, args.model, bundle.name, metrics, matrix, bundle.class_names, history)
    model_dir = Path("models") / MODEL_DIR[args.model]; model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"{bundle.name}.keras")
    metadata = {"config": config, "dataset": bundle.name, "selected_features": bundle.selected_features, "classes": bundle.class_names, "preprocessing": bundle.metadata, "metrics": metrics, "status": "generated"}
    with open(result_dir / "run_metadata.json", "w", encoding="utf-8") as stream: json.dump(metadata, stream, indent=2)
    print(f"Wrote generated artifacts to {result_dir}")

if __name__ == "__main__": main()
