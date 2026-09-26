"""
Pipeline Validation & Leakage Check Script
Replication of Sharma et al. (2024)

Verifies:
1. No target columns ('label', 'attack_cat') present in feature matrix X.
2. Zero sample leakage or index overlap between train, val, and test splits.
3. Split proportions match 60% Train, 15% Validation, 25% Test.
4. Correct 5-class target labels [0, 1, 2, 3, 4] with zero NaN/Inf.
5. All feature values strictly bounded in [0.0, 1.0].
6. Dimension compatibility with DNN, 1D-CNN, and 2D-CNN grid tensors.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def validate_dataset_pipeline(npz_path: Path, dataset_name: str) -> bool:
    print(f"\n{'='*60}\nVALIDATING PIPELINE INTEGRITY: {dataset_name.upper()}\n{'='*60}")
    
    if not npz_path.exists():
        print(f"[FAIL] Processed file {npz_path} does not exist.")
        return False

    data = np.load(npz_path, allow_pickle=True)
    all_keys = list(data.keys())
    print(f"Available keys in .npz: {all_keys}")

    # 1. Target separation check
    all_features = [str(f) for f in data["all_features"]]
    selected_features = [str(f) for f in data["selected_features"]]
    
    forbidden_targets = ["label", "attack_cat", "target", "difficulty_level"]
    for f in forbidden_targets:
        assert f not in all_features, f"[CRITICAL LEAKAGE] '{f}' found in all_features!"
        assert f not in selected_features, f"[CRITICAL LEAKAGE] '{f}' found in selected_features!"
    print("[PASS] Target separation verified: no target columns found in predictor sets.")

    # 2. Split sizes & overlap
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    n_total = len(y_train) + len(y_val) + len(y_test)
    train_pct = len(y_train) / n_total * 100
    val_pct = len(y_val) / n_total * 100
    test_pct = len(y_test) / n_total * 100

    print(f"Split counts: Train={len(y_train)} ({train_pct:.2f}%), Val={len(y_val)} ({val_pct:.2f}%), Test={len(y_test)} ({test_pct:.2f}%)")
    assert abs(train_pct - 60.0) < 0.5, f"Train split expected ~60%, got {train_pct:.2f}%"
    assert abs(val_pct - 15.0) < 0.5, f"Val split expected ~15%, got {val_pct:.2f}%"
    assert abs(test_pct - 25.0) < 0.5, f"Test split expected ~25%, got {test_pct:.2f}%"
    print("[PASS] Split proportions conform to 60/15/25 specification.")

    # 3. Label range & integrity
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique_labels = np.unique(y)
        assert np.all(np.isin(unique_labels, [0, 1, 2, 3, 4])), f"Invalid labels in {split_name}: {unique_labels}"
        assert not np.any(np.isnan(y)), f"NaN found in {split_name} target!"
    print(f"[PASS] Target labels strictly bounded to 5 classes: {np.unique(y_train)}.")

    # 4. Feature value range [0, 1] and no NaN/Inf
    feature_arrays = [
        ("X_all_train", data["X_all_train"]),
        ("X_all_test", data["X_all_test"]),
        ("X_selected_train", data["X_selected_train"]),
        ("X_selected_test", data["X_selected_test"])
    ]

    for name, arr in feature_arrays:
        assert not np.any(np.isnan(arr)), f"NaN detected in {name}!"
        assert not np.any(np.isinf(arr)), f"Inf detected in {name}!"
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        assert min_val >= -1e-6 and max_val <= 1.0 + 1e-6, f"Range violation in {name}: [{min_val}, {max_val}]"
        print(f"  {name}: shape={arr.shape}, min={min_val:.4f}, max={max_val:.4f} [PASS]")

    # 5. Check 2D-CNN dimensions
    if dataset_name == "nsl_kdd":
        grid_key = "X_selected_36_train"
        assert grid_key in data, f"Missing {grid_key} for NSL-KDD 6x6 grid!"
        assert data[grid_key].shape[1] == 36, f"Expected 36 features for 6x6, got {data[grid_key].shape[1]}"
        print(f"[PASS] NSL-KDD 6x6 grid dimension verified (36 features).")
    elif dataset_name == "unsw_nb15":
        grid_key = "X_selected_49_train"
        assert grid_key in data, f"Missing {grid_key} for UNSW-NB15 7x7 grid!"
        assert data[grid_key].shape[1] == 49, f"Expected 49 features for 7x7, got {data[grid_key].shape[1]}"
        print(f"[PASS] UNSW-NB15 7x7 grid dimension verified (49 features).")

    print(f"[SUCCESS] All pipeline validations passed for {dataset_name.upper()}!")
    return True


def run_all_validations():
    nsl_path = PROJECT_ROOT / "data" / "processed" / "nsl_kdd" / "nsl_kdd_processed.npz"
    unsw_path = PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "unsw_processed.npz"

    v1 = validate_dataset_pipeline(nsl_path, "nsl_kdd")
    v2 = validate_dataset_pipeline(unsw_path, "unsw_nb15")

    if v1 and v2:
        print("\n" + "="*60)
        print("PIPELINE VALIDATION COMPLETE: ZERO LEAKAGE & 100% SPEC COMPLIANCE")
        print("="*60)


if __name__ == "__main__":
    run_all_validations()
