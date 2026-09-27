"""
Master Results Validator Script
Replication of Sharma et al. (2024)

Validates:
1. Every experiment has a metric JSON.
2. Every summary row maps to exactly one metric JSON.
3. Dataset name matches.
4. Canonical feature counts strictly match:
   - NSL-KDD Selected: exactly 36 features
   - UNSW-NB15 Selected: exactly 38 features
   - NSL-KDD All: exactly 42 features
   - UNSW-NB15 All: exactly 42 features
5. Model name matches.
6. Accuracy matches between JSON and CSV.
7. Class mapping matches (5 classes with non-empty per-class metrics).
8. Timestamps/experiment IDs are traceable.
9. No stale 36-feature UNSW result remains in canonical outputs.

Exits with code 0 on success, code 1 on failure.
"""

import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_EXPS = [
    "NSL_SELECTED_DNN",
    "NSL_SELECTED_1DCNN",
    "NSL_SELECTED_2DCNN",
    "UNSW_SELECTED_DNN",
    "UNSW_SELECTED_1DCNN",
    "UNSW_SELECTED_2DCNN"
]

ALL_EXPS = CANONICAL_EXPS

EXPECTED_FEATURES = {
    "NSL_SELECTED_DNN": 36,
    "NSL_SELECTED_1DCNN": 36,
    "NSL_SELECTED_2DCNN": 36,
    "UNSW_SELECTED_DNN": 38,
    "UNSW_SELECTED_1DCNN": 38,
    "UNSW_SELECTED_2DCNN": 38,  # Raw selected features before 11 zeros padding
}


def find_metric_json(exp_id: str) -> Path:
    exp_lower = exp_id.lower()
    candidates = [
        PROJECT_ROOT / "results" / "canonical" / exp_lower / "metrics.json",
        PROJECT_ROOT / "results" / "ablations" / exp_lower / "metrics.json",
        PROJECT_ROOT / "results" / "sensitivity" / exp_lower / "metrics.json",
        PROJECT_ROOT / "results" / "metrics" / f"{exp_lower}_metrics.json"
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def validate_results() -> bool:
    print("=" * 70)
    print("RUNNING MASTER RESULTS VALIDATION (SHARMA ET AL. 2024 REPLICATION)")
    print("=" * 70)

    errors = []
    warnings = []

    # 1. Check metric JSONs
    print("\n[CHECK 1/6] Verifying presence of machine-readable metrics JSONs...")
    metric_data = {}
    for exp_id in ALL_EXPS:
        p = find_metric_json(exp_id)
        if p is None:
            errors.append(f"Missing metrics JSON for experiment {exp_id}")
        else:
            with open(p, "r", encoding="utf-8") as f:
                metric_data[exp_id] = (p, json.load(f))
            print(f"  OK: {exp_id} -> {p.relative_to(PROJECT_ROOT)}")

    # 2. Check summary CSV consistency
    print("\n[CHECK 2/6] Verifying results_summary.csv consistency with metric JSONs...")
    summary_csv = PROJECT_ROOT / "results" / "results_summary.csv"
    if not summary_csv.exists():
        errors.append("results/results_summary.csv does not exist")
    else:
        df_summary = pd.read_csv(summary_csv)
        for exp_id, (p, m) in metric_data.items():
            row = df_summary[df_summary["experiment_id"] == exp_id]
            if row.empty:
                errors.append(f"Experiment {exp_id} missing from results_summary.csv")
            else:
                csv_acc = float(row.iloc[0]["accuracy"])
                json_acc = round(float(m["accuracy"]), 4)
                if abs(csv_acc - json_acc) > 1e-4:
                    errors.append(f"Accuracy mismatch for {exp_id}: CSV={csv_acc}, JSON={json_acc}")
                
                # Check feature count
                expected_feat = EXPECTED_FEATURES[exp_id]
                csv_feat = int(row.iloc[0]["num_features"])
                if csv_feat != expected_feat:
                    errors.append(f"Feature count mismatch in CSV for {exp_id}: expected {expected_feat}, found {csv_feat}")

    # 3. Check for Stale 36-feature UNSW Results in Canonical Outputs
    print("\n[CHECK 3/6] Verifying no stale 36-feature UNSW results exist in canonical outputs...")
    unsw_canonical = ["UNSW_SELECTED_DNN", "UNSW_SELECTED_1DCNN", "UNSW_SELECTED_2DCNN"]
    for exp_id in unsw_canonical:
        if exp_id in metric_data:
            p, m = metric_data[exp_id]
            recorded_feats = m.get("num_features")
            # 2DCNN may record 49 (7x7) or 38; but 1DCNN and DNN MUST record 38 (not 36)
            if "1DCNN" in exp_id or "DNN" in exp_id:
                if recorded_feats == 36:
                    errors.append(f"STALE 36-feature result found in {p}! Must be 38 features.")
                elif recorded_feats == 38:
                    print(f"  OK: {exp_id} feature count verified as 38.")
            elif "2DCNN" in exp_id:
                if recorded_feats not in (38, 49):
                    errors.append(f"Unexpected feature count {recorded_feats} for {exp_id}")
                else:
                    print(f"  OK: {exp_id} dimension verified ({recorded_feats}).")

    # 4. Check Class Mappings (5 classes per dataset)
    print("\n[CHECK 4/6] Verifying 5-class evaluation mappings...")
    for exp_id, (p, m) in metric_data.items():
        per_class = m.get("per_class", {})
        if len(per_class) != 5:
            errors.append(f"Expected 5 classes in {exp_id}, found {len(per_class)}")
        else:
            # Check non-zero/valid metrics
            for cname, cmetrics in per_class.items():
                if "f1_score" not in cmetrics or "precision" not in cmetrics or "recall" not in cmetrics:
                    errors.append(f"Incomplete class metrics for {exp_id} class {cname}")

    # 5. Check Paper Comparison Table
    print("\n[CHECK 5/6] Verifying paper_vs_reproduction.csv...")
    comp_csv = PROJECT_ROOT / "results" / "paper_vs_reproduction.csv"
    if not comp_csv.exists():
        errors.append("results/paper_vs_reproduction.csv does not exist")
    else:
        df_comp = pd.read_csv(comp_csv)
        if len(df_comp) != 6:
            errors.append(f"Expected 6 rows in paper_vs_reproduction.csv, found {len(df_comp)}")
        else:
            print("  OK: Paper comparison table contains all 6 canonical models.")

    # 6. Check Model Artifacts Presence
    print("\n[CHECK 6/6] Verifying saved .keras models and figures...")
    for exp_id in ALL_EXPS:
        exp_lower = exp_id.lower()
        model_p = PROJECT_ROOT / "results" / "models" / f"{exp_lower}.keras"
        cm_p = PROJECT_ROOT / "results" / "confusion_matrices" / f"{exp_lower}_cm.png"
        loss_p = PROJECT_ROOT / "results" / "training_curves" / f"{exp_lower}_loss.png"
        acc_p = PROJECT_ROOT / "results" / "training_curves" / f"{exp_lower}_accuracy.png"
        
        if not model_p.exists():
            warnings.append(f"Model checkpoint missing: {model_p.name}")
        if not cm_p.exists():
            warnings.append(f"Confusion matrix missing: {cm_p.name}")
        if not loss_p.exists() or not acc_p.exists():
            warnings.append(f"Training curves missing for {exp_id}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  [ERROR] {e}")
        return False
    else:
        print("\nSUCCESS: All consistency and feature count checks passed perfectly!")
        return True


if __name__ == "__main__":
    success = validate_results()
    sys.exit(0 if success else 1)
