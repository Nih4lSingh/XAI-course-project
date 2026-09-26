"""
Pearson Correlation Feature Selection Module
Sharma et al. (2024) Replication Master Pipeline

Computes pairwise Pearson correlation coefficients (|PCC| > 0.95),
generates correlation matrices and heatmaps, and identifies redundant features.
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_selection.pearson_selector import PearsonCorrelationSelector

__all__ = ["PearsonCorrelationSelector"]


def run_pearson_analysis():
    """Computes and saves correlation matrices and heatmaps for both datasets."""
    from preprocessing.nsl_kdd import load_raw_nsl_kdd, NSL_KDD_CATEGORICAL_COLS, NSL_KDD_COLUMNS, NSL_KDD_PAPER_REMOVED_FEATURES
    from preprocessing.encoders import CategoricalFeaturePipeline
    from preprocessing.unsw_nb15 import load_raw_unsw_nb15, apply_sampling_policy, UNSW_CATEGORICAL_COLS, UNSW_NON_PREDICTORS, UNSW_PAPER_REMOVED_PREDICTORS

    out_dir = PROJECT_ROOT / "results" / "feature_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[PEARSON] Running correlation analysis for NSL-KDD...")
    nsl_raw = load_raw_nsl_kdd(PROJECT_ROOT / "data" / "raw" / "nsl_kdd", use_train_only=True)
    nsl_predictors = [c for c in NSL_KDD_COLUMNS if c != "label"]
    nsl_df = nsl_raw[nsl_predictors].copy()
    nsl_enc = CategoricalFeaturePipeline(NSL_KDD_CATEGORICAL_COLS)
    nsl_encoded = nsl_enc.fit_transform(nsl_df)

    selector_nsl = PearsonCorrelationSelector(threshold=0.95)
    selector_nsl.fit(nsl_encoded, target_removed=NSL_KDD_PAPER_REMOVED_FEATURES)
    
    # Save correlation matrix
    selector_nsl.corr_matrix_.to_csv(out_dir / "nsl_kdd_correlation_matrix.csv")
    selector_nsl.save_heatmap(out_dir / "nsl_kdd_correlation_heatmap.png", title="NSL-KDD Pearson Correlation Heatmap")
    print(f"[PEARSON] NSL-KDD correlated pairs (|PCC| > 0.95): {len(selector_nsl.correlated_pairs_)}")
    print(f"[PEARSON] NSL-KDD removed features: {selector_nsl.removed_features_}")

    print("\n[PEARSON] Running correlation analysis for UNSW-NB15...")
    unsw_raw = load_raw_unsw_nb15(PROJECT_ROOT / "data" / "raw" / "unsw_nb15")
    unsw_raw["attack_clean"] = unsw_raw["attack_cat"].fillna("normal").astype(str).str.lower().str.strip()
    unsw_raw["attack_clean"] = unsw_raw["attack_clean"].replace({"backdoors": "backdoor"})
    unsw_5class = unsw_raw[unsw_raw["attack_clean"].isin(["dos", "exploits", "fuzzers", "generic", "normal"])].copy()
    unsw_capped = apply_sampling_policy(unsw_5class, cap_generic=50000, cap_normal=50000, random_seed=42)

    unsw_predictors = [c for c in unsw_capped.columns if c not in UNSW_NON_PREDICTORS and c not in ("attack_clean", "target")]
    unsw_df = unsw_capped[unsw_predictors].copy()
    unsw_enc = CategoricalFeaturePipeline(UNSW_CATEGORICAL_COLS)
    unsw_encoded = unsw_enc.fit_transform(unsw_df)

    selector_unsw = PearsonCorrelationSelector(threshold=0.95)
    selector_unsw.fit(unsw_encoded, target_removed=UNSW_PAPER_REMOVED_PREDICTORS)

    # Save correlation matrix
    selector_unsw.corr_matrix_.to_csv(out_dir / "unsw_correlation_matrix.csv")
    selector_unsw.save_heatmap(out_dir / "unsw_correlation_heatmap.png", title="UNSW-NB15 Pearson Correlation Heatmap")
    print(f"[PEARSON] UNSW-NB15 correlated pairs (|PCC| > 0.95): {len(selector_unsw.correlated_pairs_)}")
    print(f"[PEARSON] UNSW-NB15 removed features: {selector_unsw.removed_features_}")


if __name__ == "__main__":
    run_pearson_analysis()
