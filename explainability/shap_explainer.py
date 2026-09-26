"""
SHAP Explainer Module
Replication of Sharma et al. (2024)

Rules:
- Evaluates the DNN model.
- Uses exact 50 testing samples for global explanation (Section 5.2).
- Generates:
  1. Global summary / beeswarm plot across 50 test samples.
  2. Mean absolute SHAP feature importance ranking: mean(|SHAP|).
  3. Local explanation (waterfall / bar plot) for representative instances.
- Records exact indices and saves high-resolution plots.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf


class ShapExplainerWrapper:
    """
    Computes global and local SHAP explanations for Deep Neural Networks.
    """
    def __init__(
        self,
        model: tf.keras.Model,
        background_data: np.ndarray,
        feature_names: List[str],
        class_names: List[str]
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Build KernelExplainer or DeepExplainer
        def predict_fn(x):
            return model.predict(x, verbose=0)

        self.predict_fn = predict_fn
        # Use a background summary (e.g. 50-100 k-means or median samples)
        if len(background_data) > 100:
            self.background = shap.sample(background_data, 100, random_state=42)
        else:
            self.background = background_data

        print("[SHAP] Initializing KernelExplainer with background sample...")
        self.explainer = shap.KernelExplainer(self.predict_fn, self.background)

    def explain_global(
        self,
        X_test_50: np.ndarray,
        test_indices_50: np.ndarray,
        output_dir: Path,
        dataset_name: str = "dataset"
    ) -> Dict:
        """
        Executes paper's global SHAP analysis across 50 testing samples.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SHAP] Computing SHAP values for {len(X_test_50)} test instances...")

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_test_50, nsamples=100)
        # shap_values is a list of [n_samples, n_features] for each class or array (50, features, classes)

        if isinstance(shap_values, list):
            # Shape: list of (50, n_features) for each class
            # Calculate mean absolute SHAP across all classes and samples
            abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)  # (50, n_features)
            mean_abs_shap = np.mean(abs_shap, axis=0)  # (n_features,)
        else:
            # (50, n_features, n_classes)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))

        # Rank features by mean(|SHAP|)
        ranked_indices = np.argsort(mean_abs_shap)[::-1]
        ranking = [
            {
                "rank": rank + 1,
                "feature": self.feature_names[idx],
                "mean_abs_shap": float(mean_abs_shap[idx])
            }
            for rank, idx in enumerate(ranked_indices)
        ]

        print(f"\n[SHAP Top 10 Features - {dataset_name.upper()}]:")
        for item in ranking[:10]:
            print(f"  {item['rank']}. {item['feature']}: {item['mean_abs_shap']:.5f}")

        # 1. Bar Plot of Feature Importance
        plt.figure(figsize=(9, 7))
        top_k = min(15, len(ranking))
        top_feats = [r["feature"] for r in ranking[:top_k]][::-1]
        top_scores = [r["mean_abs_shap"] for r in ranking[:top_k]][::-1]

        plt.barh(range(top_k), top_scores, color="#1f77b4", align="center")
        plt.yticks(range(top_k), top_feats)
        plt.xlabel("Mean Absolute SHAP Value (Global Impact)", fontsize=10)
        plt.title(f"SHAP Global Feature Importance ({dataset_name.upper()})\n(Evaluated over 50 test samples)", fontsize=11)
        plt.tight_layout()
        bar_plot_path = output_dir / f"shap_global_importance_{dataset_name}.png"
        plt.savefig(bar_plot_path, dpi=300)
        plt.close()
        print(f"[SHAP] Saved global importance bar plot to {bar_plot_path}")

        # 2. Summary Beeswarm Plot for primary predicted class (e.g. Normal or Attack)
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            sv_to_plot = shap_values[1]  # Class 1 (Normal in NSL, Exploits in UNSW)
        else:
            sv_to_plot = shap_values[:, :, 1]

        shap.summary_plot(
            sv_to_plot,
            X_test_50,
            feature_names=self.feature_names,
            show=False,
            max_display=15
        )
        plt.title(f"SHAP Beeswarm Summary Plot ({dataset_name.upper()} - 50 Test Samples)", fontsize=12, pad=12)
        plt.tight_layout()
        summary_plot_path = output_dir / f"shap_beeswarm_summary_{dataset_name}.png"
        plt.savefig(summary_plot_path, dpi=300)
        plt.close()
        print(f"[SHAP] Saved beeswarm summary plot to {summary_plot_path}")

        # Save metadata JSON
        meta = {
            "dataset": dataset_name,
            "num_test_samples": len(X_test_50),
            "test_sample_indices": [int(idx) for idx in test_indices_50],
            "feature_importance_ranking": ranking
        }
        with open(output_dir / f"shap_global_{dataset_name}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    def explain_local_instance(
        self,
        instance: np.ndarray,
        actual_class: int,
        instance_id: int,
        output_dir: Path,
        dataset_name: str = "dataset"
    ) -> Dict:
        """
        Generates local SHAP explanation for an individual test instance.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        probs = self.predict_fn(instance.reshape(1, -1))[0]
        pred_class = int(np.argmax(probs))

        # Compute SHAP values for this instance
        shap_vals = self.explainer.shap_values(instance.reshape(1, -1), nsamples=100)

        if isinstance(shap_vals, list):
            inst_shap = shap_vals[pred_class][0]
            base_val = float(self.explainer.expected_value[pred_class])
        else:
            inst_shap = shap_vals[0, :, pred_class]
            base_val = float(self.explainer.expected_value[pred_class])

        # Top local features
        top_indices = np.argsort(np.abs(inst_shap))[::-1][:10]
        local_features = [self.feature_names[i] for i in top_indices]
        local_contribs = [float(inst_shap[i]) for i in top_indices]
        local_vals = [float(instance[i]) for i in top_indices]

        # Plot local contribution bar chart
        plt.figure(figsize=(9, 5))
        y_pos = np.arange(len(local_features))
        colors = ["green" if c > 0 else "red" for c in local_contribs[::-1]]

        plt.barh(y_pos, local_contribs[::-1], color=colors, align="center")
        plt.yticks(y_pos, [f"{f} = {v:.2f}" for f, v in zip(local_features[::-1], local_vals[::-1])])
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
        plt.xlabel("SHAP Contribution Value", fontsize=10)
        pred_name = self.class_names[pred_class]
        act_name = self.class_names[actual_class]
        plt.title(f"SHAP Local Explanation - Instance #{instance_id} ({dataset_name.upper()})\nActual: {act_name} | Predicted: {pred_name} (Prob: {probs[pred_class]*100:.1f}%)", fontsize=11)
        plt.tight_layout()

        local_plot_path = output_dir / f"shap_local_instance_{instance_id}_{dataset_name}.png"
        plt.savefig(local_plot_path, dpi=300)
        plt.close()
        print(f"[SHAP] Saved local plot to {local_plot_path}")

        meta = {
            "instance_id": int(instance_id),
            "dataset": dataset_name,
            "actual_class": self.class_names[actual_class],
            "predicted_class": self.class_names[pred_class],
            "base_value": base_val,
            "prediction_probability": float(probs[pred_class]),
            "feature_contributions": [
                {
                    "feature": f,
                    "value": v,
                    "shap_contribution": c,
                    "direction": "positive" if c > 0 else "negative"
                }
                for f, v, c in zip(local_features, local_vals, local_contribs)
            ]
        }
        with open(output_dir / f"shap_local_instance_{instance_id}_{dataset_name}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta
