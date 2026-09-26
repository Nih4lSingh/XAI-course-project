"""
LIME Explainer Module
Replication of Sharma et al. (2024)

Rules:
- Uses LIME TabularExplainer.
- Target model: DNN trained on selected features.
- Generates local explanations for representative test instances (Attack and Normal).
- Records: actual class, predicted class, prediction probabilities, feature values, positive/negative contributions.
- Plots contribution bar charts and saves JSON explanation metadata.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class LimeExplainerWrapper:
    """
    Wraps LIME TabularExplainer for multi-class deep learning models.
    """
    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        random_state: int = 42
    ):
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.random_state = random_state

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification",
            random_state=self.random_state
        )

    def explain_instance(
        self,
        model: tf.keras.Model,
        instance: np.ndarray,
        actual_class: int,
        instance_id: int,
        num_features: int = 10,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Explains single prediction and saves contribution bar plot.
        """
        # Model prediction function
        def predict_fn(x):
            return model.predict(x, verbose=0)

        probs = predict_fn(instance.reshape(1, -1))[0]
        predicted_class = int(np.argmax(probs))

        # Generate LIME explanation for the predicted class
        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            labels=[predicted_class]
        )

        exp_map = exp.as_list(label=predicted_class)

        # Plot explanation
        features = [x[0] for x in exp_map]
        weights = [x[1] for x in exp_map]
        colors = ["green" if w > 0 else "red" for w in weights]

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(9, 5))
            y_pos = np.arange(len(features))
            plt.barh(y_pos, weights, color=colors, align="center")
            plt.yticks(y_pos, features)
            plt.gca().invert_yaxis()
            plt.xlabel("LIME Feature Weight (Contribution)", fontsize=10)
            pred_name = self.class_names[predicted_class]
            act_name = self.class_names[actual_class]
            plt.title(f"LIME Explanation - Instance #{instance_id}\nActual: {act_name} | Predicted: {pred_name} ({probs[predicted_class]*100:.1f}%)", fontsize=11)
            plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
            plt.tight_layout()
            
            plot_path = output_dir / f"lime_instance_{instance_id}_pred_{pred_name.lower()}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"[LIME] Saved plot to {plot_path}")

        # Metadata
        explanation_data = {
            "instance_id": int(instance_id),
            "actual_class": int(actual_class),
            "actual_class_name": self.class_names[actual_class],
            "predicted_class": int(predicted_class),
            "predicted_class_name": self.class_names[predicted_class],
            "probabilities": {name: float(probs[i]) for i, name in enumerate(self.class_names)},
            "top_features": [
                {
                    "feature_condition": rule,
                    "weight": float(weight),
                    "direction": "positive" if weight > 0 else "negative"
                }
                for rule, weight in exp_map
            ]
        }

        if output_dir is not None:
            json_path = output_dir / f"lime_instance_{instance_id}_meta.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(explanation_data, f, indent=2)

        return explanation_data
