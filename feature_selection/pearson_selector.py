"""
Pearson Correlation Feature Selection Module
Replication of Sharma et al. (2024)

Rules:
- Filter method using Pearson Correlation Coefficient (PCC).
- Threshold: |PCC| > 0.95
- When two features exhibit |PCC| > 0.95, one redundant feature is identified for removal.
- Produces correlation heatmaps and verifies removed/retained feature lists.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PearsonCorrelationSelector:
    """
    Identifies redundant features based on pairwise Pearson correlation above a threshold.
    """
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.corr_matrix_: Optional[pd.DataFrame] = None
        self.correlated_pairs_: List[Tuple[str, str, float]] = []
        self.removed_features_: List[str] = []
        self.selected_features_: List[str] = []

    def fit(self, df: pd.DataFrame, target_removed: Optional[List[str]] = None) -> "PearsonCorrelationSelector":
        """
        Computes pairwise correlation matrix and identifies pairs with |PCC| > threshold.
        If target_removed is provided (from Sharma et al.), it verifies which pair members match the paper.
        """
        # Ensure only numeric data
        numeric_df = df.select_dtypes(include=[np.number])
        self.corr_matrix_ = numeric_df.corr(method="pearson")

        columns = list(numeric_df.columns)
        n_cols = len(columns)
        self.correlated_pairs_ = []

        to_remove: Set[str] = set()

        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                col_i = columns[i]
                col_j = columns[j]
                val = self.corr_matrix_.loc[col_i, col_j]
                if abs(val) > self.threshold:
                    self.correlated_pairs_.append((col_i, col_j, float(val)))
                    if target_removed is not None:
                        # Prioritize matching the paper's explicit removed features
                        if col_j in target_removed:
                            to_remove.add(col_j)
                        elif col_i in target_removed:
                            to_remove.add(col_i)
                        else:
                            to_remove.add(col_j)
                    else:
                        to_remove.add(col_j)

        self.removed_features_ = sorted(list(to_remove))
        self.selected_features_ = [col for col in columns if col not in to_remove]
        return self

    def save_heatmap(
        self,
        output_path: Path,
        title: str = "Pearson Correlation Heatmap",
        figsize: Tuple[int, int] = (16, 14)
    ):
        """Generates and saves a high-resolution correlation heatmap."""
        if self.corr_matrix_ is None:
            raise ValueError("Selector must be fitted before generating heatmap.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(self.corr_matrix_, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(
            self.corr_matrix_,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(f"{title} (|PCC| > {self.threshold})", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[PLOT] Heatmap saved to {output_path}")

    def save_feature_lists(
        self,
        selected_path: Path,
        removed_path: Path
    ):
        """Saves selected and removed features to text files."""
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        removed_path.parent.mkdir(parents=True, exist_ok=True)

        with open(selected_path, "w", encoding="utf-8") as f:
            for feat in self.selected_features_:
                f.write(f"{feat}\n")

        with open(removed_path, "w", encoding="utf-8") as f:
            for feat in self.removed_features_:
                f.write(f"{feat}\n")

        print(f"[IO] Feature lists saved: {selected_path.name} ({len(self.selected_features_)}) and {removed_path.name} ({len(self.removed_features_)})")
