# Full Scientific Replication of Sharma et al. (2024)

> **"Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach"**  
> **Authors:** Bhawana Sharma, Lokesh Sharma, Chhagan Lal, Satyabrata Roy  
> **Journal:** *Expert Systems with Applications*, Volume 238, March 2024, Article 121751  
> **DOI:** [10.1016/j.eswa.2023.121751](https://doi.org/10.1016/j.eswa.2023.121751)

This repository provides an open-source, scientifically rigorous, end-to-end replication of the experimental methodology from Sharma et al. (2024). It contains full dataset ingestion, preprocessing, Pearson feature selection, model definitions (DNN, 1D-CNN, 2D-CNN), the complete 12-model training matrix, evaluation metrics, and Explainable AI (LIME and SHAP).

---

## 1. Key Objectives & Methodological Architecture

### A. Paper Replication (Selected Features)
The paper applies a Pearson Correlation Coefficient ($|PCC| > 0.95$) filter method to eliminate redundant features:
- **NSL-KDD (`NSL-KDDnew`):** 5 classes (DoS, Normal, Probe, R2L, U2R). 6 features dropped (`srv_serror_rate`, `dst_host_srv_rerror_rate`, `num_root`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `srv_rerror_rate`), leaving 35 network predictors (padded to 36 for $6 \times 6$ 2D-CNN grid).
- **UNSW-NB15 (`UNSW-NBnew`):** Filtered to 5 classes (DoS, Exploits, Fuzzers, Generic, Normal). 6 features dropped (`ct_src_dport_ltm`, `loss`, `dwin`, `ct_ftp_cmd`, `label`, `ct_srv_dst`), leaving 36 network predictors (padded to 49 for $7 \times 7$ 2D-CNN grid). Target columns `attack_cat` and `label` are strictly isolated to prevent target leakage.

Models trained on selected features:
- **DNN** (Dense 64 $\to$ 64 $\to$ 64 $\to$ 5)
- **1D-CNN** (Conv1D 64 $\to$ MaxPool 2 $\to$ Conv1D 32 $\to$ Softmax 5)
- **2D-CNN** (Conv2D 64 $\to$ MaxPool 2 $\to$ Conv2D 32 $\to$ Conv2D 32 $\to$ Softmax 5)

### B. Our Additional Ablation Baseline (All Features)
As an independent ablation not explicitly conducted by Sharma et al., we train all three architectures on all eligible predictors (41 for NSL-KDD, 42 for UNSW-NB15) under identical splits and hyperparameters.

$$\text{Total Matrix: } 2 \text{ datasets} \times 2 \text{ feature modes} \times 3 \text{ architectures} = 12 \text{ primary model experiments}$$

---

## 2. Directory Structure

```text
.
├── README.md                                  # This document
├── paper_extraction.md                        # Systematic extraction table & taxonomy
├── replication_deviations.md                  # Detailed ambiguity & deviation log
├── unsw_feature_dimension_discrepancy.md      # Detailed UNSW feature analysis
├── feature_selection_report.md                # Feature selection verification & heatmaps
├── requirements.txt                           # Pip requirements
├── environment.yml                            # Conda environment definition
│
├── data/
│   ├── download_datasets.py                   # Automated downloader with mirrors
│   ├── raw/                                   # Downloaded raw CSV / TXT files
│   └── processed/                             # Preprocessed arrays (.npz)
│
├── splits/                                    # Exact train/val/test split indices (.npy)
│
├── preprocessing/
│   ├── encoders.py                            # Categorical & target label encoders
│   ├── normalization.py                       # Min-Max scaler mapping to [0, 1]
│   ├── nsl_kdd_preprocessing.py               # NSL-KDD 5-class cleaning & 60/15/25 split
│   └── unsw_preprocessing.py                  # UNSW-NB15 5-class cleaning & 60/15/25 split
│
├── feature_selection/
│   ├── pearson_selector.py                    # Pearson correlation (|PCC| > 0.95) & heatmap
│   ├── feature_to_grid_mapping.py             # Deterministic 1D-to-2D grid reshape
│   ├── nsl_kdd_features.py                    # NSL-KDD selection verification
│   └── unsw_features.py                       # UNSW-NB15 selection verification
│
├── models/
│   ├── dnn.py                                 # 3-layer Dense(64,64,64) DNN
│   ├── cnn1d.py                               # 1D-CNN (kernel=3, pool=2)
│   └── cnn2d.py                               # 2D-CNN (Conv 64->32->32)
│
├── training/
│   ├── train_utils.py                         # 20-epoch training loop & curve plotter
│   ├── train_experiment.py                    # Single experiment runner
│   └── train_all.py                           # 12-model matrix orchestrator
│
├── evaluation/
│   ├── metrics.py                             # Accuracy, Macro/Weighted Precision/Recall/F1
│   ├── confusion_matrix.py                    # Canonical confusion matrix plotter
│   └── timing.py                              # High-resolution runtime & hardware profiler
│
├── explainability/
│   ├── lime_explainer.py                      # LIME local explanation on attack/normal samples
│   ├── shap_explainer.py                      # SHAP 50-test-sample global & local explanation
│   └── run_xai.py                             # Master XAI runner
│
├── experiments/
│   ├── configs/                               # Central JSON experiment configurations
│   └── validate_pipeline.py                   # Data leakage & dimension validation script
│
├── results/
│   ├── results_summary.csv                    # Complete metrics table across all experiments
│   ├── paper_vs_reproduction.csv              # Side-by-side paper comparison
│   ├── models/                                # Saved .keras model checkpoints
│   ├── confusion_matrices/                    # Output confusion matrix PNGs
│   ├── training_curves/                       # Loss/accuracy curve plots
│   └── xai/                                   # LIME and SHAP output visualizations
│
└── notebooks/
    └── Sharma_et_al_2024_Replication.ipynb    # All-in-one Google Colab interactive notebook
```

---

## 3. Google Colab Execution Guide

To run the entire study seamlessly on Google Colab:
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **File $\to$ Upload notebook** and select `notebooks/Sharma_et_al_2024_Replication.ipynb`.
3. Set runtime to **GPU** (`Runtime -> Change runtime type -> T4 GPU`).
4. Click **Runtime $\to$ Run all**.

The notebook will automatically download the datasets, execute preprocessing, generate heatmaps, train all 12 models, generate confusion matrices and training curves, and execute SHAP and LIME visualizations inline.

---

## 4. Local Execution Guide

### Step 1: Environment Setup
```bash
git clone <repo_url>
cd xai_code
pip install -r requirements.txt
```

### Step 2: Download Raw Datasets
```bash
python data/download_datasets.py
```

### Step 3: Run Preprocessing & Splitting (60/15/25)
```bash
python preprocessing/nsl_kdd_preprocessing.py
python preprocessing/unsw_preprocessing.py
```

### Step 4: Verify Data Leakage & Integrity
```bash
python experiments/validate_pipeline.py
```

### Step 5: Verify Feature Selection & Generate Heatmaps
```bash
python feature_selection/nsl_kdd_features.py
python feature_selection/unsw_features.py
```

### Step 6: Train Models (Individual or Full Matrix)
To train an individual model:
```bash
python training/train_experiment.py --exp_id NSL_SELECTED_DNN --epochs 20
```
To run the full 12-model matrix + sensitivity analysis:
```bash
python training/train_all.py
```

### Step 7: Run Explainable AI (LIME & SHAP)
```bash
python explainability/run_xai.py
```

---

## 5. Empirical Benchmark Comparison (Trained on Colab GPU)

| Dataset | Model | Feature Mode | Paper Reported Acc | Our Empirical Acc | Difference ($\Delta$) | Replication Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **NSL-KDD** | **DNN** | Selected (35 feat) | **0.9930 (99.3%)** | **0.9825 (98.25%)** | $-0.0105$ | **Faithfully Reproduced** |
| **NSL-KDD** | **1D-CNN** | Selected (35 feat) | **0.9920 (99.2%)** | **0.9757 (97.57%)** | $-0.0163$ | **Faithfully Reproduced** |
| **NSL-KDD** | **2D-CNN** | Selected (36 grid) | **0.9940 (99.4%)** | **0.9847 (98.47%)** | $-0.0093$ | **Faithfully Reproduced** |
| **UNSW-NB15** | **DNN** | Selected (36 feat) | **0.8000 (80.0%)** | **0.8320 (83.20%)** | $+0.0320$ | **Faithfully Reproduced** |
| **UNSW-NB15** | **1D-CNN** | Selected (36 feat) | **0.8000 (80.0%)** | **0.8312 (83.12%)** | $+0.0312$ | **Faithfully Reproduced** |
| **UNSW-NB15** | **2D-CNN** | Selected (49 grid) | **0.8100 (81.0%)** | **0.8353 (83.53%)** | $+0.0253$ | **Faithfully Reproduced** |

---

## 6. Scientific Traceability & Ethics
- **No target leakage:** Predictor set $X$ never contains ground-truth targets (`label` or `attack_cat`).
- **No data snooping:** Min-max scalers and encoders are preserved deterministically.
- **Explicit parameter labeling:** Every parameter is explicitly designated as `EXPLICIT`, `INFERRED`, or `AMBIGUOUS` in `paper_extraction.md` and `replication_deviations.md`.
