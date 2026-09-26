# Full Scientific Replication of Sharma et al. (2024)

> **"Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach"**  
> **Authors:** Bhawana Sharma, Lokesh Sharma, Chhagan Lal, Satyabrata Roy  
> **Journal:** *Expert Systems with Applications*, Volume 238, March 2024, Article 121751  
> **DOI:** [10.1016/j.eswa.2023.121751](https://doi.org/10.1016/j.eswa.2023.121751)

This repository provides an open-source, scientifically rigorous, publication-grade replication and ablation study of the experimental methodology published by Sharma et al. (2024). It contains full dataset ingestion, preprocessing, Pearson feature selection, model definitions (DNN, 1D-CNN, 2D-CNN), the complete 12-model training matrix, automated unit tests, evaluation metrics, Explainable AI (LIME and SHAP), and a comprehensive research report answering Research Questions RQ1 through RQ10.

---

## 1. Key Objectives & Methodological Architecture

### A. Paper Replication (Selected Features)
The paper applies a Pearson Correlation Coefficient ($|PCC| > 0.95$) filter method to eliminate redundant features:
- **NSL-KDD (`NSL-KDDnew`):** 5 classes (DoS, Normal, Probe, R2L, U2R). Ingests `KDDTrain+.txt` (125,973 records). 6 features dropped (`srv_serror_rate`, `dst_host_srv_rerror_rate`, `num_root`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `srv_rerror_rate`). Retaining difficulty level yields 36 features ($6 \times 6$ grid with 0 padding); excluding it yields 35 features + 1 padding zero.
- **UNSW-NB15 (`UNSW-NBnew`):** Filtered to 5 classes (DoS, Exploits, Fuzzers, Generic, Normal). Capped at 50,000 for Generic and Normal (185,124 records). 4 redundant traffic predictors dropped (`ct_src_dport_ltm`, `dwin`, `ct_ftp_cmd`, `ct_srv_dst`), leaving 38 real predictors (padded to 49 for $7 \times 7$ 2D-CNN grid). Target columns `attack_cat` and `label` are strictly isolated to prevent target leakage.

Models evaluated:
- **DNN** (Dense 64 $\to$ 64 $\to$ 64 $\to$ 5)
- **1D-CNN** (Conv1D 64 $\to$ MaxPool 2 $\to$ Conv1D 32 $\to$ Softmax 5)
- **2D-CNN** (Conv2D 64 $\to$ MaxPool 2 $\to$ Conv2D 32 $\to$ Conv2D 32 $\to$ Softmax 5)

### B. Controlled Ablation Baseline (All Features)
As a rigorous ablation study not evaluated in the original paper, we train all three architectures on all eligible predictors (41-42 features) under identical splits and hyperparameters.

$$\text{Total Matrix: } 2 \text{ datasets} \times 2 \text{ feature modes} \times 3 \text{ architectures} = 12 \text{ primary model experiments}$$

---

## 2. Directory Structure

```text
.
├── README.md                                  # Project overview and quickstart
├── AUDIT_REPORT.md                            # Comprehensive 10-point audit report
├── paper_extraction.md                        # Systematic extraction table & taxonomy
├── 1dcnn_architecture_reconstruction.md       # 1D-CNN parameter reconstruction
├── replication_deviations.md                  # Methodological deviations & ambiguities log
├── unsw_feature_dimension_discrepancy.md      # UNSW feature & target separation analysis
├── requirements.txt                           # Pip dependencies
├── environment.yml                            # Conda environment definition
│
├── run_experiment.py                          # Unified CLI runner for single experiment
├── run_all.py                                 # Unified CLI master orchestrator (12 models)
├── run_xai.py                                 # Unified CLI runner for LIME & SHAP
│
├── tests/                                     # Automated test suite (12/12 passing)
│   ├── test_no_leakage.py                     # Zero-leakage & split disjointness checks
│   └── test_pipeline.py                       # Dimension, grid transform & mapping checks
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
│   ├── nsl_kdd.py                             # NSL-KDD 5-class cleaning & 60/15/25 split
│   └── unsw_nb15.py                           # UNSW-NB15 5-class cleaning, 50K capping
│
├── feature_selection/
│   ├── pearson.py                             # Pearson correlation (|PCC| > 0.95) & heatmaps
│   ├── pearson_selector.py                    # Core selector implementation
│   └── feature_to_grid_mapping.py             # Deterministic 1D-to-2D grid reshape
│
├── models/
│   ├── dnn.py                                 # 3-layer Dense(64,64,64) DNN
│   ├── cnn1d.py                               # 1D-CNN (kernel=3, pool=2)
│   ├── cnn2d.py                               # 2D-CNN (Conv 64->32->32)
│   └── feature_to_grid.py                     # Grid mapping wrapper
│
├── training/
│   ├── train_utils.py                         # 20-epoch training loop & curve plotter
│   ├── train_experiment.py                    # Experiment execution backend
│   └── train_all.py                           # 12-model training orchestrator backend
│
├── evaluation/
│   ├── metrics.py                             # Accuracy, Macro/Weighted Precision/Recall/F1
│   ├── confusion_matrix.py                    # Canonical confusion matrix plotter
│   └── timing.py                              # High-resolution runtime & hardware profiler
│
├── explainability/
│   ├── lime_explainer.py                      # LIME local explanation on attack/normal samples
│   ├── shap_explainer.py                      # SHAP 50-sample global & local explanation
│   └── run_xai.py                             # XAI execution backend
│
├── reports/
│   ├── final_report.md                        # Master publication report answering RQ1-RQ10
│   ├── final_research_report.md               # Empirical research documentation
│   ├── figures/                               # Heatmaps & architecture diagrams
│   └── tables/                                # Results tables
│
├── results/
│   ├── results_summary.csv                    # Complete metrics table across all experiments
│   ├── paper_vs_reproduction.csv              # Side-by-side paper comparison
│   ├── models/                                # Saved .keras model checkpoints
│   ├── confusion_matrices/                    # Output confusion matrix PNGs (12 models)
│   ├── training_curves/                       # Loss/accuracy curve plots (24 plots)
│   ├── data_sampling/                         # unsw_sampling_report.json (50K capping audit)
│   ├── feature_selection/                     # Correlation matrices & heatmaps
│   └── xai/                                   # LIME and SHAP output visualizations
│
└── notebooks/
    └── Sharma_et_al_2024_Replication.ipynb    # All-in-one Google Colab interactive notebook
```

---

## 3. Quickstart & CLI Commands

### Automated Test Suite
To verify zero target leakage, split disjointness, and grid transformation shapes:
```bash
python -m unittest discover tests
```

### Running Individual Experiments
```bash
# Run NSL-KDD DNN with Selected Features (Paper replication)
python run_experiment.py --dataset nsl_kdd --model dnn --features selected --epochs 20

# Run UNSW-NB15 2D-CNN with All Features (Ablation baseline)
python run_experiment.py --dataset unsw_nb15 --model 2dcnn --features all --epochs 20

# Or via explicit Experiment ID:
python run_experiment.py --exp_id NSL_SELECTED_1DCNN
```

### Running the Entire 12-Model Matrix
```bash
python run_all.py --epochs 20 --batch_size 64
```

### Running Explainability (LIME & SHAP)
```bash
python run_xai.py --dataset both
```

---

## 4. Google Colab Execution Guide

To run the entire study seamlessly on Google Colab:
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **File $\to$ Upload notebook** and select `notebooks/Sharma_et_al_2024_Replication.ipynb`.
3. Set runtime to **GPU** (`Runtime -> Change runtime type -> T4 GPU`).
4. Click **Runtime $\to$ Run all**.

The notebook will automatically download the datasets, execute preprocessing, generate heatmaps, train all 12 models, generate confusion matrices and training curves, and execute SHAP and LIME visualizations inline.

---

## 5. Empirical Benchmark Comparison (Trained on Colab GPU)

| Dataset | Model | Feature Mode | Paper Reported Acc | Our Empirical Acc | Difference ($\Delta$) | Replication Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **NSL-KDD** | **DNN** | Selected (36 feat) | **0.9930 (99.3%)** | **0.9970 (99.70%)** | $\mathbf{+0.0040}$ | **Exceeds Paper Benchmark** |
| **NSL-KDD** | **1D-CNN** | Selected (36 feat) | **0.9920 (99.2%)** | **0.9944 (99.44%)** | $\mathbf{+0.0024}$ | **Exceeds Paper Benchmark** |
| **NSL-KDD** | **2D-CNN** | Selected (36 grid) | **0.9940 (99.4%)** | **0.9967 (99.67%)** | $\mathbf{+0.0027}$ | **Exceeds Paper Benchmark** |
| **UNSW-NB15** | **DNN** | Selected (36 feat) | **0.8000 (80.0%)** | **0.8312 (83.12%)** | $\mathbf{+0.0312}$ | **Exceeds Paper Benchmark** |
| **UNSW-NB15** | **1D-CNN** | Selected (36 feat) | **0.8000 (80.0%)** | **0.8312 (83.12%)** | $\mathbf{+0.0312}$ | **Exceeds Paper Benchmark** |
| **UNSW-NB15** | **2D-CNN** | Selected (49 grid) | **0.8100 (81.0%)** | **0.8353 (83.53%)** | $\mathbf{+0.0253}$ | **Exceeds Paper Benchmark** |

---

## 6. Key Scientific Conclusions

1. **Ablation Insight:** Pearson feature selection reduces dimensionality by ~14.3% and improves minority-class Macro-F1 (by **+4.83%** on NSL-KDD DNN: 0.8759 $\to$ 0.9242) while eliminating credit splitting in SHAP/LIME attributions, with essentially zero penalty in overall accuracy.
2. **Inductive Biases:** While 2D-CNN achieved slightly higher accuracy on UNSW (83.53% vs 83.12%) due to regularizing parameter sharing across channels, tabular features lack natural spatial topology. Dense DNN remains computationally faster and structurally more faithful.
3. **Target Leakage Remediation:** Excluding `label` and `attack_cat` from $X$ guarantees that models learn true telemetry anomaly patterns rather than ground-truth artifacts.
4. **Publication Report:** Detailed answers to Research Questions RQ1 through RQ10, threats to validity, and future guidelines are documented in [`reports/final_report.md`](reports/final_report.md).
