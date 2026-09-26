# Comprehensive Codebase Audit Report: Sharma et al. (2024) Replication

**Audit Date:** 2026-09-27  
**Scope:** Evaluation of existing codebase (`preprocessing/`, `feature_selection/`, `models/`, `training/`, `evaluation/`, `explainability/`, `results/2d_cnn/`, `notebooks/`) against the explicit experimental methodology of:  
**Bhawana Sharma, Lokesh Sharma, Chhagan Lal, Satyabrata Roy (2024)**, *“Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach”*, *Expert Systems with Applications*, 238, 121751.

---

## 1. What Already Works
- **Data Ingestion Infrastructure:** `data/download_datasets.py` reliably downloads both `KDDTrain+.txt`, `KDDTest+.txt`, `UNSW_NB15_training-set.csv`, and `UNSW_NB15_testing-set.csv` with mirror failover.
- **Min-Max Scaling:** `preprocessing/normalization.py` implements the exact normalization formula $F_{new} = \frac{F - F_{min}}{F_{max} - F_{min}} \in [0, 1]$ and saves scaling parameters to JSON.
- **Pearson Filter Logic:** `feature_selection/pearson_selector.py` accurately identifies pairwise Pearson correlation $|PCC| > 0.95$ and generates correlation heatmaps.
- **DNN Architecture:** `models/dnn.py` correctly implements the 3-layer Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(5, Softmax) with Adam ($lr=0.001, decay=0.0001$) and Sparse Categorical Cross-Entropy.
- **Evaluation Infrastructure:** `evaluation/metrics.py` and `evaluation/confusion_matrix.py` compute macro/weighted Precision, Recall, F1, Accuracy, and canonical 5-class confusion matrices.
- **XAI Pipeline:** `explainability/lime_explainer.py` and `explainability/shap_explainer.py` support local LIME explanations and 50-sample global/local SHAP attributions.

---

## 2. What Matches the Paper
- **Five-Class Formulations:**
  - NSL-KDD: 0 = DoS, 1 = Normal, 2 = Probe, 3 = R2L, 4 = U2R.
  - UNSW-NB15: 0 = DoS, 1 = Exploits, 2 = Fuzzers, 3 = Generic, 4 = Normal.
- **Normalization:** Min-Max scaling to $[0, 1]$.
- **Correlation Filter Threshold:** $|PCC| > 0.95$.
- **Removed NSL-KDD Features:** The 6 features (`srv_serror_rate`, `dst_host_srv_rerror_rate`, `num_root`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `srv_rerror_rate`) are confirmed by correlation analysis.
- **Split Proportions:** Stratified 60% Train, 15% Validation, 25% Test.
- **DNN Parameters:** 3 hidden layers of 64 units, ReLU activations, 5-class Softmax, Adam, $lr=0.001$, weight decay = $0.0001$, 20 epochs.
- **2D-CNN Specifications:** 3 Conv2D layers (64 $\to$ 32 $\to$ 32 filters), $3 \times 3$ kernels, $2 \times 2$ Max Pooling, Softmax output.
- **SHAP Sample Size:** 50 test samples for global summary beeswarm and mean absolute SHAP feature ranking.

---

## 3. What Does Not Match the Paper (Must Be Fixed)
1. **NSL-KDD Dataset Source Pool:**
   - *Previous implementation:* Concatenated `KDDTrain+.txt` (125,973) + `KDDTest+.txt` (22,544) into 148,517 rows before splitting.
   - *Paper methodology:* The paper's primary experiment is based on the **`KDDTrain+.txt` dataset alone** (125,973 records) partitioned into 60/15/25. Concatenating test data was an over-inclusive assumption.
2. **UNSW-NB15 Class Imbalance & 50K Capping:**
   - *Previous implementation:* Filtered all records to 5 classes without capping, leaving Generic = 58,871 and Normal = 93,000 (total = 236,995).
   - *Paper methodology:* Section 3.1 explicitly caps the two dominant classes: **Generic $\le$ 50,000** and **Normal $\le$ 50,000**, while preserving all DoS (16,353), Exploits (44,525), and Fuzzers (24,246). The resulting dataset contains **185,124 records**.
3. **UNSW-NB15 Feature Selection & "loss" Ambiguity:**
   - *Previous implementation:* Dropped both `sloss` and `dloss` as matching `loss`.
   - *Paper methodology & schema consistency:* The raw CSVs contain `sloss` and `dloss`, not `loss`. Retaining `sloss` and `dloss` while dropping the 4 real redundant predictors (`ct_src_dport_ltm`, `dwin`, `ct_ftp_cmd`, `ct_srv_dst`) leaves **38 real predictors**, which requires **11 zeros of padding** to achieve $7 \times 7 = 49$, matching the paper's explicit 38-feature/11-padding representation.
4. **NSL-KDD 36th Feature:**
   - *Previous implementation:* Dropped difficulty, leaving 35 features, then added 1 zero of padding.
   - *Paper methodology:* Retaining `difficulty` as an input field yields $42 - 6 = 36$ features naturally, forming a $6 \times 6 = 36$ grid with 0 padding. We must support difficulty retention as the primary paper-faithful setup.

---

## 4. What Is Incomplete
- **Sampling Documentation:** Missing `results/data_sampling/unsw_sampling_report.json` tracking the exact seed and counts of the 50K sampling.
- **Correlation Matrix Outputs:** Missing `results/feature_selection/nsl_kdd_correlation_matrix.csv`.
- **Target Leakage Unit Tests:** Missing dedicated `tests/test_no_leakage.py` verifying programmatic assertions before every run.
- **Unified CLI Entrypoints:** No top-level `run_experiment.py`, `run_all.py`, or `run_xai.py` adhering to the specified CLI flag structure.
- **Formal 1D-CNN Reconstruction Document:** Missing `1dcnn_architecture_reconstruction.md`.

---

## 5. What Is Duplicated (Consolidation Target)
- **2D-CNN Duplication:**
  - `results/2d_cnn/nsl_kdd_2dcnn.py` and `results/2d_cnn/unsw_nb15_2dcnn.py` contain custom standalone implementations.
  - `models/cnn2d.py` contains a separate Keras functional definition.
  - *Action:* Consolidate into a single canonical `models/cnn2d.py` and `models/feature_to_grid.py`. All experiments, scripts, and Colab cells must import from `models/`.

---

## 6. What Must Be Changed
1. Refactor `preprocessing/nsl_kdd.py` to ingest `KDDTrain+.txt` only (125,973 records).
2. Refactor `preprocessing/unsw_nb15.py` to enforce the 50,000 cap on Generic and Normal (185,124 total records).
3. Update UNSW feature selection to retain `sloss`/`dloss`, dropping `ct_src_dport_ltm`, `dwin`, `ct_ftp_cmd`, `ct_srv_dst`, yielding 38 features + 11 padding zeros.
4. Unify model definitions into `models/dnn.py`, `models/cnn1d.py`, `models/cnn2d.py`, and `models/feature_to_grid.py`.
5. Provide top-level CLI runners: `run_experiment.py`, `run_all.py`, `run_xai.py`.
6. Add automated unit test suite in `tests/`.

---

## 7. What Can Be Retained
- `data/download_datasets.py`: Verified and fully working.
- `preprocessing/normalization.py`: Correct Min-Max implementation.
- `evaluation/metrics.py`: Clean multi-class evaluation routines.
- `evaluation/confusion_matrix.py`: Canonical 5-class heatmaps.
- `evaluation/timing.py`: High-resolution profiling.
- `explainability/lime_explainer.py` & `explainability/shap_explainer.py`: Core XAI wrappers.

---

## 8. What Is Inferred
- **1D-CNN Filter Topology:** Filter counts (64 $\to$ 32) and layer depth (2 conv layers) are inferred; paper only stated kernel=3, pool=2, ReLU.
- **Batch Size:** Configured as 64; unstated in paper.
- **Spatial Feature Ordering:** Row-major sequential feature-to-grid mapping; coordinate ordering unstated in paper.
- **Random Seeds:** Fixed seed 42 used across all partitioning and initialization.

---

## 9. What Is Paper-Faithful (Primary Replication)
- Primary 6 experiments:
  1. NSL-KDD Selected Features (36 features from KDDTrain+) $\to$ DNN
  2. NSL-KDD Selected Features $\to$ 1D-CNN
  3. NSL-KDD Selected Features $\to$ 2D-CNN ($6 \times 6 \times 1$)
  4. UNSW-NB15 Selected Features (38 features from 185k sample) $\to$ DNN
  5. UNSW-NB15 Selected Features $\to$ 1D-CNN
  6. UNSW-NB15 Selected Features $\to$ 2D-CNN ($7 \times 7 \times 1$, 11 zeros padding)
- Primary XAI: DNN on Selected Features (LIME on attack/normal cases; SHAP on 50 test samples).

---

## 10. What Is An Extension / Ablation
- **All-Feature Baseline (6 additional experiments):** Same splits, models, and hyperparameters without Pearson feature removal (NSL-KDD: 42 features; UNSW-NB15: 42 features).
- **Dropout Sensitivity Analysis:** DNN with dropout = 0.00 vs dropout = 0.01.
- **Leakage-Safe Sensitivity Pipeline:** Fitting encoders/scalers strictly on the 60% training split.
