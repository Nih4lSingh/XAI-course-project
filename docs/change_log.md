# Replication Change Log & Audit History

## Sharma et al. (2024) — Complete Audit, Correction, and Finalization

This document records all methodological audits, codebase corrections, architectural reconstructions, empirical executions, and verification procedures implemented to achieve an academically defensible replication of:

> **Bhawana Sharma, Lokesh Sharma, Chhagan Lal, Satyabrata Roy (2024)**  
> *“Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach”*  
> *Expert Systems with Applications*, 238, 121751.

---

### 1. Repository Inventory & Quarantine of Legacy Duplicate Scripts
- **Audit Finding**: Standalone duplicate directories `results/1d_cnn/` and `results/2d_cnn/` contained hardcoded scripts, conflicting data split logic (80/20 train/test vs. paper's 60/15/25), hardcoded padding assumptions, and outdated metrics.
- **Action**: Quarantined legacy code into `archive/legacy_1d_cnn/` and `archive/legacy_2d_cnn/`.
- **Standardization**: Created unified modules in `src/` and top-level packages: `preprocessing/`, `models/`, `training/`, `evaluation/`, `explainability/`, and `feature_selection/`.

---

### 2. Feature Count & Dimension Corrections
- **NSL-KDD Selected**:
  - Raw predictors = 42 (41 network attributes + `difficulty_level`, excluding `label`).
  - Dropped 6 redundant predictors (|PCC| > 0.95): `srv_serror_rate`, `dst_host_srv_rerror_rate`, `num_root`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `srv_rerror_rate`.
  - Canonical selected feature count: **36 features** $\to 6 \times 6$ grid (0 padding values).
- **UNSW-NB15 Selected**:
  - Raw predictors = 42 (excluding `id`, `attack_cat`, `label`).
  - Dropped 4 redundant predictors (|PCC| > 0.95): `ct_src_dport_ltm`, `dwin`, `ct_ftp_cmd`, `ct_srv_dst`.
  - Retained `sloss` and `dloss` (resolving the paper's ambiguous `loss` notation).
  - Canonical selected feature count: **38 features** $\to 7 \times 7$ grid with **exactly 11 trailing zeros padding**.
  - **Eliminated Inconsistency**: Any prior stale references or model artifacts with 36 features for UNSW Selected were audited, purged, retrained, and verified.

---

### 3. Sampling Policy & Split Disjointness
- **NSL-KDD**: Default ingestion of `KDDTrain+.txt` (125,973 records).
- **UNSW-NB15**: Capping of dominant classes at 50,000 records (`Generic` capped from 58,871 $\to$ 50,000; `Normal` capped from 93,000 $\to$ 50,000; all 16,353 `DoS`, 44,525 `Exploits`, and 24,246 `Fuzzers` retained). Total dataset = **185,124 records**.
- **Split Protocol**: Exact 60% Train, 15% Validation, 25% Test split using stratified sampling (random_state=42). Zero index overlap verified across splits.

---

### 4. Architectural Reconstruction
- **DNN**: Fully reconstructed matching paper Section 4.1 specifications: 3 Dense Hidden Layers with 64 units each and ReLU activation, Output Layer (5 units, Softmax): Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(5, Softmax). Evaluated with canonical Table 1 `dropout=0.0` and sensitivity `dropout=0.01`. All legacy 128 $\to$ 64 $\to$ 32 references were audited and purged.
- **2D-CNN**: Reconstructed matching paper Figure 5 topology: 3 Conv2D layers and 3 MaxPooling2D layers: Conv2D(64, 3x3) $\to$ MaxPool2D(2x2) $\to$ Conv2D(32, 3x3) $\to$ MaxPool2D(2x2) $\to$ Conv2D(32, 3x3) $\to$ MaxPool2D(2x2) $\to$ Flatten $\to$ Dense(5, Softmax). Configured with deterministic `padding='same'` on all convolutions and pooling layers to prevent spatial collapse on small $6 \times 6$ and $7 \times 7$ grids without dropping layers.
- **1D-CNN**: Reconstructed matching the inferred 1D convolutional filter progression: Conv1D(64, kernel=3, padding='same', ReLU) $\to$ MaxPool1D(2) $\to$ Conv1D(32, kernel=3, padding='same', ReLU) $\to$ Flatten $\to$ Dense(5, Softmax). Filter counts 64 $\to$ 32 are explicitly documented as an `INFERRED PARAMETER`.

---

### 5. Leakage-Safe Sensitivity Pipeline (Mode B)
- Implemented `preprocessing/leakage_safe.py` to contrast paper-faithful global preprocessing (Mode A) with a modern leakage-free pipeline (Mode B):
  - Stratified Train/Val/Test split executed *prior* to fitting any encoders or scalers.
  - Categorical label encoders fitted strictly on training partition.
  - Min-Max scaling parameters ($F_{\min}, F_{\max}$) fitted strictly on training partition.
  - Pearson correlation matrix and thresholding computed strictly on training partition.
  - Verified that train-only Pearson correlation on NSL-KDD extracts the *identical 6 redundant features* as the global dataset.

---

### 6. Explainability (XAI) Alignment
- **SHAP Alignment**:
  - NSL-KDD Global SHAP targets **`DoS`** (Class index 0 in canonical mapping). Top predictors: `serror_rate`, `logged_in`, `dst_host_same_src_port_rate`.
  - UNSW-NB15 Global SHAP targets **`Normal`** (Class index 4 in canonical mapping). Top predictors: `dttl`, `swin`, `ct_srv_src`, `sttl`.
  - Dynamic class name resolution implemented to prevent arbitrary hardcoded class targeting.
- **LIME Alignment**:
  - Explanations generated for canonical attack and normal instances across both datasets.

---

### 7. Results Directory Hierarchy & Machine-Readable Artifacts
- Organized results into:
  - `results/canonical/<experiment_id>/` (the 6 primary models)
  - `results/ablations/<experiment_id>/` (the 6 All-features models)
  - `results/sensitivity/<experiment_id>/` (dropout=0.01 and leakage-safe preprocessing)
- Every experiment directory includes: `metrics.json`, `timing.json`, `reproducibility.json`, `cm.png`, `accuracy.png`, `loss.png`, `y_pred.npy`, `y_prob.npy`, and `model.keras`.
- Summary tables generated programmatically from JSON metrics via `scripts/generate_tables.py`:
  - `results/results_summary.csv`
  - `results/paper_vs_reproduction.csv`
  - `reports/tables/results_table.md`

---

### 8. Automated Test Suite & Validation
- Implemented complete test suite (26 passing tests):
  - `tests/test_no_leakage.py`: Zero target column in features, split mutual exclusivity and completeness, normalization bounds [0, 1].
  - `tests/test_pipeline.py`: NSL 36 features, UNSW 38 features, 11 padding zeros, 50k sampling report, class index mappings.
  - `tests/test_feature_selection.py`: 36 and 38 canonical counts, removed feature sets matching paper exactly.
  - `tests/test_models.py`: 5-class softmax output dimensions across DNN, 1D-CNN, 2D-CNN.
  - `tests/test_metrics_and_summary.py`: Metric ranges, CSV-to-JSON alignment, no stale UNSW 36 features.
  - `tests/test_xai_integrity.py`: SHAP ranking completeness (36 for NSL, 38 for UNSW), class target alignment (DoS=0, Normal=4).
- Implemented `scripts/validate_results.py` for automated pre-submission consistency auditing.
