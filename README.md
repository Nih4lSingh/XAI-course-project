# Replication Study: Explainable Artificial Intelligence for Intrusion Detection in IoT Networks

## Reference Publication
> **Bhawana Sharma, Lokesh Sharma, Chhagan Lal, Satyabrata Roy (2024)**  
> *“Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach”*  
> *Expert Systems with Applications*, Volume 238, March 2024, Article 121751.  
> [DOI: 10.1016/j.eswa.2023.121751](https://doi.org/10.1016/j.eswa.2023.121751)

---

## 1. Project Purpose
This repository provides an open, audited, academically defensible replication of the deep-learning and Explainable AI (XAI) intrusion detection framework proposed by Sharma et al. (2024). The project implements the published pipeline across two canonical network intrusion datasets (NSL-KDD and UNSW-NB15), evaluates the three reported deep architectures (DNN, 1D-CNN, and 2D-CNN), and provides global (SHAP) post-hoc explanations.

---

## 2. Reference Paper Summary
Sharma et al. proposed an end-to-end intrusion detection system for IoT networks:
- **Datasets**: NSL-KDD (`NSL-KDDnew`, 5 classes) and UNSW-NB15 (`UNSW-NBnew`, 5 classes).
- **Preprocessing**: Deterministic categorical label encoding and Min-Max normalization to $[0, 1]$.
- **Feature Selection**: Pearson Correlation Coefficient ($|PCC| > 0.95$) filter to remove collinear traffic descriptors.
- **Deep Architectures**: 3-layer Deep Neural Network (DNN), 1D Convolutional Neural Network (1D-CNN), and 2D Convolutional Neural Network (2D-CNN) with image-like spatial mapping.
- **Interpretability**: Global and local feature importance analysis using SHAP (KernelExplainer over 50 test instances).

---

## 3. Research Questions
Our replication and extension evaluate one core research question:
- **RQ1 (Reproducibility)**: Can the paper's reported classification accuracies (99.2%–99.4% on NSL-KDD, 80.0%–81.0% on UNSW-NB15) be reproduced under identical splits and hyperparameter specifications?

---

## 4. Datasets & Sampling Logic

### Canonical Feature Counts and Grid Dimensions
- **NSL-KDD Selected Features**: **36 features** $\to 6 \times 6$ grid (0 padding elements).
- **UNSW-NB15 Selected Features**: **38 features** $\to 7 \times 7$ grid (**exactly 11 trailing zeros padding** to 49).

### Partitioning & Sampling Policies
- **NSL-KDD (`NSL-KDDnew`)**: Default ingestion uses `KDDTrain+.txt` (125,973 records) partitioned into 5 classes:
  - `DoS` (0): 45,927 | `Normal` (1): 67,343 | `Probe` (2): 11,656 | `R2L` (3): 995 | `U2R` (4): 52.
- **UNSW-NB15 (`UNSW-NBnew`)**: 5-class subset with dominant class capping at 50,000 records (Sharma et al. Section 3.1):
  - `Generic` (3): capped at 50,000 (sampled from 58,871; seed=42)
  - `Normal` (4): capped at 50,000 (sampled from 93,000; seed=42)
  - `DoS` (0): 16,353 | `Exploits` (1): 44,525 | `Fuzzers` (2): 24,246
  - **Total Records**: **185,124 records**.
- **Split Protocol**: Exact 60% Train (111,074), 15% Validation (27,769), 25% Test (46,281) with stratified sampling. All splits are mutually exclusive (zero index overlap verified).

---

## 5. Preprocessing Pipeline

Implements the exact procedural sequence described in the text of Sharma et al. (2024):
1. Load dataset (125,973 NSL-KDD records; 185,124 capped UNSW-NB15 records).
2. Deterministic label encoding on categorical variables (`protocol_type`, `service`, `flag` for NSL; `proto`, `service`, `state` for UNSW).
3. Min-Max normalization to $[0, 1]$ applied across the full dataset.
4. Stratified 60% / 15% / 25% partitioning.

---

## 6. Feature-Selection Methodology
Collinear redundancy filtering with threshold $|PCC| > 0.95$:
- **NSL-KDD (42 raw predictors $\to$ 36 selected)**:
  - 6 redundant predictors dropped: `srv_serror_rate`, `dst_host_srv_rerror_rate`, `num_root`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `srv_rerror_rate`.
  - Retaining `difficulty_level` yields exactly 36 features ($6 \times 6$ grid, 0 padding).
- **UNSW-NB15 (42 raw predictors $\to$ 38 selected)**:
  - 4 redundant traffic predictors dropped: `ct_src_dport_ltm`, `dwin`, `ct_ftp_cmd`, `ct_srv_dst`.
  - Retaining `sloss` and `dloss` yields exactly 38 selected features (padded with 11 zeros to 49 for $7 \times 7$ grid).
  - Target variables `id`, `attack_cat`, and `label` are strictly isolated from the predictor matrix.

---

## 7. Model Architectures
- **DNN**: Input Layer (36 or 38 units) $\to$ Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(5, Softmax). In strict accordance with Section 4.1.
- **1D-CNN**: Input Layer $(D, 1)$ $\to$ Conv1D(64, kernel=3, ReLU, same padding) $\to$ MaxPool1D(2) $\to$ Conv1D(32, kernel=3, ReLU, same padding) $\to$ Flatten $\to$ Dense(5, Softmax). Filter counts 64 $\to$ 32 are an inferred parameter aligning with 2D-CNN filter scale.
- **2D-CNN (Paper Fig. 5 Topology)**: Input Layer $(H, W, 1)$ $\to$ Conv2D(64, 3x3, ReLU, same padding) $\to$ MaxPool2D(2x2, same padding) $\to$ Conv2D(32, 3x3, ReLU, same padding) $\to$ MaxPool2D(2x2, same padding) $\to$ Conv2D(32, 3x3, ReLU, same padding) $\to$ MaxPool2D(2x2, same padding) $\to$ Flatten $\to$ Dense(5, Softmax). Implements exact 3-Conv / 3-Pool layer topology of Figure 5 with deterministic `padding='same'`.

---

## 8. Explainable AI (XAI) Methodology
- **SHAP (SHapley Additive exPlanations)**:
  - Uses `KernelExplainer` with 100 background samples and 50 representative test samples.
  - **NSL-KDD Global SHAP Target**: **`DoS`** (Class 0 in canonical mapping). Top predictors: `serror_rate`, `logged_in`, `dst_host_same_src_port_rate`.
  - **UNSW-NB15 Global SHAP Target**: **`Normal`** (Class 4 in canonical mapping). Top predictors: `dttl`, `swin`, `ct_srv_src`, `sttl`.
  - Generates global beeswarm summary plots and mean absolute attribution rankings ($mean(|SHAP|)$).

---

## 9. Experimental Protocol
- **Optimizer**: Adam ($\text{learning rate} = 0.001$, $\text{weight decay} = 0.0001$).
- **Loss Function**: Sparse Categorical Cross-Entropy.
- **Batch Size**: 128.
- **Epochs**: 20.
- **Hardware Platform**: NVIDIA Tesla T4 GPU (Google Colab) and Intel CPU verification.
- **Deterministic Controls**: Random seed 42 set across Python, NumPy, and TensorFlow.

---

## 10. Reproduction Instructions

### Environment Setup
```bash
# Clone repository
git clone https://github.com/Srr28/XAI-course-project.git
cd XAI-course-project

# Install dependencies
pip install -r requirements.txt
```

### Preprocessing & Feature Selection
```bash
# Preprocess NSL-KDD and UNSW-NB15
python -m preprocessing.nsl_kdd
python -m preprocessing.unsw_nb15

# Verify Pearson correlation feature selection
python -m feature_selection.nsl_kdd_features
python -m feature_selection.unsw_features
```

### Model Training & Evaluation
```bash
# Train single model (example: NSL-KDD Selected DNN)
python run_experiment.py --exp_id NSL_SELECTED_DNN --epochs 20 --batch_size 128

# Run the complete 6-model matrix
python run_all.py

# Run SHAP explainability pipeline
python run_xai.py
```

### Verification & Table Generation
```bash
# Run automated test suite (30 tests)
python -m unittest discover tests

# Validate results consistency
python scripts/validate_results.py

# Regenerate results summary tables
python scripts/generate_tables.py
```

---

## 11. Canonical Results (Paper Replication)

| Dataset | Model | Selected Feats | Our Accuracy | Paper Accuracy | Difference | Our Macro-F1 | Our Weighted-F1 | Paper Training Time (ms) | Our Total Training Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NSL-KDD** | DNN | 36 | **0.9966** | 0.9930 | +0.0036 | 0.8058 | 0.9965 | 142.0 | 18.77 |
| **NSL-KDD** | 1D-CNN | 36 | **0.9947** | 0.9920 | +0.0027 | 0.9017 | 0.9947 | 325.0 | 53.05 |
| **NSL-KDD** | 2D-CNN | 36 | **0.9962** | 0.9940 | +0.0022 | 0.8085 | 0.9961 | 340.0 | 103.90 |
| **UNSW-NB15** | DNN | 38 | **0.8052** | 0.8000 | +0.0052 | 0.6627 | 0.7754 | 323.0 | 26.46 |
| **UNSW-NB15** | 1D-CNN | 38 | **0.7995** | 0.8000 | -0.0005 | 0.6454 | 0.7639 | 442.0 | 75.45 |
| **UNSW-NB15** | 2D-CNN | 38 | **0.8079** | 0.8100 | -0.0021 | 0.6595 | 0.7755 | 455.0 | 166.65 |

*Note: Differences are reported as (Our Accuracy - Paper Accuracy). Paper training times (142/325/340 ms for NSL-KDD; 323/442/455 ms for UNSW-NB15) are labeled by the authors as training time. Our reproduction measures total 20-epoch wall-clock training time in seconds. The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons.*

---

## 12. Limitations
1. **Benchmark Age & Representativeness**: NSL-KDD derives from 1999 network traffic, lacking contemporary IoT protocol flows (e.g., MQTT, CoAP).
2. **2D-CNN Inductive Bias**: Reshaping tabular network attributes into a 2D matrix imposes artificial spatial adjacency between unrelated features that varies under row-major ordering.
3. **Severe Class Imbalance**: High global accuracy on NSL-KDD (99.7%) is dominated by majority classes (`Normal`, `DoS`), while `U2R` contains only 52 total instances in 125K records.
4. **Hardware-Dependent Runtime**: Paper-reported training times in milliseconds cannot be directly normalized against our total wall-clock training times executed in different environments.

---

## 13. Methodological Ambiguities in Sharma et al. (2024)
- **1D-CNN Architecture**: The paper did not specify kernel size, pooling dimensions, or filter counts for 1D-CNN. We reconstructed standard parameters ($\text{kernel}=3, \text{pool}=2$, filters $64 \to 32$) and documented the design choice as an inferred parameter.
- **Feature Count Reconstruction**: Retaining `difficulty_level` (NSL-KDD), isolating `label` to avoid leakage, and retaining `sloss`/`dloss` (UNSW-NB15) to achieve 36 and 38 selected features are project reconstruction decisions to reconcile canonical model dimensions and grids, not mechanical copies of paper procedures or arithmetic.
- **UNSW-NB15 SHAP 'data' Inconsistency**: The paper cites `data` as the #1 most important feature for UNSW Normal global SHAP, but `data` does not exist in the UNSW-NB15 dataset schema or paper feature table. Our reproduction faithfully reports the empirical top feature (`dttl`).
- **Non-Causal Disclaimer**: SHAP and LIME explain the model's learned prediction behavior; they do not establish causal relationships between a feature and the underlying network attack. Attribution values reflect how strongly input perturbations shift output activations within the learned decision boundaries.

---

## 14. Reproducibility Information
- **Random Seed**: 42 fixed across NumPy, Python, and TensorFlow.
- **Software Stack**: Python 3.10+, TensorFlow 2.17.0 / 2.22.0, scikit-learn 1.5.2, SHAP 0.46.0.
- **Reproducibility Metadata**: Machine-readable metadata saved in `reproducibility.json` within every experiment directory in `results/`.
- **Hardware Nondeterminism Note**: Exact floating-point bitwise parity cannot be guaranteed across distinct GPU architectures due to non-deterministic atomic operations in cuDNN.

---

## 15. Repository Structure
```text
XAI-course-project/
├── README.md                                  # Authoritative project documentation
├── requirements.txt                           # Python dependencies
├── environment.yml                            # Conda environment file
├── run_all.py                                 # Single entry-point script for 6 models
├── run_experiment.py                          # Single experiment runner
├── run_xai.py                                 # SHAP explainability runner
│
├── configs/                                   # Class mappings & experiment JSON configs
├── preprocessing/                             # Encoders, scalers, dataset pipelines
├── feature_selection/                         # Pearson correlation selector & grid mapping
├── models/                                    # DNN, 1D-CNN, 2D-CNN architectures
├── training/                                  # Training loop & experiment backend
├── evaluation/                                # 5-class metrics, confusion matrices, timing
├── explainability/                            # SHAP explainer backends
│
├── scripts/                                   # Automation & verification scripts
│   ├── organize_results.py                    # Results directory organizer
│   ├── generate_tables.py                     # Programmatic table generator from JSON
│   ├── validate_results.py                    # Pre-submission consistency auditor
│   └── validate_pipeline.py                   # Data integrity & leakage auditor
│
├── tests/                                     # Automated test suite (29 tests)
│   ├── test_no_leakage.py                     # Split disjointness & leakage prevention
│   ├── test_pipeline.py                       # Dimension, grid transform & mapping
│   ├── test_feature_selection.py              # Canonical 36 & 38 feature count checks
│   ├── test_models.py                         # Architecture & layer configuration checks
│   ├── test_metrics_and_summary.py            # JSON-to-CSV alignment & bounds
│   └── test_xai_integrity.py                  # SHAP feature alignment & target classes
│
├── data/                                      # Preprocessed dataset arrays (.npz)
├── splits/                                    # Exact 60/15/25 split indices (.npy)
├── docs/                                      # Research & audit documentation
│   ├── change_log.md                          # Detailed change & audit log
│   ├── AUDIT_REPORT.md                        # Initial codebase audit report
│   ├── paper_extraction.md                    # Systematic paper parameter extraction
│   ├── 1dcnn_architecture_reconstruction.md   # 1D-CNN architectural analysis
│   ├── replication_deviations.md              # Ambiguity resolution log
│   ├── unsw_feature_dimension_discrepancy.md  # UNSW feature analysis
│   └── xai_report.md                          # Comprehensive XAI analysis
│
├── reports/                                   # Final reports & markdown tables
│   ├── final_report.md                        # Master academic submission report
│   ├── tables/results_table.md                # Markdown summary tables
│   └── figures/                               # Confusion matrices & heatmaps
│
└── results/                                   # Structured experimental outputs
    ├── canonical/                             # 6 primary paper replication models
    ├── results_summary.csv                    # Complete master metrics summary
    ├── paper_vs_reproduction.csv              # Side-by-side paper comparison
    ├── models/                                # Saved .keras checkpoints
    ├── confusion_matrices/                    # High-res confusion matrix plots
    ├── training_curves/                       # Loss and accuracy curves
    └── xai/                                   # SHAP figures and metadata
```
