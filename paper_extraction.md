# Paper Extraction: Sharma et al. (2024)

**Paper Title:** Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach  
**Authors:** Bhawana Sharma, Lokesh Sharma, Chhagan Lal, Satyabrata Roy  
**Journal:** *Expert Systems with Applications*, Volume 238, March 2024, Article 121751  
**DOI:** [10.1016/j.eswa.2023.121751](https://doi.org/10.1016/j.eswa.2023.121751)  
**Primary Source:** Published Article (ESWA, Elsevier)

---

## 1. Methodological Extraction Table

Every parameter in this study is cataloged below and classified according to its status in the published paper:
- **`EXPLICIT`**: Directly specified in the text, tables, or figures.
- **`INFERRED`**: Required for implementation but unstated; derived from standard Keras/DL practice or structural constraints.
- **`AMBIGUOUS`**: Contradictory statements or conflicting values within the paper text/figures.
- **`NOT REPORTED`**: Completely omitted from the paper.
- **`ADDITIONAL EXPERIMENT`**: Our independent ablation/extension (e.g. all-feature baseline, leakage-safe pipeline).

| Category | Item | Paper Specification | Source Location | Status | Implementation Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Datasets** | NSL-KDD | Public benchmark dataset for network IDS; transformed into `NSL-KDDnew` | Section 3.1 | `EXPLICIT` | Full benchmark dataset used |
| **Datasets** | UNSW-NB15 | Public benchmark dataset for network IDS; transformed into `UNSW-NBnew` | Section 3.1 | `EXPLICIT` | Full benchmark dataset used |
| **Classification Setup** | NSL-KDD Target Classes | 5 classes: DoS (0), Normal (1), Probe (2), R2L (3), U2R (4) | Section 3.1, Section 5 (XAI) | `EXPLICIT` | Categorical attack types grouped into 5 classes with exact integer mapping |
| **Classification Setup** | UNSW-NB15 Target Classes | 5 classes: DoS (0), Exploits (1), Fuzzers (2), Generic (3), Normal (4) | Section 3.1, Section 5 (XAI) | `EXPLICIT` | Filtered to 5 classes with exact integer mapping (0: DoS, 1: Exploits, 2: Fuzzers, 3: Generic, 4: Normal) |
| **Preprocessing** | Categorical Encoding | Label encoding (integer mapping) followed by min-max normalization | Section 3.2 | `EXPLICIT` | LabelEncoder applied to non-numeric columns (`protocol_type`, `service`, `flag` for NSL; `proto`, `service`, `state` for UNSW) |
| **Preprocessing** | Normalization | Min-max normalization: $F_{new} = \frac{F - F_{min}}{F_{max} - F_{min}} \in [0, 1]$ | Section 3.2, Eq. (1) | `EXPLICIT` | MinMaxScaler applied per feature column |
| **Preprocessing** | Train/Val/Test Split | 60% Training, 15% Validation, 25% Testing (75% train split further into 60/15) | Section 3.3 | `EXPLICIT` | Random split: 75% train-val, 25% test; then train-val split into 80% train (60% total) and 20% val (15% total) |
| **Feature Selection** | Selection Method | Filter method using Pearson Correlation Coefficient (PCC) | Section 3.2, Eq. (2) | `EXPLICIT` | Compute pairwise Pearson correlation matrix |
| **Feature Selection** | Threshold | $|PCC| > 0.95$; one feature removed from highly correlated pairs | Section 3.2 | `EXPLICIT` | Pairs with correlation magnitude $> 0.95$ identified; redundant features removed |
| **Feature Selection** | NSL-KDD Removed Features | 6 features: `srv_serror_rate`, `dst_host_srv_rerror_rate`, `num_root`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `srv_rerror_rate` | Section 3.2 | `EXPLICIT` | Exactly these 6 redundant features removed |
| **Feature Selection** | NSL-KDD Feature Count | Reports 36 features after removal | Section 3.2, Section 4 | `AMBIGUOUS` | Standard NSL-KDD has 41 predictors ($41 - 6 = 35$). We verify if `difficulty_level` was included or 1 padding feature added to reach 36 for $6 \times 6$ grid |
| **Feature Selection** | UNSW-NB15 Removed Features | 6 features: `ct_src_dport_ltm`, `loss`, `dwin`, `ct_ftp_cmd`, `label`, `ct_srv_dst` | Section 3.2 | `AMBIGUOUS` | Note: `label` is the binary target column. Strict separation applied so target is never in $X$. Documented in `unsw_feature_dimension_discrepancy.md` |
| **Feature Selection** | UNSW-NB15 Input Dimension | Zero-padded to $7 \times 7 = 49$ for 2D-CNN | Section 4.3 | `EXPLICIT` | Input feature vector padded with zeros up to 49 elements |
| **Models: DNN** | Architecture | 3 hidden Dense layers (64 $\to$ 64 $\to$ 64) with ReLU; Output Dense(5) with Softmax | Section 4.1 | `EXPLICIT` | Input $\to$ Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(5, Softmax) |
| **Models: DNN** | Optimizer & LR | Adam optimizer, Learning rate = 0.001 | Section 4.1, Table 1 | `EXPLICIT` | Adam(learning_rate=0.001) |
| **Models: DNN** | Weight Decay | 0.0001 (L2 regularization / weight decay) | Section 4.1, Table 1 | `EXPLICIT` | $10^{-4}$ weight decay |
| **Models: DNN** | Epochs | 20 epochs | Section 4.1, Table 1 | `EXPLICIT` | Trained for exactly 20 epochs |
| **Models: DNN** | Loss Function | Sparse Categorical Cross-Entropy | Section 4.1 | `EXPLICIT` | `sparse_categorical_crossentropy` |
| **Models: DNN** | Dropout Rate | Dropout = 0 (Table 1) vs Dropout = 0.01 (text discussion) | Table 1 vs Sec 4.1 text | `AMBIGUOUS` | Configurable parameter: evaluated with 0.00 (primary) and 0.01 (sensitivity analysis) |
| **Models: 1D-CNN** | Kernel Size | Kernel size = 3 | Section 4.2 | `EXPLICIT` | `kernel_size=3` |
| **Models: 1D-CNN** | Pooling Size | Max Pooling 1D with pool size = 2 | Section 4.2 | `EXPLICIT` | `pool_size=2` |
| **Models: 1D-CNN** | Activation | ReLU activation | Section 4.2 | `EXPLICIT` | `activation='relu'` |
| **Models: 1D-CNN** | Filters & Layer Count | Not explicitly reported in paper text | Section 4.2 | `INFERRED` | Standard architecture: Conv1D(64, 3) $\to$ MaxPool1D(2) $\to$ Conv1D(32, 3) $\to$ Flatten $\to$ Dense(5, Softmax) |
| **Models: 1D-CNN** | Training Hyperparams | Adam, lr=0.001, decay=0.0001, epochs=20 | Section 4.2 | `EXPLICIT` | Same optimizer, lr, decay, and epochs as DNN |
| **Models: 2D-CNN** | Architecture | 3 Conv2D layers: Filters 64 $\to$ 32 $\to$ 32; Kernel 3 $\times$ 3; MaxPool2D(2, 2); Dense(5, Softmax) | Section 4.3 | `EXPLICIT` | Conv2D(64, (3,3)) $\to$ MaxPool2D((2,2)) $\to$ Conv2D(32, (3,3)) $\to$ Conv2D(32, (3,3)) $\to$ Flatten $\to$ Dense(5, Softmax) |
| **Models: 2D-CNN** | Input Reshape | NSL-KDD: $6 \times 6 \times 1$; UNSW-NB15: $7 \times 7 \times 1$ (zero-padded) | Section 4.3 | `EXPLICIT` | Reshape 1D vector into 2D grid deterministically |
| **Models: 2D-CNN** | Padding / Stride | Stride=1, padding='same' to preserve spatial grid across 3 conv layers | Section 4.3 | `INFERRED` | `padding='same'` ensures tensor does not collapse below $1 \times 1$ during pooling |
| **Models: 2D-CNN** | Training Hyperparams | Adam, lr=0.001, decay=0.0001, epochs=20 | Section 4.3 | `EXPLICIT` | Same optimizer, lr, decay, and epochs |
| **General Training** | Batch Size | Not explicitly stated in text | Section 4 | `INFERRED` | Default Keras batch size 64 used (configurable: 32 / 64) |
| **General Training** | Random Seed | Seed number not specified in paper | Section 4 | `INFERRED` | Fixed random seed = 42 for NumPy, Python, and TensorFlow; split indices saved |
| **General Training** | Early Stopping | Not mentioned in paper | Section 4 | `EXPLICIT` | Fixed 20 epochs; no early stopping applied |
| **General Training** | Class Balancing | No SMOTE, oversampling, or class weighting reported | Section 4 | `EXPLICIT` | Natural class distribution preserved |
| **Explainability (XAI)** | Target Model | DNN model exclusively | Section 5 | `EXPLICIT` | Primary XAI focuses on the trained DNN model |
| **Explainability (XAI)** | LIME | Local explanations for representative test instances (attack vs normal) | Section 5.1 | `EXPLICIT` | LIME TabularExplainer applied to test samples from both datasets |
| **Explainability (XAI)** | SHAP Global | 50 test samples used for global summary and mean absolute SHAP importance | Section 5.2 | `EXPLICIT` | Exact 50 test samples evaluated for summary beeswarm and feature ranking |
| **Explainability (XAI)** | SHAP Local | Force / waterfall plot for individual instance prediction | Section 5.2 | `EXPLICIT` | Local force/waterfall explanation for individual predictions |
| **Ablation Baseline** | All-Feature Baseline | All eligible predictor features retained (no PCC feature selection removal) | Independent addition | `ADDITIONAL EXPERIMENT` | 6 additional experiments (DNN, 1D-CNN, 2D-CNN on NSL-KDD and UNSW-NB15) |
| **Methodological Rigor** | Leakage-Safe Sensitivity | Fitting scaler and feature selection on training split only | Independent addition | `ADDITIONAL EXPERIMENT` | Evaluated separately to assess potential data leakage in whole-dataset pre-split filtering |

---

## 2. Quantitative Paper Reference Benchmarks

These numbers serve strictly as reference comparison points and must never be forced:

### A. Classification Performance
| Dataset | Model | Reported Accuracy | Reported Precision | Reported Recall | Reported F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NSL-KDDnew** | DNN | 99.3% | 0.99 | 0.99 | 0.99 |
| **NSL-KDDnew** | 1D-CNN | 99.2% | 0.99 | 0.99 | 0.99 |
| **NSL-KDDnew** | 2D-CNN | 99.4% | 0.99 | 0.99 | 0.99 |
| **UNSW-NBnew** | DNN | 80.0% | 0.80 | 0.80 | 0.80 |
| **UNSW-NBnew** | 1D-CNN | 80.0% | 0.80 | 0.80 | 0.80 |
| **UNSW-NBnew** | 2D-CNN | 81.0% | 0.81 | 0.81 | 0.81 |

### B. Reported Training Time (Reference Hardware)
| Dataset | Model | Reported Training Time (Paper) | Note |
| :--- | :--- | :---: | :--- |
| **NSL-KDDnew** | DNN | $\approx 142$ ms | Reported per-epoch or total (unit context documented) |
| **NSL-KDDnew** | 1D-CNN | $\approx 325$ ms | Hardware dependent |
| **NSL-KDDnew** | 2D-CNN | $\approx 340$ ms | Hardware dependent |
| **UNSW-NBnew** | DNN | $\approx 323$ ms | Hardware dependent |
| **UNSW-NBnew** | 1D-CNN | $\approx 442$ ms | Hardware dependent |
| **UNSW-NBnew** | 2D-CNN | $\approx 455$ ms | Hardware dependent |

---

## 3. Top Influential XAI Features Reported in Paper

### NSL-KDD Influential Features
- `same_srv_rate`
- `dst_host_same_srv_rate`
- `dst_host_srv_count`
- `flag`
- `diff_srv_rate`
- `srv_count`
- `hot`
- `dst_host_count`
- `serror_rate`

### UNSW-NB15 Influential Features
- `dttl`
- `state`
- `spkts`
- `proto`
- `smean`
- `ct_srv_src`
- `sttl`
- `swin`
- `ackdat`
- `ct_src_ltm`
