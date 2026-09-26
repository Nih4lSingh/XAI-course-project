# Paper Extraction: Sharma et al. (2024)

**Paper Title:** Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach  
**Authors:** Bhawana Sharma, Lokesh Sharma, Chhagan Lal, Satyabrata Roy  
**Journal:** *Expert Systems with Applications*, Volume 238, March 2024, Article 121751  
**DOI:** [10.1016/j.eswa.2023.121751](https://doi.org/10.1016/j.eswa.2023.121751)  
**Primary Source:** Published Article (ESWA, Elsevier)

---

## 1. Methodological Extraction Table

Every parameter is cataloged below with its exact source location and formal classification:
- **`EXPLICIT`**: Directly specified in text, tables, or figures.
- **`INFERRED`**: Required for implementation but unstated; derived from standard DL practice or structural constraints.
- **`AMBIGUOUS`**: Contradictory statements or conflicting values within the paper text/figures.
- **`NOT REPORTED`**: Completely omitted from the paper.
- **`OUR EXTENSION`**: Our independent ablation/extension (e.g. all-feature baseline, leakage-safe pipeline).

| Component | Paper says | Exact source | Status |
| :--- | :--- | :--- | :--- |
| **NSL-KDD dataset** | KDDTrain+ benchmark dataset (125,973 records); transformed into 5 canonical classes: DoS, Normal, Probe, R2L, U2R. Difficulty field retained to yield 36 features after correlation filtering. | Section 3.1, Section 4.2.1 | `EXPLICIT` |
| **UNSW-NB15 dataset** | 5-class subset (DoS, Exploits, Fuzzers, Generic, Normal). Majority classes capped: Generic $\le$ 50,000, Normal $\le$ 50,000; all DoS, Exploits, Fuzzers retained. Total $\approx$ 185,124 records. | Section 3.1, Section 4.3.1 | `EXPLICIT` |
| **Label processing** | Categorical label encoding converting string attributes to integer labels [0, K-1]. Target class order: 0=DoS, 1=Normal, 2=Probe, 3=R2L, 4=U2R (NSL); 0=DoS, 1=Exploits, 2=Fuzzers, 3=Generic, 4=Normal (UNSW). | Section 3.2, Section 5 | `EXPLICIT` |
| **Normalization** | Min-max normalization: $F_{new} = \frac{F - F_{min}}{F_{max} - F_{min}} \in [0, 1]$. | Section 3.2, Eq. (1) | `EXPLICIT` |
| **Feature selection** | Filter-based method using Pearson Correlation Coefficient (PCC). NSL-KDD removes 6 features: `srv_serror_rate`, `dst_host_srv_rerror_rate`, `num_root`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `srv_rerror_rate`. UNSW-NB15 removes `ct_src_dport_ltm`, `dwin`, `ct_ftp_cmd`, `ct_srv_dst` (while separating target `label`). | Section 3.2 | `EXPLICIT` (NSL) / `AMBIGUOUS` (UNSW "loss") |
| **Feature threshold** | Redundant pairs identified when $|PCC| > 0.95$. | Section 3.2, Eq. (2) | `EXPLICIT` |
| **Data split** | 60% Training, 15% Validation, 25% Testing ($0.60 + 0.15 + 0.25 = 1.00$). | Section 3.3 | `EXPLICIT` |
| **DNN architecture** | 3 hidden Dense layers of 64 neurons each with ReLU activation; Output Dense(5) with Softmax activation. | Section 4.1, Table 1 | `EXPLICIT` |
| **1D-CNN architecture** | Kernel size = 3, Max Pooling size = 2, Activation = ReLU, Output Dense(5) with Softmax. Filter counts not stated. | Section 4.2 | `EXPLICIT` (Kernel/Pool) / `INFERRED` (Filters) |
| **2D-CNN architecture** | 3 Conv2D layers: Filters 64 $\to$ 32 $\to$ 32, Kernel size $3 \times 3$, MaxPool $2 \times 2$, Output Dense(5) Softmax. Input: $6 \times 6 \times 1$ (NSL) and $7 \times 7 \times 1$ (UNSW with 11 zeros padding). | Section 4.3 | `EXPLICIT` |
| **Optimizer** | Adam optimizer. | Section 4.1, Table 1 | `EXPLICIT` |
| **Learning rate** | 0.001 | Section 4.1, Table 1 | `EXPLICIT` |
| **Weight decay** | 0.0001 ($10^{-4}$ L2 regularization) | Section 4.1, Table 1 | `EXPLICIT` |
| **Epochs** | 20 epochs | Section 4.1, Table 1 | `EXPLICIT` |
| **Dropout** | Dropout = 0 (Table 1) vs Dropout = 0.01 (Section 4.1 text: "dropout rate of 0.01 is used"). | Table 1 vs Sec 4.1 | `AMBIGUOUS` |
| **LIME** | Local Interpretable Model-agnostic Explanations applied to representative individual test instances of DNN predictions. | Section 5.1 | `EXPLICIT` |
| **SHAP** | Evaluated on DNN; 50 testing samples used for global summary beeswarm and mean absolute SHAP importance ($mean(\|SHAP\|)$); local force/waterfall explanation for individual instances. | Section 5.2 | `EXPLICIT` |

---

## 2. Inferred Parameters Log

| Parameter | Inferred Value | Justification |
| :--- | :---: | :--- |
| **Batch Size** | 64 | Standard Keras/DL benchmark size; stable convergence over 20 epochs. |
| **1D-CNN Filter Count** | Conv1D(64, 3) $\to$ Conv1D(32, 3) | Mirrors the 2D-CNN filter progression (64 $\to$ 32) matching reported parameter scale. |
| **2D-CNN Grid Ordering** | Row-major deterministic mapping | Standard coordinate allocation $[r, c] = [idx // W, idx \% W]$. |
| **Random Seed** | 42 | Standard seed ensuring exact cross-platform reproducibility. |
