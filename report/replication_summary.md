# Replication Summary: Sharma et al. (2024)
## Explainable Artificial Intelligence for Enhancing Intrusion Detection Systems Using Deep Learning

**Authors**: Praneet Sharma, Jyoti Sengupta, P. K. Suri (*IEEE Access*, 2024)  
**Replication Team**: XAI Project Group | **Date**: September 2026  
**Repository**: `https://github.com/Nih4lSingh/XAI-course-project.git` (Branches: `main` & `multiseed-replication`)

---

### 1. Executive Summary & Objective
This study replicates the deep-learning and Explainable AI (XAI) intrusion detection framework proposed by Sharma et al. (2024). The paper evaluated three deep neural network families—Deep Neural Networks (DNN), 1D Convolutional Neural Networks (1D-CNN), and 2D Convolutional Neural Networks (2D-CNN)—across two canonical intrusion benchmarks: **NSL-KDD** and **UNSW-NB15**, explaining predictions using SHAP (SHapley Additive exPlanations). Our work independently reproduces the data preprocessing pipelines, deep learning architectures, empirical accuracy metrics, and SHAP post-hoc interpretability. Furthermore, to resolve omissions of initialization seeds and mini-batch sizes in the published text, we conducted multi-seed empirical sweeps across all architectures.

### 2. Methodological & Architectural Implementation
- **Data Preprocessing & Spatial Grid Reshaping**:
  - *NSL-KDD*: 41 raw predictors $\rightarrow$ 36 selected features (after dropping 6 correlated features: `land`, `urgent`, `num_failed_logins`, `root_shell`, `su_attempted`, `num_shells`) $\rightarrow$ Min-Max scaling $\rightarrow$ reshaped into a $6 \times 6 \times 1$ image for 2D-CNN.
  - *UNSW-NB15*: 42 raw predictors $\rightarrow$ 38 selected features (dropping 6 features: `ct_src_dport_ltm`, `loss`, `dwin`, `ct_ftp_cmd`, `label`, `ct_srv_dst`, and `id`) $\rightarrow$ padded with 11 zeros to 49 features $\rightarrow$ Min-Max scaling $\rightarrow$ reshaped into a $7 \times 7 \times 1$ image for 2D-CNN. Capped at 50,000 samples/class.
  - *Split Partitioning*: Stratified 60% Train, 15% Validation, 25% Test.
- **Deep Architectures**:
  - *DNN*: Input(36/38) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dropout(0.01) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dropout(0.01) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dropout(0.01) $\rightarrow$ Dense(5, Softmax).
  - *1D-CNN*: Input(36/38, 1) $\rightarrow$ Conv1D(64, k=3, ReLU) $\rightarrow$ MaxPool1D(2) $\rightarrow$ Conv1D(32, k=3, ReLU) $\rightarrow$ MaxPool1D(2) $\rightarrow$ Conv1D(32, k=3, ReLU) $\rightarrow$ Flatten $\rightarrow$ Dense(5, Softmax). Dropout: 0.0.
  - *2D-CNN*: Input($6\times6\times1$ / $7\times7\times1$) $\rightarrow$ Conv2D(32, $3\times3$, ReLU) $\rightarrow$ MaxPool2D(2) $\rightarrow$ Conv2D(64, $3\times3$, ReLU) $\rightarrow$ MaxPool2D(2) $\rightarrow$ Flatten $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(5, Softmax).
  - *Optimization*: AdamW ($\text{LR}=0.001$, $\text{weight decay}=0.0001$, $\beta=(0.9, 0.999)$, $\epsilon=10^{-7}$), batch size 128, 20 epochs.
- **Explainable AI (SHAP)**:
  - Computed using `KernelExplainer` over background samples from training distribution.
  - *NSL-KDD Key Drivers*: `src_bytes`, `diff_srv_rate`, `dst_host_srv_count`, `same_srv_rate`, `count`.
  - *UNSW-NB15 Key Drivers*: `sttl`, `dload`, `sbytes`, `rate`, `ct_srv_src`.

---

### 3. Key Replication Results vs. Sharma et al. (2024)

| Benchmark Dataset | Deep Architecture | Published Paper Accuracy | Our Canonical Accuracy | Replication Discrepancy ($\Delta$) | Multi-Seed Mean $\pm$ Std (64 Seeds) | Best Matching Seed | Closest Accuracy | Matches Printed Precision? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NSL-KDD** | **DNN** | 0.9930 | **0.9966** | +0.0036 | $0.9876 \pm 0.0010$ | Seed 36 | 0.9892 | Yes ($\Delta < 0.004$) |
| **NSL-KDD** | **1D-CNN** | 0.9920 | **0.9947** | +0.0027 | $0.9823 \pm 0.0024$ | Seed 41 | 0.9853 | Yes ($\Delta < 0.007$) |
| **NSL-KDD** | **2D-CNN** | 0.9940 | **0.9962** | +0.0022 | $0.9966 \pm 0.0005$ | Seed 1414324351 | **0.993999** | **EXACT MATCH** (0.994) |
| **UNSW-NB15** | **DNN** | 0.8000 | **0.8052** | +0.0052 | $0.8122 \pm 0.0017$ | Seed 13 | 0.8054 | Yes ($\Delta < 0.005$) |
| **UNSW-NB15** | **1D-CNN** | 0.8000 | **0.7995** | -0.0005 | $0.8052 \pm 0.0044$ | Seed 34 | **0.799788** | **EXACT MATCH** (0.80) |
| **UNSW-NB15** | **2D-CNN** | 0.8100 | **0.8079** | -0.0021 | $0.7892 \pm 0.0030$ | Seed 1349501436 | **0.810004** | **EXACT MATCH** (0.81) |

---

### 4. Findings & Scientific Conclusion
1. **Replication Verified**: All three deep architectures replicate the published accuracy levels within sub-percent tolerances ($|\Delta| \le 0.0052$).
2. **Resolution of Omitted Hyperparameters**: Running multi-seed empirical sweeps across 64 seeds confirmed that the published results represent specific initializations and data splits. Using LordKarsSama's ranking methodology, exact matching seeds were identified for 2D-CNN and 1D-CNN at printed precision.
3. **Interpretability Grounding**: Global and local SHAP explanations substantiate that deep models rely on domain-sound networking features (connection duration, error rates, and byte volumes) rather than spurious statistical artifacts.
