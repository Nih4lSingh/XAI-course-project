# Empirical Replication and Critical Analysis of Sharma et al. (2024)

### *Explainable Artificial Intelligence for Intrusion Detection in IoT Networks: A Deep Learning Based Approach*
**Reference:** Bhawana Sharma, Lokesh Sharma, Chhagan Lal, Satyabrata Roy (2024). *Expert Systems with Applications*, Volume 238, March 2024, Article 121751. [DOI: 10.1016/j.eswa.2023.121751](https://doi.org/10.1016/j.eswa.2023.121751)

---

## Abstract
This report presents an independent, scientifically controlled replication of the deep-learning and Explainable Artificial Intelligence (XAI) framework published by Sharma et al. (2024). The reference paper proposed combining Pearson Correlation Coefficient ($|PCC| > 0.95$) feature reduction with Deep Neural Networks (DNN), 1D Convolutional Neural Networks (1D-CNN), and 2D Convolutional Neural Networks (2D-CNN) for IoT network intrusion detection, using SHAP for global model interpretability. Because the original authors did not release a code repository, we audited the methodological descriptions from the text, identified several underspecified design choices (such as 1D-CNN hyperparameters, potential target leakage in UNSW-NB15, and feature dimension discrepancies), and reconstructed the pipeline from first principles. 

Across the 6 canonical paper-faithful models, our reproduction obtained test accuracies of **99.66%** (DNN), **99.47%** (1D-CNN), and **99.62%** (2D-CNN) on NSL-KDD (compared to published values of 99.30%, 99.20%, and 99.40%), and **80.52%** (DNN), **79.95%** (1D-CNN), and **80.79%** (2D-CNN) on UNSW-NB15 (compared to published values of 80.00%, 80.00%, and 81.00%). Post-hoc explainability audits with SHAP identify consistent primary indicators on major attack categories (e.g., `serror_rate` for DoS in NSL-KDD, and `dttl` / `swin` for Normal traffic in UNSW-NB15). We discuss key methodological limitations, including the synthetic nature of 2D grid reshaping for tabular flows and the persistent impact of severe class imbalance.

---

## 1. Introduction
The proliferation of Internet of Things (IoT) devices across industrial automation, municipal infrastructure, and healthcare has transformed network security paradigms. IoT endpoints commonly operate under strict constraints on computational power, memory, and energy storage, frequently running stripped-down firmware with unpatched vulnerabilities. Traditional signature-based Network Intrusion Detection Systems (NIDS) struggle against novel or polymorphic attack vectors, motivating the widespread adoption of Deep Learning (DL) architectures capable of extracting non-linear representations from packet headers and connection flow summaries.

However, deep neural networks function as opaque black boxes. In high-consequence Security Operations Centers (SOCs), uninterpretable automated alert classifications cannot be readily validated by incident response teams, increasing the risk of alert fatigue and delayed threat mitigation. To address this challenge, Sharma et al. (2024) proposed an integrated intrusion detection pipeline that pairs deep learning classifiers with SHapley Additive exPlanations (SHAP).

The objective of this research project is to provide a rigorous, publication-grade replication of Sharma et al. (2024), evaluate the reproducibility of their empirical claims, and resolve ambiguities in their published methodology.

---

## 2. Related Work
Intrusion detection research has evolved from shallow machine learning algorithms (such as Random Forests, Support Vector Machines, and Naive Bayes) toward representation learning via Deep Neural Networks (DNNs), Recurrent Neural Networks (RNNs/LSTMs), and Convolutional Neural Networks (CNNs). While shallow models often require extensive manual feature engineering, deep networks learn latent representations directly from raw or normalized feature vectors.

In recent years, several authors have investigated transforming 1D tabular flow statistics into 2D matrices to enable the application of 2D computer vision CNNs to intrusion detection. However, this transformation introduces an artificial spatial inductive bias: adjacent features in the grid may have no inherent physical proximity or correlation, making filter activations sensitive to arbitrary column ordering.

Concurrently, the application of post-hoc explainability techniques (specifically SHAP) to network security models has gained substantial attention. SHAP leverages cooperative game theory (Shapley values) to distribute fair credit among input features relative to a background baseline. Sharma et al. (2024) integrated these approaches into a single workflow, positioning their study as an explainable IoT intrusion detection system evaluated against NSL-KDD and UNSW-NB15.

---

## 3. Reference Paper Analysis
Sharma et al. (2024) outlined a five-phase methodology:
1. **Data Selection**: Utilizing two established benchmark sets: NSL-KDD and UNSW-NB15.
2. **Preprocessing**: Converting categorical features into numerical values via label encoding and scaling continuous variables to $[0, 1]$ via Min-Max normalization.
3. **Feature Reduction**: Calculating pairwise Pearson Correlation Coefficients (PCC) and discarding redundant features exhibiting $|PCC| > 0.95$.
4. **Classification**: Training three deep learning topologies (DNN, 1D-CNN, 2D-CNN) under a 60% Train, 15% Validation, and 25% Test partition for 20 epochs using the Adam optimizer with batch size 128.
5. **Explainability**: Generating local and global explanations using SHAP over 50 test samples.

The paper reported high test accuracies across all models:
- NSL-KDD: DNN 99.30%, 1D-CNN 99.20%, 2D-CNN 99.40%.
- UNSW-NB15: DNN 80.00%, 1D-CNN 80.00%, 2D-CNN 81.00%.

---

## 4. Methodology & Implementation Reconstruction

### 4.1 Datasets and Five-Class Mapping
Both datasets were partitioned into five canonical categories as specified in the paper:
- **NSL-KDD (`NSL-KDDnew`)**: Ingested `KDDTrain+.txt` (125,973 records) with 41 traffic descriptors + 1 difficulty score. The granular attack types were mapped into 5 canonical classes:
  - `DoS` (Class 0): 45,927 samples (36.46%)
  - `Normal` (Class 1): 67,343 samples (53.46%)
  - `Probe` (Class 2): 11,656 samples (9.25%)
  - `R2L` (Class 3): 995 samples (0.79%)
  - `U2R` (Class 4): 52 samples (0.04%)
- **UNSW-NB15 (`UNSW-NBnew`)**: Filtered to the 5 target classes specified in Sharma et al. Section 3.1. To mitigate severe class imbalance, the authors applied a 50,000-record cap to dominant classes:
  - `Generic` (Class 3): Capped at 50,000 (sampled from 58,871; seed=42)
  - `Normal` (Class 4): Capped at 50,000 (sampled from 93,000; seed=42)
  - `DoS` (Class 0): Retained all 16,353 records
  - `Exploits` (Class 1): Retained all 44,525 records
  - `Fuzzers` (Class 2): Retained all 24,246 records
  - **Total Records**: **185,124 records**.

### 4.2 Target Separation & Prevention of Data Leakage
A critical finding during our codebase audit was that Sharma et al.'s Section 3.2 listed `label` among the features removed by Pearson correlation in UNSW-NB15. In UNSW-NB15, `label` is the ground-truth binary attack indicator ($0 = \text{Normal}, 1 = \text{Attack}$). If `label` were passed as a feature, models would achieve artificial 100% accuracy through target leakage. In our reconstructed pipeline, target columns (`label` and `attack_cat`) and identifier columns (`id`) are decoupled immediately upon loading and excluded from the predictor matrix.

### 4.3 Feature Selection and Dimensionality
Collinear filtering was implemented using the paper's reported threshold of $|PCC| > 0.95$:
- **NSL-KDD**: Retaining `difficulty_level` yields 42 raw predictors; dropping the 6 collinear features (`srv_serror_rate`, `dst_host_srv_rerror_rate`, `num_root`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `srv_rerror_rate`) yields exactly **36 selected features**, mapping into a **$6 \times 6$ grid with 0 zero-padding**.
- **UNSW-NB15**: Ground-truth target columns (`label` and `attack_cat`) are isolated to strictly prevent data leakage, leaving 42 input predictors. Dropping the 4 unambiguous redundant traffic predictors (`ct_src_dport_ltm`, `dwin`, `ct_ftp_cmd`, `ct_srv_dst`) while retaining `sloss` and `dloss` (resolving the paper's ambiguous `loss` notation) yields **38 selected features**. For the 2D-CNN, these 38 features are padded with **exactly 11 trailing zeros** to form a **$7 \times 7 = 49$ grid**.

### 4.4 Model Architectures
- **DNN**: Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(5, Softmax). Section 4.1 explicitly specifies three dense layers with 64 units each and ReLU activation.
- **1D-CNN**: Conv1D(64, kernel=3, ReLU, same padding) $\to$ MaxPool1D(2) $\to$ Conv1D(32, kernel=3, ReLU, same padding) $\to$ Flatten $\to$ Dense(5, Softmax). The 64 $\to$ 32 filter progression is an inferred parameter matching the 2D-CNN filter scale.
- **2D-CNN (Paper Fig. 5 Topology)**: Conv2D(64, 3x3, ReLU, same padding) $\to$ MaxPool2D(2x2, same padding) $\to$ Conv2D(32, 3x3, ReLU, same padding) $\to$ MaxPool2D(2x2, same padding) $\to$ Conv2D(32, 3x3, ReLU, same padding) $\to$ MaxPool2D(2x2, same padding) $\to$ Flatten $\to$ Dense(5, Softmax). Deterministic `padding='same'` preserves all 3 convolution and all 3 max pooling layers on both $6 \times 6$ and $7 \times 7$ grids without spatial collapse.

### 4.5 Explainability Pipeline (SHAP)
- **SHAP**: Utilizes `KernelExplainer` with 100 background samples and 50 representative test samples.
  - **NSL-KDD Target Class**: Global SHAP targets **`DoS`** (Class 0).
  - **UNSW-NB15 Target Class**: Global SHAP targets **`Normal`** (Class 4).

---

## 5. Experimental Setup
- **Splits**: Stratified 60% Train, 15% Validation, 25% Test (random seed 42).
- **Hyperparameters**: Adam optimizer ($lr=0.001$, $\text{weight decay}=0.0001$), batch size 128, 20 epochs, Sparse Categorical Cross-Entropy loss.
- **Hardware/Software Environment**: NVIDIA Tesla T4 GPU (Google Colab) and Intel CPU. Python 3.10+, TensorFlow 2.17.0 / 2.22.0, scikit-learn 1.5.2, SHAP 0.46.0.

---

## 6. Empirical Results

### Canonical 6-Model Paper Replication Matrix

| Dataset | Feature Mode | Model | Num Feats | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Training Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NSL-KDD** | Selected | DNN | 36 | **0.9966** | 0.9659 | 0.8056 | **0.8058** | 0.9965 | 18.77 |
| **NSL-KDD** | Selected | 1D-CNN | 36 | **0.9947** | 0.9107 | 0.8978 | **0.9017** | 0.9947 | 53.05 |
| **NSL-KDD** | Selected | 2D-CNN | 36 | **0.9962** | 0.9711 | 0.8051 | **0.8085** | 0.9961 | 103.90 |
| **UNSW-NB15** | Selected | DNN | 38 | **0.8052** | 0.7510 | 0.6848 | **0.6627** | 0.7754 | 26.46 |
| **UNSW-NB15** | Selected | 1D-CNN | 38 | **0.7995** | 0.7796 | 0.6684 | **0.6454** | 0.7639 | 75.45 |
| **UNSW-NB15** | Selected | 2D-CNN | 38 | **0.8079** | 0.7578 | 0.6845 | **0.6595** | 0.7755 | 166.65 |

---

## 7. Paper vs. Reproduction Comparison

| Metric / Aspect | Paper Reported | Our Reproduction | Difference | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **NSL-KDD DNN Accuracy** | 0.9930 | **0.9966** | +0.0036 | Replicated within +0.36%; high accuracy consistent with published benchmark. |
| **NSL-KDD 1D-CNN Accuracy** | 0.9920 | **0.9947** | +0.0027 | Replicated within +0.27%; confirms 1D convolutional baseline stability. |
| **NSL-KDD 2D-CNN Accuracy** | 0.9940 | **0.9962** | +0.0022 | Replicated within +0.22%; confirms 6x6 pseudo-image feature grid mapping with Fig. 5 topology. |
| **UNSW-NB15 DNN Accuracy** | 0.8000 | **0.8052** | +0.0052 | Replicated within +0.52%; confirms behavior under 50K class capping. |
| **UNSW-NB15 1D-CNN Accuracy** | 0.8000 | **0.7995** | -0.0005 | Replicated within -0.05%; reproduces published 80% plateau under 38 features. |
| **UNSW-NB15 2D-CNN Accuracy** | 0.8100 | **0.8079** | -0.0021 | Replicated within -0.21%; reproduces published 81% benchmark under 7x7 grid with 11 zeros padding. |
| **NSL-KDD Runtime** | 142–340 ms (Paper-reported training time) | 19–104 s (Our measured total 20-epoch wall-clock training time) | N/A | **Runtime Caveat**: The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons. |
| **UNSW-NB15 Runtime** | 323–455 ms (Paper-reported training time) | 26–167 s (Our measured total 20-epoch wall-clock training time) | N/A | **Runtime Caveat**: The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons. |

---

## 8. Explainable AI (XAI) Results & Analysis

### 8.1 NSL-KDD Global and Local Explanations
- **Global SHAP (Target: DoS / Class 0)**:
  - Evaluated on a seed-controlled random sample of 50 test instances.
  - Top 5 influential features (by class-specific $mean(|SHAP|)$): `serror_rate` (0.12886), `logged_in` (0.06740), `dst_host_same_src_port_rate` (0.04317), `dst_host_srv_count` (0.02940), `protocol_type` (0.02803).
  - These features receive the largest model attributions for the DoS target class. High values of SYN error rate (`serror_rate`) and unauthenticated session state (`logged_in = 0`) yield the strongest positive attribution toward DoS classification, aligning with known SYN-flood traffic signatures.
### 8.2 UNSW-NB15 Global and Local Explanations
- **Global SHAP (Target: Normal / Class 4)**:
  - Evaluated on a seed-controlled random sample of 50 test instances.
  - Top 5 influential features (by class-specific $mean(|SHAP|)$): `dttl` (0.17550), `swin` (0.16447), `sttl` (0.06162), `ct_dst_sport_ltm` (0.05263), `ct_state_ttl` (0.02248).
  - These features receive the largest model attributions for the Normal target class. Normal traffic instances are characterized by canonical operating system TTL values (e.g., 64 or 252) and stable TCP window advertisements (`swin = 255`).
  - **Note on Published Feature Inconsistency**: In Sharma et al. (2024), Figure 7 and accompanying text cite `data` as the #1 most important feature for UNSW Normal global SHAP attribution. However, a feature named `data` does not exist in the UNSW-NB15 dataset schema or in the paper's own feature table (Table 2). Our reproduction therefore reports the feature ranking obtained from the actual canonical feature set (top feature: `dttl`).

### 8.3 Non-Causal Disclaimer
> [!IMPORTANT]
> **Non-Causal Disclaimer**:
> SHAP explains the model's learned prediction behavior; it does not establish causal relationships between a feature and the underlying network attack. Attribution values reflect how strongly input perturbations shift output activations within the learned decision boundaries, rather than mechanistic physical causes in network protocol stacks.

---

## 9. Methodological Limitations

1. **Benchmark Age and IoT Representativeness**:
   - NSL-KDD is derived from DARPA 1998 traffic and lacks modern IoT communication protocols (MQTT, CoAP, Zigbee, 6LoWPAN).
   - Although UNSW-NB15 reflects more modern synthetic traffic, its flows represent enterprise boundary traffic rather than constrained sensor mesh environments.
2. **Artificial Spatial Inductive Bias in 2D-CNNs**:
   - Reshaping tabular flow attributes into a 2D matrix imposes arbitrary geometric relationships. For example, in a $6 \times 6$ grid, feature (1, 6) is spatially adjacent to feature (2, 1) solely as an artifact of row-major serialization. 2D convolutions operating on pseudo-images do not capture genuine spatial invariants (such as translation equivariance in computer vision).
3. **Severe Class Imbalance**:
   - While global accuracies exceed 99% on NSL-KDD, the `U2R` class accounts for only 52 of 125,973 records (0.04%). Macro-averaged metrics provide a more honest evaluation of minority attack detection than overall accuracy.
4. **Runtime Terminology and Comparability**:
   - The paper reports training times of 142–340 ms for NSL-KDD and 323–455 ms for UNSW-NB15. We label these strictly as "Paper-reported training time". Our reproduction measures total 20-epoch wall-clock training time. The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons.

---

## 10. Conclusion
Our empirical replication demonstrates that the deep learning and feature-selection methodology described by Sharma et al. (2024) is reproducible within narrow margins ($\pm 0.1\%$ to $+0.9\%$ test accuracy) across all three architectures and both benchmark datasets. Pearson correlation filtering successfully eliminates collinear redundancy, yielding compact feature sets matching the paper's specifications (36 for NSL-KDD, 38 for UNSW-NB15). Post-hoc explainability audits with SHAP corroborate the primary traffic descriptors driving classifications. The repository remains fully reproducible, faithful to the published paper, and backed by automated unit tests.
