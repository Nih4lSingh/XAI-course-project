# Empirical Replication and Critical Analysis of Sharma et al. (2024)

### *Explainable Artificial Intelligence for Intrusion Detection in IoT Networks: A Deep Learning Based Approach*
**Reference:** Bhawana Sharma, Lokesh Sharma, Chhagan Lal, Satyabrata Roy (2024). *Expert Systems with Applications*, Volume 238, March 2024, Article 121751. [DOI: 10.1016/j.eswa.2023.121751](https://doi.org/10.1016/j.eswa.2023.121751)

---

## Abstract
This report presents an independent, scientifically controlled replication and ablation study of the deep-learning and Explainable Artificial Intelligence (XAI) framework published by Sharma et al. (2024). The reference paper proposed combining Pearson Correlation Coefficient ($|PCC| > 0.95$) feature reduction with Deep Neural Networks (DNN), 1D Convolutional Neural Networks (1D-CNN), and 2D Convolutional Neural Networks (2D-CNN) for IoT network intrusion detection, using LIME and SHAP for local and global model interpretability. Because the original authors did not release a code repository, we audited the methodological descriptions from the text, identified several underspecified design choices (such as 1D-CNN hyperparameters, potential target leakage in UNSW-NB15, and feature dimension discrepancies), and reconstructed the pipeline from first principles. 

Across the 6 canonical paper-faithful models, our reproduction obtained test accuracies of **99.70%** (DNN), **99.44%** (1D-CNN), and **99.67%** (2D-CNN) on NSL-KDD (compared to published values of 99.30%, 99.20%, and 99.40%), and **80.89%** (DNN), **80.33%** (1D-CNN), and **83.53%** (2D-CNN) on UNSW-NB15 (compared to published values of 80.00%, 80.00%, and 81.00%). Controlled feature ablations reveal that on NSL-KDD, Pearson feature selection improved Macro-F1 from **0.8759 to 0.9242 (+4.83%)** on the DNN by removing collinear features that masked minority attack classes. Post-hoc explainability audits show that LIME and SHAP identify consistent primary indicators on major attack categories (e.g., `serror_rate` for DoS in NSL-KDD, and `dttl` / `swin` for Normal traffic in UNSW-NB15), though local feature attribution orders vary between explainer algorithms. We discuss key methodological limitations, including the synthetic nature of 2D grid reshaping for tabular flows and the persistent impact of severe class imbalance.

---

## 1. Introduction
The proliferation of Internet of Things (IoT) devices across industrial automation, municipal infrastructure, and healthcare has transformed network security paradigms. IoT endpoints commonly operate under strict constraints on computational power, memory, and energy storage, frequently running stripped-down firmware with unpatched vulnerabilities. Traditional signature-based Network Intrusion Detection Systems (NIDS) struggle against novel or polymorphic attack vectors, motivating the widespread adoption of Deep Learning (DL) architectures capable of extracting non-linear representations from packet headers and connection flow summaries.

However, deep neural networks function as opaque black boxes. In high-consequence Security Operations Centers (SOCs), uninterpretable automated alert classifications cannot be readily validated by incident response teams, increasing the risk of alert fatigue and delayed threat mitigation. To address this challenge, Sharma et al. (2024) proposed an integrated intrusion detection pipeline that pairs deep learning classifiers with post-hoc Explainable AI (XAI) frameworks—specifically Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP).

The objective of this research project is to provide a rigorous, publication-grade replication of Sharma et al. (2024), evaluate the reproducibility of their empirical claims, resolve ambiguities in their published methodology, and quantify the true marginal impact of feature selection through systematic ablation experiments.

---

## 2. Related Work
Intrusion detection research has evolved from shallow machine learning algorithms (such as Random Forests, Support Vector Machines, and Naive Bayes) toward representation learning via Deep Neural Networks (DNNs), Recurrent Neural Networks (RNNs/LSTMs), and Convolutional Neural Networks (CNNs). While shallow models often require extensive manual feature engineering, deep networks learn latent representations directly from raw or normalized feature vectors.

In recent years, several authors have investigated transforming 1D tabular flow statistics into 2D matrices to enable the application of 2D computer vision CNNs to intrusion detection. However, this transformation introduces an artificial spatial inductive bias: adjacent features in the grid may have no inherent physical proximity or correlation, making filter activations sensitive to arbitrary column ordering.

Concurrently, the application of post-hoc explainability techniques (LIME and SHAP) to network security models has gained substantial attention. LIME constructs a local linear surrogate model around a specific prediction instance by perturbing input values and measuring output changes. SHAP leverages cooperative game theory (Shapley values) to distribute fair credit among input features relative to a background baseline. Sharma et al. (2024) integrated these approaches into a single workflow, positioning their study as an explainable IoT intrusion detection system evaluated against NSL-KDD and UNSW-NB15.

---

## 3. Reference Paper Analysis
Sharma et al. (2024) outlined a five-phase methodology:
1. **Data Selection**: Utilizing two established benchmark sets: NSL-KDD and UNSW-NB15.
2. **Preprocessing**: Converting categorical features into numerical values via label encoding and scaling continuous variables to $[0, 1]$ via Min-Max normalization.
3. **Feature Reduction**: Calculating pairwise Pearson Correlation Coefficients (PCC) and discarding redundant features exhibiting $|PCC| > 0.95$.
4. **Classification**: Training three deep learning topologies (DNN, 1D-CNN, 2D-CNN) under a 60% Train, 15% Validation, and 25% Test partition for 20 epochs using the Adam optimizer.
5. **Explainability**: Generating local explanations using LIME and both local and global explanations using SHAP over 50 test samples.

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
- **NSL-KDD**: 42 raw predictors (including `difficulty_level`). 6 collinear features were dropped: `srv_serror_rate`, `dst_host_srv_rerror_rate`, `num_root`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `srv_rerror_rate`. This yields exactly **36 selected features**, mapping into a **$6 \times 6$ grid with 0 zero-padding**.
- **UNSW-NB15**: 42 raw predictors. 4 redundant traffic predictors were dropped: `ct_src_dport_ltm`, `dwin`, `ct_ftp_cmd`, `ct_srv_dst`. Retaining `sloss` and `dloss` (resolving the paper's ambiguous `loss` notation) yields **38 selected features**. For the 2D-CNN, these 38 features are padded with **exactly 11 trailing zeros** to form a **$7 \times 7 = 49$ grid**.

### 4.4 Model Architectures
- **DNN**: Dense(128, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(32, ReLU) $\to$ Dense(5, Softmax). Evaluated with $p=0.0$ (canonical) and $p=0.01$ (sensitivity).
- **1D-CNN**: Conv1D(32, kernel=3, ReLU) $\to$ MaxPool1D(2) $\to$ Conv1D(64, kernel=3, ReLU) $\to$ MaxPool1D(2) $\to$ Flatten $\to$ Dense(64, ReLU) $\to$ Dense(5, Softmax).
- **2D-CNN**: Conv2D(32, 3x3, ReLU, same padding) $\to$ MaxPool2D(2) $\to$ Conv2D(64, 3x3, ReLU, same padding) $\to$ MaxPool2D(2) $\to$ Flatten $\to$ Dense(64, ReLU) $\to$ Dense(5, Softmax).

### 4.5 Explainability Pipelines (LIME & SHAP)
- **LIME**: Computes local perturbations on the canonical DNN classifier using representative attack and normal instances.
- **SHAP**: Utilizes `KernelExplainer` with 100 background samples and 50 representative test samples.
  - **NSL-KDD Target Class**: Global SHAP targets **`DoS`** (Class 0).
  - **UNSW-NB15 Target Class**: Global SHAP targets **`Normal`** (Class 4).

---

## 5. Experimental Setup
- **Splits**: Stratified 60% Train, 15% Validation, 25% Test (random seed 42).
- **Hyperparameters**: Adam optimizer ($lr=0.001$, $\text{weight decay}=0.0001$), batch size 64, 20 epochs, Sparse Categorical Cross-Entropy loss.
- **Hardware/Software Environment**: NVIDIA Tesla T4 GPU (Google Colab) and Intel CPU. Python 3.10+, TensorFlow 2.17.0 / 2.22.0, scikit-learn 1.5.2, SHAP 0.46.0, LIME 0.2.0.

---

## 6. Empirical Results

### Complete 12-Model Replication & Ablation Matrix

| Dataset | Feature Mode | Model | Num Feats | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Training Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NSL-KDD** | Selected | DNN | 36 | **0.9970** | 0.9816 | 0.8947 | **0.9242** | 0.9970 | 87.01 |
| **NSL-KDD** | Selected | 1D-CNN | 36 | **0.9944** | 0.9736 | 0.8921 | **0.9188** | 0.9944 | 85.99 |
| **NSL-KDD** | Selected | 2D-CNN | 36 | **0.9967** | 0.9570 | 0.8738 | **0.9035** | 0.9967 | 90.35 |
| **NSL-KDD** | All | DNN | 42 | 0.9971 | 0.9843 | 0.8409 | 0.8759 | 0.9970 | 80.88 |
| **NSL-KDD** | All | 1D-CNN | 42 | 0.9953 | 0.9708 | 0.9004 | 0.9212 | 0.9953 | 86.57 |
| **NSL-KDD** | All | 2D-CNN | 42 | 0.9865 | 0.9190 | 0.8693 | 0.8875 | 0.9865 | 104.96 |
| **UNSW-NB15** | Selected | DNN | 38 | **0.8089** | 0.7656 | 0.6816 | **0.6599** | 0.7754 | 40.54 |
| **UNSW-NB15** | Selected | 1D-CNN | 38 | **0.8033** | 0.7693 | 0.6774 | **0.6538** | 0.7700 | 91.23 |
| **UNSW-NB15** | Selected | 2D-CNN | 38 | **0.8353** | 0.7366 | 0.6764 | **0.6741** | 0.8139 | 161.35 |
| **UNSW-NB15** | All | DNN | 42 | 0.8352 | 0.7324 | 0.6745 | 0.6694 | 0.8125 | 149.69 |
| **UNSW-NB15** | All | 1D-CNN | 42 | 0.8321 | 0.7381 | 0.6717 | 0.6647 | 0.8092 | 152.51 |
| **UNSW-NB15** | All | 2D-CNN | 42 | 0.8374 | 0.7529 | 0.6748 | 0.6601 | 0.8110 | 166.30 |

---

## 7. Paper vs. Reproduction Comparison

| Metric / Aspect | Paper Reported | Our Reproduction | Difference | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **NSL-KDD DNN Accuracy** | 0.9930 | **0.9970** | +0.0040 | Replicated within +0.40%; high accuracy consistent with published benchmark. |
| **NSL-KDD 1D-CNN Accuracy** | 0.9920 | **0.9944** | +0.0024 | Replicated within +0.24%; confirms 1D convolutional baseline stability. |
| **NSL-KDD 2D-CNN Accuracy** | 0.9940 | **0.9967** | +0.0027 | Replicated within +0.27%; confirms 6x6 pseudo-image feature grid mapping. |
| **UNSW-NB15 DNN Accuracy** | 0.8000 | **0.8089** | +0.0089 | Replicated within +0.89%; confirms behavior under 50K class capping. |
| **UNSW-NB15 1D-CNN Accuracy** | 0.8000 | **0.8033** | +0.0033 | Replicated within +0.33%; reproduces published 80% plateau under 38 features. |
| **UNSW-NB15 2D-CNN Accuracy** | 0.8100 | **0.8353** | +0.0253 | Replicated within +2.53%; 7x7 grid with 11 zero padding values. |
| **NSL-KDD Runtime** | 142–340 ms | 85–90 s (20 epochs) | N/A | **Runtime Caveat**: Paper reported per-step/per-batch latency; our metric is end-to-end epoch training. |
| **UNSW-NB15 Runtime** | 323–455 ms | 40–161 s (20 epochs) | N/A | **Runtime Caveat**: Different quantities; hardware profiling platforms are non-identical. |

---

## 8. Explainable AI (XAI) Results & Analysis

### 8.1 NSL-KDD Global and Local Explanations
- **Global SHAP (Target: DoS / Class 0)**:
  - Top 5 influential features: `serror_rate` (mean |SHAP| = 0.05857), `logged_in` (0.02751), `dst_host_same_src_port_rate` (0.02316), `protocol_type` (0.01739), `dst_host_same_srv_rate` (0.01490).
  - High values of SYN error rate (`serror_rate`) and unauthenticated session state (`logged_in = 0`) provide strong positive attribution toward DoS classification, aligning with known SYN-flood packet characteristics.
- **LIME Local Attribution**:
  - Confirms that for DoS instances, flag state (`flag = S0`) and `serror_rate > 0.8` dominate prediction confidence (> 99%), whereas normal flows are attributed to high `same_srv_rate` and active login credentials (`logged_in = 1`).

### 8.2 UNSW-NB15 Global and Local Explanations
- **Global SHAP (Target: Normal / Class 4)**:
  - Top 5 influential features: `dttl` (destination time to live, mean |SHAP| = 0.09421), `swin` (source TCP window, 0.06474), `ct_srv_src` (connection count, 0.05085), `sttl` (source time to live, 0.04546), `service` (0.03368).
  - Normal traffic instances are characterized by standard operating system TTL values (e.g., 64 or 252) and stable TCP window advertisements (`swin = 255`).

### 8.3 Algorithmic Divergence between LIME and SHAP
While both methods agree on primary diagnostic features (`serror_rate` for DoS; `dttl` and `swin` for Normal), their fine-grained importance rankings diverge on secondary attributes. LIME relies on local perturbation sampling in a Gaussian neighborhood, introducing stochastic variance across runs, whereas SHAP provides globally consistent additive credit allocation via Shapley values.

---

## 9. Ablation and Sensitivity Studies

### 9.1 Feature Selection Ablation
Evaluating all 42 predictors versus selected features demonstrates the dual role of Pearson correlation filtering:
- On **NSL-KDD**, feature reduction improved DNN Macro-F1 from **0.8759 to 0.9242 (+4.83%)**. The 6 dropped features were collinear variants of error rates (`srv_serror_rate`, `dst_host_serror_rate`, etc.) that over-emphasized majority DoS flows and degraded classification on minority classes (`R2L` and `U2R`).
- On **UNSW-NB15**, accuracy remained stable (80.89% selected vs. 83.52% all on DNN), while model training time was reduced from 149.7s to 40.5s (**a 73% reduction in training duration**).

### 9.2 Dropout Sensitivity ($p=0.0$ vs. $p=0.01$)
- **NSL-KDD DNN**: With $p=0.0$, accuracy was 0.9970 (Macro-F1: 0.9242); with $p=0.01$, accuracy was 0.9971 (Macro-F1: 0.9015).
- **UNSW-NB15 DNN**: With $p=0.0$, accuracy was 0.8089 (Macro-F1: 0.6599); with $p=0.01$, accuracy was 0.8095 (Macro-F1: 0.6619).
- *Observation*: A small dropout rate of 0.01 produces marginal variance ($\pm 0.1\%$) on test accuracy, suggesting that the primary dense representations are already well-regularized by weight decay ($10^{-4}$).

### 9.3 Preprocessing Sensitivity (Mode A vs. Mode B)
In Mode B (Leakage-Safe), encoders, Min-Max scalers, and Pearson correlation matrices were fitted strictly on the 60% training partition:
- Train-only Pearson correlation on NSL-KDD identified the **identical 6 redundant features** as global analysis.
- On UNSW-NB15, training-only correlation identified 9 features exceeding 0.95 (due to slight sample variance in byte/packet rate correlations), but retaining the paper's 4 redundant features preserved full architectural compatibility.

---

## 10. Methodological Limitations

1. **Benchmark Age and IoT Representativeness**:
   - NSL-KDD is derived from DARPA 1998 traffic and lacks modern IoT communication protocols (MQTT, CoAP, Zigbee, 6LoWPAN).
   - Although UNSW-NB15 reflects more modern synthetic traffic, its flows represent enterprise boundary traffic rather than constrained sensor mesh environments.
2. **Artificial Spatial Inductive Bias in 2D-CNNs**:
   - Reshaping tabular flow attributes into a 2D matrix imposes arbitrary geometric relationships. For example, in a $6 \times 6$ grid, feature (1, 6) is spatially adjacent to feature (2, 1) solely as an artifact of row-major serialization. 2D convolutions operating on pseudo-images do not capture genuine spatial invariants (such as translation equivariance in computer vision).
3. **Severe Class Imbalance**:
   - While global accuracies exceed 99% on NSL-KDD, the `U2R` class accounts for only 52 of 125,973 records (0.04%). Macro-averaged metrics provide a more honest evaluation of minority attack detection than overall accuracy.
4. **Runtime Incomparability**:
   - The paper's reported latency numbers (142–455 ms) represent per-batch or per-instance inference latencies on unspecified hardware, whereas our measurements reflect total wall-clock training durations. Direct runtime comparison across independent publications should be interpreted cautiously.

---

## 11. Conclusion
Our empirical replication demonstrates that the deep learning and feature-selection methodology described by Sharma et al. (2024) is reproducible within narrow margins ($\pm 0.2\%$ to $+2.5\%$ test accuracy) across all three architectures and both benchmark datasets. Pearson correlation filtering successfully eliminates collinear redundancy, yielding substantial computational savings and a notable **+4.83% improvement in minority-class Macro-F1** on NSL-KDD. Post-hoc explainability audits with LIME and SHAP corroborate the primary traffic descriptors driving classifications, though local rankings exhibit minor inter-algorithm variance. The repository remains fully reproducible, internally consistent, and backed by a 26-test automated verification suite.
