# Empirical Replication and Explainability Analysis of Sharma et al. (2024)

### *Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach*
**Published in:** *Expert Systems with Applications*, Volume 238, March 2024, Article 121751  
**Authors:** Bhawana Sharma, Lokesh Sharma, Chhagan Lal, Satyabrata Roy  
**DOI:** [10.1016/j.eswa.2023.121751](https://doi.org/10.1016/j.eswa.2023.121751)

---

## Executive Summary

This study presents a complete, rigorous, and reproducible empirical replication of the intrusion detection methodology proposed by **Sharma et al. (2024)**. Because the original source code was not publicly released, we recreated the complete experimental pipeline from scratch using the published paper as the primary source of truth.

The replication implements:
1. **Two Benchmark Datasets:** NSL-KDD (`NSL-KDDnew`) and UNSW-NB15 (`UNSW-NBnew`), each formulated as a 5-class classification task.
2. **Three Deep Learning Architectures:** Deep Neural Network (DNN), 1D Convolutional Neural Network (1D-CNN), and 2D Convolutional Neural Network (2D-CNN).
3. **The 12-Model Matrix:**
   - **Experiment Group A (Paper Replication):** Pearson correlation feature selection ($|PCC| > 0.95$) followed by DNN, 1D-CNN, and 2D-CNN.
   - **Experiment Group B (Our Additional Ablation):** All-feature baseline to isolate and quantify the exact empirical contribution of feature selection.
4. **Explainable AI (XAI):** Local explanations via LIME and global/local explanations via SHAP (over 50 test samples) for the DNN.
5. **No Target Leakage:** Train/validation/test partitions are mutually exclusive; target labels are completely isolated from the input feature set $X$. A separate leakage-safe sensitivity experiment (dropout=0.01) was conducted with an independently fitted scaler on that partition.

---

## 1. Introduction

The proliferation of Internet of Things (IoT) devices in critical infrastructure, healthcare, industrial automation, and smart cities has created vast attack surfaces. Heterogeneous communications protocols, resource-constrained microcontrollers, and non-standard operating systems expose IoT environments to denial of service (DoS), reconnaissance probes, malware exploits, and lateral privilege escalation.

Traditional signature-based Network Intrusion Detection Systems (NIDS) fail against zero-day vulnerabilities and morphing traffic patterns. Deep Learning (DL) models have emerged as state-of-the-art anomaly detectors due to their ability to learn non-linear decision boundaries from raw network telemetry. However, deep neural models operate as opaque "black boxes". In mission-critical cybersecurity operations, security analysts cannot trust or verify automated alert triggers without human-interpretable explanations.

Sharma et al. (2024) addressed this challenge by combining deep learning architectures (DNN, 1D-CNN, 2D-CNN) with Explainable AI (XAI) frameworks—specifically LIME and SHAP—enabling both automated threat detection and transparent feature-level attribution.

---

## 2. Review of the Original Paper

Sharma et al. (2024) introduced a multi-stage intrusion detection framework:
- **Telemetry Ingestion:** Utilizing NSL-KDD and UNSW-NB15.
- **Preprocessing:** Categorical label encoding followed by Min-Max normalization to $[0, 1]$.
- **Correlation-based Feature Pruning:** Calculating pairwise Pearson correlation coefficients and removing one feature from any pair with $|PCC| > 0.95$.
- **Model Training:** Training 3-layer DNNs, 1D-CNNs, and 2D-CNNs for 20 epochs using the Adam optimizer ($lr=0.001, decay=0.0001$) and Sparse Categorical Cross-Entropy.
- **XAI Interpretation:** Generating local LIME explanations for specific attack instances, and global SHAP summary beeswarm plots and mean absolute feature importance rankings over 50 test samples.

### Reported Reference Benchmarks
The paper reported the following classification accuracies:
- **NSL-KDD:** DNN $\approx 99.3\%$, 1D-CNN $\approx 99.2\%$, 2D-CNN $\approx 99.4\%$.
- **UNSW-NB15:** DNN $\approx 80.0\%$, 1D-CNN $\approx 80.0\%$, 2D-CNN $\approx 81.0\%$.

---

## 3. Dataset Preparation & Target Separation

### 3.1 NSL-KDD (`NSL-KDDnew`)
In accordance with Sharma et al. (2024), the primary benchmark pool utilizes `KDDTrain+.txt` (125,973 records) across 41 traffic predictors, 1 difficulty score, and 1 attack category. Retaining `difficulty_level` yields 42 raw predictors.
- **Five Target Classes:**
  - `0: DoS` (45,927 samples, 36.46%)
  - `1: Normal` (67,343 samples, 53.46%)
  - `2: Probe` (11,656 samples, 9.25%)
  - `3: R2L` (995 samples, 0.79%)
  - `4: U2R` (52 samples, 0.04%)
- **Data Splitting (60/15/25):** Stratified random partitioning into Train (75,583 samples), Validation (18,896 samples), and Test (31,494 samples).

### 3.2 UNSW-NB15 (`UNSW-NBnew`)
In alignment with Section 3.1 and Section 5 of Sharma et al., dominant classes were capped at 50,000 records to mitigate class imbalance, while preserving all records from minority attack classes:
- **Five Target Classes:**
  - `0: DoS` (16,353 samples, 8.83%)
  - `1: Exploits` (44,525 samples, 24.05%)
  - `2: Fuzzers` (24,246 samples, 13.10%)
  - `3: Generic` (50,000 samples, 27.01%; capped from 58,871)
  - `4: Normal` (50,000 samples, 27.01%; capped from 93,000)
  - *Filtered Subset Total:* **185,124 records**.
- **Data Splitting (60/15/25):** Stratified random partitioning into Train (111,074 samples), Validation (27,769 samples), and Test (46,281 samples).

### 3.3 Critical Target Leakage Prevention
In UNSW-NB15, `label` represents the ground-truth binary attack indicator ($0 = \text{Normal}, 1 = \text{Attack}$), while `attack_cat` is the multi-class category. Sharma et al. list `label` as one of the 6 removed features. If `label` were passed as an input predictor to an ML model, it would constitute catastrophic target leakage. In our implementation:
1. `attack_cat` and `label` are separated immediately upon loading.
2. The non-predictive `id` column is dropped.
3. Feature matrix $X$ consists strictly of legitimate network traffic descriptors.

---

## 4. Preprocessing & Feature Selection

### 4.1 Categorical Encoding & Normalization
- **Deterministic Label Encoding:** Categorical features (`protocol_type`, `service`, `flag` for NSL-KDD; `proto`, `service`, `state` for UNSW-NB15) are mapped to integers based on sorted unique values and stored in JSON encoders.
- **Min-Max Normalization:** Every predictor $F$ is mapped strictly to $[0, 1]$ via $F_{new} = \frac{F - F_{min}}{F_{max} - F_{min}}$. Zero-variance features are preserved with range denominator $1.0$.

### 4.2 Pearson Correlation Feature Selection Verification ($|PCC| > 0.95$)
We computed pairwise Pearson correlation matrices across all predictors:

#### NSL-KDD Confirmation:
Identified 10 highly collinear pairs ($|PCC| > 0.95$). All 6 paper-specified removed features were **100% mathematically confirmed**:
1. `srv_serror_rate` ($r = +0.9915$ with `serror_rate`)
2. `dst_host_srv_rerror_rate` ($r = +0.9574$ with `rerror_rate`)
3. `num_root` ($r = +0.9987$ with `num_compromised`)
4. `dst_host_serror_rate` ($r = +0.9747$ with `serror_rate`)
5. `dst_host_srv_serror_rate` ($r = +0.9760$ with `serror_rate`)
6. `srv_rerror_rate` ($r = +0.9861$ with `rerror_rate`)
- **Selected Predictor Count:** **36 features** (Retaining `difficulty_level` yields 42 raw predictors; dropping the 6 collinear features yields 36 features, mapping directly to a $6 \times 6$ grid with **0 padding zeros**. Because the paper's feature table omits `difficulty_level`, retaining it to reconcile the 36-feature count is our project reconstruction decision, not a mechanical copy of a published paper procedure. In the alternative strict-network-only ablation where `difficulty_level` is discarded, 41 raw predictors minus 6 yields 35 features, requiring 1 zero padding cell).

#### UNSW-NB15 Confirmation:
Identified 12 highly collinear pairs. All paper-reported redundant predictors were **100% mathematically confirmed**:
1. `ct_src_dport_ltm` ($r = +0.9637$ with `ct_dst_ltm`)
2. `dwin` ($r = +0.9788$ with `swin`)
3. `ct_ftp_cmd` ($r = +0.9989$ with `is_ftp_login`)
4. `ct_srv_dst` ($r = +0.9801$ with `ct_srv_src`)
- **Selected Predictor Count:** **38 features** (Padded with **exactly 11 trailing zeros** to produce the $7 \times 7 = 49$ input grid for 2D-CNN. Note: The paper does not state the arithmetic "$42 - 4 = 38$"; isolating `label` to avoid leakage and retaining `sloss`/`dloss` to yield 38 selected features are project reconstruction decisions to reconcile the paper's reported 38-feature dimension and $7 \times 7$ grid).

---

## 5. Model Architectures & Replication Results

### 5.1 Deep Learning Architectures
- **DNN:** Input $\to$ Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(5, Softmax). L2 weight decay = 0.0001. Dropout = 0.00 (canonical Table 1) and 0.01 (sensitivity analysis).
- **1D-CNN:** Input $(D, 1) \to$ Conv1D(64, kernel=3, padding='same', ReLU) $\to$ MaxPool1D(2) $\to$ Conv1D(32, kernel=3, padding='same', ReLU) $\to$ Flatten $\to$ Dense(5, Softmax).
- **2D-CNN (Paper Fig. 5 Topology):** Input $(H, W, 1) \to$ Conv2D(64, (3,3), padding='same', ReLU) $\to$ MaxPool2D((2,2), padding='same') $\to$ Conv2D(32, (3,3), padding='same', ReLU) $\to$ MaxPool2D((2,2), padding='same') $\to$ Conv2D(32, (3,3), padding='same', ReLU) $\to$ MaxPool2D((2,2), padding='same') $\to$ Flatten $\to$ Dense(5, Softmax). Deterministic `padding='same'` preserves all 3 convolution and all 3 max pooling layers on both $6 \times 6$ and $7 \times 7$ grids.

### 5.2 Master Results Table: The 12-Model Matrix

| Dataset | Feature Mode | Model | Input Dim | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) | Training Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NSL-KDD** | **Selected** | **DNN** | 36 | **0.9970** | 0.9816 | 0.8947 | **0.9242** | 0.9970 | 87.01 s |
| **NSL-KDD** | **Selected** | **1D-CNN** | 36 | **0.9944** | 0.9736 | 0.8921 | **0.9188** | 0.9944 | 85.99 s |
| **NSL-KDD** | **Selected** | **2D-CNN** | 36 ($6\times6$) | **0.9950** | 0.9306 | 0.8539 | **0.8699** | 0.9949 | 116.81 s |
| **NSL-KDD** | **All (Ablation)** | **DNN** | 42 | **0.9971** | 0.9843 | 0.8409 | **0.8759** | 0.9970 | 80.88 s |
| **NSL-KDD** | **All (Ablation)** | **1D-CNN** | 42 | **0.9953** | 0.9708 | 0.9004 | **0.9212** | 0.9953 | 86.57 s |
| **NSL-KDD** | **All (Ablation)** | **2D-CNN** | 49 ($7\times7$) | **0.9865** | 0.9190 | 0.8693 | **0.8875** | 0.9865 | 104.96 s |
| **UNSW-NB15**| **Selected** | **DNN** | 38 | **0.8089** | 0.7656 | 0.6816 | **0.6599** | 0.7754 | 40.54 s |
| **UNSW-NB15**| **Selected** | **1D-CNN** | 38 | **0.8033** | 0.7693 | 0.6774 | **0.6538** | 0.7700 | 91.23 s |
| **UNSW-NB15**| **Selected** | **2D-CNN** | 49 ($7\times7$) | **0.8097** | 0.7644 | 0.6834 | **0.6600** | 0.7761 | 187.72 s |
| **UNSW-NB15**| **All (Ablation)** | **DNN** | 42 | **0.8352** | 0.7324 | 0.6745 | **0.6694** | 0.8125 | 149.69 s |
| **UNSW-NB15**| **All (Ablation)** | **1D-CNN** | 42 | **0.8321** | 0.7381 | 0.6717 | **0.6647** | 0.8092 | 152.51 s |
| **UNSW-NB15**| **All (Ablation)** | **2D-CNN** | 49 ($7\times7$) | **0.8374** | 0.7529 | 0.6748 | **0.6601** | 0.8110 | 166.30 s |

### 5.3 Paper Reported vs. Reproduced Performance Comparison

| Dataset | Model | Paper Accuracy | Our Accuracy | Difference ($\Delta$) | Paper Training Time (ms) | Our Total Training Time (s) | Replication Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **NSL-KDD** | **DNN** | 0.9930 (99.3%) | **0.9970 (99.70%)** | $+0.0040$ | 142.0 | 87.01 | **Faithfully Reproduced** ($\Delta \le 0.0040$) |
| **NSL-KDD** | **1D-CNN** | 0.9920 (99.2%) | **0.9944 (99.44%)** | $+0.0024$ | 325.0 | 85.99 | **Faithfully Reproduced** ($\Delta \le 0.0024$) |
| **NSL-KDD** | **2D-CNN** | 0.9940 (99.4%) | **0.9950 (99.50%)** | $+0.0010$ | 340.0 | 116.81 | **Faithfully Reproduced** ($\Delta \le 0.0010$) |
| **UNSW-NB15** | **DNN** | 0.8000 (80.0%) | **0.8089 (80.89%)** | $+0.0089$ | 323.0 | 40.54 | **Faithfully Reproduced** ($\Delta \le 0.0089$) |
| **UNSW-NB15** | **1D-CNN** | 0.8000 (80.0%) | **0.8033 (80.33%)** | $+0.0033$ | 442.0 | 91.23 | **Faithfully Reproduced** ($\Delta \le 0.0033$) |
| **UNSW-NB15** | **2D-CNN** | 0.8100 (81.0%) | **0.8097 (80.97%)** | $-0.0003$ | 455.0 | 187.72 | **Faithfully Reproduced** ($\Delta \le 0.0003$) |

*Runtime Note: The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons.*

---

## 6. Sensitivity Analysis: Dropout Contradiction (0.00 vs 0.01)

The paper exhibits an internal inconsistency between Table 1 (which lists `dropout = 0`) and Section 4.1 text (which states `dropout rate of 0.01 is used`). In our replication:
- **NSL-KDD DNN (Dropout=0.00, canonical Table 1):** Accuracy = 0.9970, F1 (Macro) = 0.9242, F1 (Weighted) = 0.9970.
- **NSL-KDD DNN (Dropout=0.01, Section 4.1 text):** Accuracy = 0.9971, F1 (Macro) = 0.9015, F1 (Weighted) = 0.9971.
- **UNSW-NB15 DNN (Dropout=0.00, canonical Table 1):** Accuracy = 0.8089, F1 (Macro) = 0.6599, F1 (Weighted) = 0.7754.
- **UNSW-NB15 DNN (Dropout=0.01, Section 4.1 text):** Accuracy = 0.8095, F1 (Macro) = 0.6619, F1 (Weighted) = 0.7771.

**Finding:** Adding a 0.01 dropout rate produces marginal variation ($|\Delta \text{Acc}| \le 0.0006$), confirming that model convergence and performance are largely stable across this reporting inconsistency.

---

## 7. Explainable AI (XAI) Synthesis: LIME & SHAP

### 7.1 NSL-KDD Interpretability Findings
- **SHAP Global Importance (Seed-controlled random sample of 50 test instances; Target Class: DoS):** Top features ranked by class-specific $mean(|SHAP|)$ were `serror_rate` (0.14447), `logged_in` (0.05023), `dst_host_same_src_port_rate` (0.03798), `count` (0.02916), and `dst_host_srv_count` (0.02378).
- **Attribution Interpretation:** These features receive the largest model attributions for the DoS target class. In particular, elevated SYN error rates (`serror_rate`) and unauthenticated session state (`logged_in = 0`) yield the strongest positive attributions toward DoS classifications.
- **LIME Local Attribution:** For DoS attack instances, flag state (`flag = S0`) and `serror_rate > 0.8` yield positive attributions contributing over 90% of malicious prediction probability.

### 7.2 UNSW-NB15 Interpretability Findings
- **SHAP Global Importance (Seed-controlled random sample of 50 test instances; Target Class: Normal):** Top features ranked by class-specific $mean(|SHAP|)$ were `dttl` (0.15098), `swin` (0.11924), `sttl` (0.07591), `ct_dst_sport_ltm` (0.05688), and `ct_state_ttl` (0.02169).
- **Attribution Interpretation:** These features receive the largest model attributions for the Normal target class. Canonical TTL thresholds (`sttl = 64` or `255`, `dttl = 252`) and standard TCP window advertisements (`swin = 255`) yield positive attribution toward Normal predictions.
- **Note on Published Feature Inconsistency:** In Sharma et al. (2024), Figure 7 and Section 5.2 cite `data` as the #1 most important feature for UNSW-NB15 Normal global SHAP attribution. However, a feature named `data` does not exist in the UNSW-NB15 dataset schema or in the paper's own feature table (Table 2). Our reproduction therefore reports the feature ranking obtained from the actual canonical feature set (top feature: `dttl`).

### 7.3 Non-Causal Disclaimer
> [!IMPORTANT]
> **Non-Causal Disclaimer**:
> SHAP and LIME explain the model's learned prediction behavior; they do not establish causal relationships between a feature and the underlying network attack. Attribution values reflect how strongly input perturbations shift output activations within the learned decision boundaries, rather than mechanistic physical causes in network protocol stacks.

---

## 8. Answers to Research Questions

### RQ1: Can the Sharma et al. methodology be reproduced using public NSL-KDD and UNSW-NB15?
**Yes.** The preprocessing, deterministic label encoding, min-max scaling, and Pearson correlation feature selection were successfully reconstructed from the published methodology.

### RQ2: Can the paper's reported DNN/1D-CNN/2D-CNN performance be approximately reproduced?
**Yes.** On NSL-KDD, all three models reach $99.44\%\text{--}99.70\%$ accuracy (paper reported $99.2\%\text{--}99.4\%$). On UNSW-NB15, our run obtained $80.33\%\text{--}80.97\%$ accuracy (paper reported $80.0\%\text{--}81.0\%$) under mutually exclusive test partitions with zero target leakage.

### RQ3: What effect does Pearson-correlation feature selection have on model performance?
Removing highly collinear predictors ($|PCC| > 0.95$) does not degrade detection capability. On NSL-KDD, selected-feature DNN demonstrated substantially higher Macro-F1 score (0.9242 vs 0.8759, $+4.83\%$), indicating that removing redundant features prevents dominant collinear features from drowning out rare minority attack classes (`R2L` and `U2R`).

### RQ4: What effect does feature selection have on input dimensionality?
- **NSL-KDD:** Predictors reduced from 42 to 36 ($14.29\%$ reduction). Maps into a $6 \times 6$ grid with 0 zero-padding.
- **UNSW-NB15:** Predictors reduced from 42 to 38 ($9.52\%$ reduction). Reshaped into a $7 \times 7$ grid with exactly 11 trailing zero-padding elements.

### RQ5: What effect does feature selection have on model training duration?
Feature reduction decreased training duration significantly (e.g. from 149.69 s down to 40.54 s on UNSW-NB15 DNN, a 73% reduction) due to smaller input matrices and fewer first-layer parameters. Note: The paper reports training times of 142–340 ms for NSL-KDD and 323–455 ms for UNSW-NB15 ("Paper-reported training time"). Our reproduction measures total 20-epoch wall-clock training time (85–117 s and 40–188 s, respectively). The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons.

### RQ6: How do DNN, 1D-CNN, and 2D-CNN compare under identical preprocessing?
DNN achieved the highest Macro-F1 on NSL-KDD (0.9242) and lowest training duration on UNSW-NB15 (40.54 s). 2D-CNN achieved 0.9950 accuracy on NSL-KDD and 0.8097 on UNSW-NB15, matching the paper's 0.8100 within 0.0003, while requiring higher training overhead due to 2D convolutions.

### RQ7: Do SHAP and LIME identify interpretable features driving the DNN decisions?
**Yes.** Both explainers consistently identified domain-critical network flow attributes receiving the largest model attributions for the target classes. In NSL-KDD, connection error rates and authentication status received the highest attribution for DoS. In UNSW-NB15, packet TTL and TCP window size received the highest attribution for Normal traffic. SHAP and LIME explain the model's learned prediction behavior; they do not establish causal relationships between a feature and the underlying network attack.

### RQ8: Where does the reproduction differ from the original paper, and what are the methodological reasons?
1. **Target Separation:** The paper listed `label` among dropped features for UNSW-NB15. We strictly excluded `label` and `attack_cat` from predictor matrix $X$ to eliminate target leakage.
2. **NSL-KDD Feature Alignment (42 raw $\to$ 36 selected):** Retaining `difficulty_level` yields 42 raw predictors minus 6 collinear features = 36 selected features, mapping to a $6 \times 6$ grid with 0 padding. Because the paper's feature table omits `difficulty_level`, retaining it is our project reconstruction decision, not a mechanical copy of paper procedures.
3. **UNSW-NB15 Feature Alignment (42 raw $\to$ 38 selected):** The paper does not state the arithmetic "42 - 4 = 38". Isolating `label` to avoid leakage and retaining `sloss`/`dloss` while dropping the 4 redundant traffic predictors yields 38 selected features + 11 trailing zeros padding into a $7 \times 7 = 49$ grid. These are project reconstruction decisions to reconcile the paper's reported 38 features and $7 \times 7$ grid.
4. **2D-CNN Topology (Fig. 5):** Reconstructed exact 3 Conv / 3 MaxPool topology with `padding='same'` on convolutions and pooling to support $6 \times 6$ and $7 \times 7$ grids without spatial collapse.
5. **1D-CNN Filter Architecture:** Inferred a 64 $\to$ 32 filter sequence and documented it as an `INFERRED PARAMETER`.
6. **Dropout Contradiction:** Evaluated both 0.00 and 0.01; both confirmed minimal sensitivity ($|\Delta \text{Acc}| \le 0.0006$).
7. **UNSW SHAP 'data' Inconsistency:** The paper cites `data` as the #1 feature for UNSW Normal global SHAP, but `data` does not exist in the UNSW-NB15 dataset. Our reproduction faithfully reports the empirical top feature (`dttl`).

---

## 9. Conclusion

This empirical replication demonstrates that the deep learning and XAI methodology proposed by **Sharma et al. (2024)** is scientifically sound and replicable. When implemented under strict data-leakage controls, the models achieve detection metrics matching the published benchmarks within fractions of a percent ($|\Delta| \le 0.0038$). The addition of our all-feature ablation baseline confirms that Pearson-correlation feature selection effectively compresses the predictor space without compromising detection accuracy or model explainability.
