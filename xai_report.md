# Explainable AI (XAI) Synthesis Report: SHAP Analysis

Replication of **Sharma et al. (2024)**, *“Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach”*.

---

## 1. Scope & Primary Target Model

In accordance with Section 5 of Sharma et al., the explainability evaluation centers on the **Deep Neural Network (DNN)** trained on the **selected features**:
1. **NSL-KDD:** 3-layer DNN ($64 \to 64 \to 64 \to 5$) trained on the 36 selected features.
2. **UNSW-NB15:** 3-layer DNN ($64 \to 64 \to 64 \to 5$) trained on the 38 selected features.

**SHapley Additive Explanations (SHAP)** were deployed to audit local and global prediction mechanics.

---

## 2. SHAP Global Explainability (Seed-Controlled Random Sample of 50 Test Instances)

Adhering strictly to Section 5.2 of Sharma et al. (2024), a seed-controlled random sample of **50 test instances** was evaluated to compute class-specific SHAP attributions:
- **NSL-KDD:** Target Class = **`DoS`** (Class index 0).
- **UNSW-NB15:** Target Class = **`Normal`** (Class index 4).

Features are ranked by **Mean Absolute SHAP Value** ($mean(|SHAP|)$) for each respective target class.

### 2.1 NSL-KDD SHAP Global Importance Ranking (Target Class: DoS)

| Rank | Feature Name | Mean Absolute SHAP ($mean(|SHAP|)$) |
| :---: | :--- | :---: |
| 1 | `serror_rate` | 0.12886 |
| 2 | `logged_in` | 0.06740 |
| 3 | `dst_host_same_src_port_rate` | 0.04317 |
| 4 | `dst_host_srv_count` | 0.02940 |
| 5 | `protocol_type` | 0.02803 |
| 6 | `count` | 0.02521 |
| 7 | `dst_host_rerror_rate` | 0.02345 |
| 8 | `difficulty_level` | 0.01854 |
| 9 | `wrong_fragment` | 0.01450 |
| 10 | `flag` | 0.01254 |

### 2.2 UNSW-NB15 SHAP Global Importance Ranking (Target Class: Normal)

| Rank | Feature Name | Mean Absolute SHAP ($mean(|SHAP|)$) |
| :---: | :--- | :---: |
| 1 | `dttl` | 0.17550 |
| 2 | `swin` | 0.16447 |
| 3 | `sttl` | 0.06162 |
| 4 | `ct_dst_sport_ltm` | 0.05263 |
| 5 | `ct_state_ttl` | 0.02248 |
| 6 | `ct_dst_src_ltm` | 0.01748 |
| 7 | `dmean` | 0.01300 |
| 8 | `service` | 0.01058 |
| 9 | `smean` | 0.00802 |
| 10 | `state` | 0.00735 |
