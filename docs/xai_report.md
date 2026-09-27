# Explainable AI (XAI) Synthesis Report: LIME & SHAP Analysis

Replication of **Sharma et al. (2024)**, *“Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach”*.

---

## 1. Scope & Primary Target Model

In accordance with Section 5 of Sharma et al., the explainability evaluation centers on the **Deep Neural Network (DNN)** trained on the **selected features**:
1. **NSL-KDD:** 3-layer DNN ($64 \to 64 \to 64 \to 5$) trained on the 36 selected features.
2. **UNSW-NB15:** 3-layer DNN ($64 \to 64 \to 64 \to 5$) trained on the 38 selected features.

Both **Local Interpretable Model-agnostic Explanations (LIME)** and **SHapley Additive Explanations (SHAP)** were deployed to audit local and global prediction mechanics.

---

## 2. SHAP Global Explainability (Seed-Controlled Random Sample of 50 Test Instances)

Adhering strictly to Section 5.2 of Sharma et al. (2024), a seed-controlled random sample of **50 test instances** was evaluated to compute class-specific SHAP attributions:
- **NSL-KDD:** Target Class = **`DoS`** (Class index 0).
- **UNSW-NB15:** Target Class = **`Normal`** (Class index 4).

Features are ranked by **Mean Absolute SHAP Value** ($mean(|SHAP|)$) for each respective target class.

### 2.1 NSL-KDD SHAP Global Importance Ranking (Target Class: DoS)

| Rank | Feature Name | Mean Absolute SHAP ($mean(|SHAP|)$) |
| :---: | :--- | :---: |
| 1 | `serror_rate` | 0.13955 |
| 2 | `logged_in` | 0.05795 |
| 3 | `dst_host_same_src_port_rate` | 0.03883 |
| 4 | `dst_host_srv_count` | 0.03649 |
| 5 | `count` | 0.02537 |
| 6 | `protocol_type` | 0.02327 |
| 7 | `dst_host_rerror_rate` | 0.02241 |
| 8 | `difficulty_level` | 0.01880 |
| 9 | `srv_count` | 0.01328 |
| 10 | `wrong_fragment` | 0.01308 |

### 2.2 UNSW-NB15 SHAP Global Importance Ranking (Target Class: Normal)

| Rank | Feature Name | Mean Absolute SHAP ($mean(|SHAP|)$) |
| :---: | :--- | :---: |
| 1 | `dttl` | 0.15098 |
| 2 | `swin` | 0.11924 |
| 3 | `sttl` | 0.07591 |
| 4 | `ct_dst_sport_ltm` | 0.05688 |
| 5 | `ct_state_ttl` | 0.02169 |
| 6 | `dmean` | 0.01636 |
| 7 | `ct_dst_src_ltm` | 0.01604 |
| 8 | `ct_srv_src` | 0.01172 |
| 9 | `smean` | 0.01161 |
| 10 | `service` | 0.00977 |

---

## 3. LIME Local Explanations

Representative test instances were audited using LIME TabularExplainer:
- **NSL-KDD:** DoS attack instance and Normal instance.
- **UNSW-NB15:** Normal traffic instance and Exploits attack instance.

High-resolution contribution plots and structured JSON explanations are archived in `results/xai/lime/` and `results/xai/shap/`.
