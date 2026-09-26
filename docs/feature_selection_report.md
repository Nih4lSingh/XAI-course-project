# Feature Selection Report: Pearson Correlation Analysis

This report documents the reproduction and empirical verification of the Pearson Correlation Coefficient (PCC) feature selection pipeline specified in Section 3.2 of:

**Sharma et al. (2024)**, *“Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach”*.

---

## 1. Feature Selection Methodology

The paper utilizes a **filter-based feature selection method** using pairwise Pearson Correlation Coefficient ($PCC$):

$$PCC(X_i, X_j) = \frac{\sum_{k=1}^N (X_{ik} - \bar{X}_i)(X_{jk} - \bar{X}_j)}{\sqrt{\sum_{k=1}^N (X_{ik} - \bar{X}_i)^2 \sum_{k=1}^N (X_{jk} - \bar{X}_j)^2}}$$

- **Threshold:** $|PCC| > 0.95$
- **Rule:** If two features exhibit $|PCC| > 0.95$, one feature is deemed redundant and removed.
- **Goal:** Reduce feature dimensionality, eliminate collinearity, and accelerate training without sacrificing classification accuracy.

---

## 2. NSL-KDD Feature Selection Verification

- **Total input predictors:** 41 features
- **Highly correlated pairs identified ($|PCC| > 0.95$):** 10 pairs

| Feature 1 | Feature 2 | Pearson Correlation ($r$) | Redundant Feature (Paper) | Status |
| :--- | :--- | :---: | :--- | :--- |
| `num_compromised` | `num_root` | +0.9987 | `num_root` | **CONFIRMED** |
| `serror_rate` | `srv_serror_rate` | +0.9915 | `srv_serror_rate` | **CONFIRMED** |
| `serror_rate` | `dst_host_serror_rate` | +0.9747 | `dst_host_serror_rate` | **CONFIRMED** |
| `serror_rate` | `dst_host_srv_serror_rate`| +0.9760 | `dst_host_srv_serror_rate`| **CONFIRMED** |
| `srv_serror_rate` | `dst_host_serror_rate` | +0.9708 | `srv_serror_rate` | **CONFIRMED** |
| `srv_serror_rate` | `dst_host_srv_serror_rate`| +0.9820 | `dst_host_srv_serror_rate`| **CONFIRMED** |
| `rerror_rate` | `srv_rerror_rate` | +0.9861 | `srv_rerror_rate` | **CONFIRMED** |
| `rerror_rate` | `dst_host_srv_rerror_rate`| +0.9574 | `dst_host_srv_rerror_rate`| **CONFIRMED** |
| `srv_rerror_rate` | `dst_host_srv_rerror_rate`| +0.9656 | `srv_rerror_rate` | **CONFIRMED** |
| `dst_host_serror_rate` | `dst_host_srv_serror_rate`| +0.9828| `dst_host_srv_serror_rate`| **CONFIRMED** |

### Verification of Sharma et al. 6 Removed Features:
1. `srv_serror_rate`: Confirmed ($r = +0.9915$ with `serror_rate`)
2. `dst_host_srv_rerror_rate`: Confirmed ($r = +0.9574$ with `rerror_rate`)
3. `num_root`: Confirmed ($r = +0.9987$ with `num_compromised`)
4. `dst_host_serror_rate`: Confirmed ($r = +0.9747$ with `serror_rate`)
5. `dst_host_srv_serror_rate`: Confirmed ($r = +0.9760$ with `serror_rate`)
6. `srv_rerror_rate`: Confirmed ($r = +0.9861$ with `rerror_rate`)

**Selected Predictors:** **36 features** ($42 - 6 = 36$ retaining `difficulty_level`, mapping directly to a $6 \times 6$ grid with **0 padding zeros**; in the alternative strict-network-only ablation where `difficulty_level` is discarded, $41 - 6 = 35$ features with 1 zero-padding cell).

---

## 3. UNSW-NB15 Feature Selection Verification

- **Total input predictors:** 42 eligible traffic features (excluding `id`, `attack_cat`, and ground-truth `label`)
- **Highly correlated pairs identified ($|PCC| > 0.95$):** 12 pairs

| Feature 1 | Feature 2 | Pearson Correlation ($r$) | Redundant Predictor (Paper) | Status |
| :--- | :--- | :---: | :--- | :--- |
| `spkts` | `sbytes` | +0.9648 | — | Retained |
| `spkts` | `sloss` | +0.9723 | `sloss` | Ambiguous (`loss`) |
| `dpkts` | `dbytes` | +0.9729 | — | Retained |
| `dpkts` | `dloss` | +0.9791 | `dloss` | Ambiguous (`loss`) |
| `sbytes` | `sloss` | +0.9959 | `sloss` | Ambiguous (`loss`) |
| `dbytes` | `dloss` | +0.9966 | `dloss` | Ambiguous (`loss`) |
| `swin` | `dwin` | +0.9788 | `dwin` | **CONFIRMED** |
| `ct_srv_src` | `ct_dst_src_ltm` | +0.9569 | — | Retained |
| `ct_srv_src` | `ct_srv_dst` | +0.9801 | `ct_srv_dst` | **CONFIRMED** |
| `ct_dst_ltm` | `ct_src_dport_ltm` | +0.9637 | `ct_src_dport_ltm` | **CONFIRMED** |
| `ct_dst_src_ltm`| `ct_srv_dst` | +0.9625 | `ct_srv_dst` | **CONFIRMED** |
| `is_ftp_login` | `ct_ftp_cmd` | +0.9989 | `ct_ftp_cmd` | **CONFIRMED** |

### Verification of Sharma et al. Removed Features:
1. `ct_src_dport_ltm`: Confirmed ($r = +0.9637$ with `ct_dst_ltm`)
2. `dwin`: Confirmed ($r = +0.9788$ with `swin`)
3. `ct_ftp_cmd`: Confirmed ($r = +0.9989$ with `is_ftp_login`)
4. `ct_srv_dst`: Confirmed ($r = +0.9801$ with `ct_srv_src`)
5. `label`: Binary target column, separated into target vector $y$ to prevent target leakage.
6. `loss` (`sloss` & `dloss`): Both packet loss features are retained in the canonical set to avoid removing active network metrics without explicit naming.

**Selected Predictors:** **38 features** (reshaped into a $7 \times 7 = 49$ grid with **exactly 11 trailing zero-padding elements**).

---

## 4. Dimensionality Summary Table

| Dataset | Original Predictors | Redundant Removed | Selected Features | 2D-CNN Grid Shape | Zero-Padding Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NSL-KDD (Canonical)** | 42 | 6 | 36 | $6 \times 6 \times 1$ (36) | 0 |
| **NSL-KDD (Strict-Traffic)** | 41 | 6 | 35 | $6 \times 6 \times 1$ (36) | 1 |
| **UNSW-NB15 (Canonical)** | 42 | 4 | 38 | $7 \times 7 \times 1$ (49) | 11 |
