# UNSW-NB15 Feature Dimension & Target Separation Analysis

This document provides a rigorous, transparent breakdown of the feature dimensions and target column separation for the **UNSW-NB15** dataset in the replication of **Sharma et al. (2024)**.

---

## 1. Raw Dataset Structure

The standard UNSW-NB15 benchmark distribution consists of 45 attributes in its partitioned form (`UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`):
- `id`: Non-predictive unique row identifier (must be dropped).
- 42 network traffic features (numeric and categorical).
- `attack_cat`: Multi-class ground-truth attack category (e.g. Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms).
- `label`: Binary ground-truth target ($0 = \text{Normal}, 1 = \text{Attack}$).

In the full 4-file raw set (`UNSW-NB15_1.csv` to `_4.csv`), there are 49 attributes including source/destination IP and port addresses (`srcip`, `sport`, `dstip`, `dsport`), which are standardly omitted in machine learning evaluations to prevent network topology overfitting.

---

## 2. Five-Class Target Filtering (`UNSW-NBnew`)

Sharma et al. specify that the classification problem is conducted across **five classes**:
1. `DoS`
2. `Exploits`
3. `Fuzzers`
4. `Generic`
5. `Normal`

The original UNSW-NB15 dataset contains 10 categories (Normal + 9 attack families). To reconstruct `UNSW-NBnew`:
- Rows belonging to the 5 designated classes are retained.
- Minor attack classes (`Analysis`, `Backdoors`, `Reconnaissance`, `Shellcode`, `Worms`) are filtered out.
- The 5 target categories are encoded according to the paper's XAI mapping:
  - `0 = DoS`
  - `1 = Exploits`
  - `2 = Fuzzers`
  - `3 = Generic`
  - `4 = Normal`

---

## 3. The "label" Discrepancy & Target Leakage Prevention

In Section 3.2, Sharma et al. list the following 6 features as removed by Pearson correlation ($|PCC| > 0.95$):
1. `ct_src_dport_ltm`
2. `loss` (representing `sloss` and/or `dloss`)
3. `dwin`
4. `ct_ftp_cmd`
5. `label`
6. `ct_srv_dst`

### Critical Analysis:
- `label` is **not an input traffic predictor**; it is the ground-truth binary label of the record.
- If an IDS model were trained with `label` as an input feature, it would have direct access to whether each packet is malicious or normal ($1$ vs $0$), leading to catastrophic target leakage.
- We deduce that the original authors likely computed a correlation matrix across **all dataframe columns** (including `label`). Because `label` correlated with attack features at $|PCC| > 0.95$, it appeared in their list of dropped columns.

### Methodological Rule:
In our replication:
1. `label` and `attack_cat` are extracted as targets immediately upon data loading and **never** placed into predictor matrix $X$.
2. The remaining 5 redundant predictor features (`ct_src_dport_ltm`, `sloss`/`dloss`, `dwin`, `ct_ftp_cmd`, `ct_srv_dst`) are dropped from $X$ as specified.
3. The non-predictive `id` column is dropped.

---

## 4. Input Dimension & 2D-CNN Zero-Padding

The paper states that for the 2D-CNN, the UNSW-NB15 feature vector is **zero-padded** to fit into a $7 \times 7 = 49$ dimensional grid:
$$\text{Grid Capacity} = 7 \times 7 = 49 \text{ elements}$$

- Starting eligible traffic features: $\approx 42$ (or 39-40 numeric/categorical features after dropping `id` and redundant packet counters).
- Selected features after correlation pruning: $\approx 36\text{--}37$ features.
- Padded with zeros up to 49 features:
  $$X_{\text{padded}} = [f_1, f_2, \dots, f_k, 0, \dots, 0]$$
- Reshaped in row-major order to $(7, 7, 1)$.

This matches Section 4.3 of the paper exactly and ensures zero target leakage.
