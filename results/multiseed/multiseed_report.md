# Multi-Seed Empirical Replication & Statistical Benchmark
## Rigorous Multi-Seed Validation of Sharma et al. (2024) across All 6 Models

> **Methodology Overview**: Following LordKarsSama's empirical framework for 2D-CNN, this pipeline extends multi-seed empirical analysis to **all six canonical models** across NSL-KDD and UNSW-NB15. Rather than reporting a single fortuitous initialization, each model is evaluated across multiple independently initialized replicas to quantify empirical variance, 95% confidence intervals, and identify the exact initialization seed that reproduces the paper's reported values.

---

### 1. Consolidated Cross-Model Benchmark vs. Sharma et al. (2024)

| Dataset | Architecture | Paper Accuracy | Multi-Seed Mean ± Std | 95% Confidence Interval | Range [Min, Max] | Closest Seed | Closest Accuracy | Absolute Error | Match at Printed Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NSL-KDD** | **DNN** | 0.9930 | 0.9876 ± 0.0010 | [0.9873, 0.9879] | [0.9839, 0.9892] | `36` | **0.9892** | 0.003800 | No |
| **NSL-KDD** | **1D-CNN** | 0.9920 | 0.9823 ± 0.0024 | [0.9817, 0.9828] | [0.9723, 0.9853] | `41` | **0.9853** | 0.006732 | No |
| **NSL-KDD** | **2D-CNN** | 0.9940 | 0.9966 ± 0.0005 | [0.9960, 0.9971] | [0.9960, 0.9970] | `2064790833` | **0.9960** | 0.001999 | No |
| **UNSW-NB15** | **DNN** | 0.8000 | 0.8122 ± 0.0017 | [0.8118, 0.8127] | [0.8054, 0.8155] | `13` | **0.8054** | 0.005389 | No |
| **UNSW-NB15** | **1D-CNN** | 0.8000 | 0.8052 ± 0.0044 | [0.8041, 0.8063] | [0.7926, 0.8107] | `34` | **0.7998** | 0.000212 | **YES** |
| **UNSW-NB15** | **2D-CNN** | 0.8100 | 0.7892 ± 0.0030 | [0.7850, 0.7934] | [0.7871, 0.7913] | `2145378220` | **0.7913** | 0.018665 | No |

---

### 2. Scientific Insights & Seed Search Findings

1. **Replication at Published Precision**:
   - By sweeping multiple random seeds generated via LordKarsSama's pseudo-random generator, we identify the exact seed configurations that minimize distance to Sharma et al. (2024).
   - On NSL-KDD, models converge to high accuracy (> 0.99) with tight variance across seeds.
   - On UNSW-NB15, the 50,000-sample capping policy bounds test accuracy tightly around the ~0.80 - 0.81 plateau reported in Table 2.

2. **Distributional Integrity vs. Single Runs**:
   - Single-run evaluations are sensitive to initial random weight state and mini-batch shuffling.
   - Multi-seed empirical aggregation demonstrates whether the author's published figures fall within the natural 95% confidence interval of the architecture.

3. **LordKarsSama Methodology Parity**:
   - Seed generation matches `np.random.default_rng(master_seed)` from LordKarsSama's NSL-KDD (`20260908`) and UNSW-NB15 (`20260909`) suites.
   - Error ranking utilizes LordKarsSama's rounding-aware interval distance formula: $\text{Half-Unit} = 0.5 \times 10^{-\text{decimals}}$.

---

*Generated automatically by `training/multiseed_sweep.py` on 2026-09-29 09:50:04*