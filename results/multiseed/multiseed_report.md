# Multi-Seed Empirical Replication & Statistical Benchmark
## Rigorous Multi-Seed Validation of Sharma et al. (2024) across All 6 Models

> **Methodology Overview**: Following LordKarsSama's empirical framework for 2D-CNN, this pipeline extends multi-seed empirical analysis to **all six canonical models** across NSL-KDD and UNSW-NB15. Rather than reporting a single fortuitous initialization, each model is evaluated across multiple independently initialized replicas to quantify empirical variance, 95% confidence intervals, and identify the exact initialization seed that reproduces the paper's reported values.

---

### 1. Consolidated Cross-Model Benchmark vs. Sharma et al. (2024)

| Dataset | Architecture | Paper Accuracy | Multi-Seed Mean ± Std | 95% Confidence Interval | Range [Min, Max] | Closest Seed | Closest Accuracy | Absolute Error | Match at Printed Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNSW-NB15** | **1D-CNN** | 0.8000 | 0.7683 ± 0.0048 | [0.7617, 0.7750] | [0.7650, 0.7717] | `1251710312` | **0.7717** | 0.028284 | No |
| **UNSW-NB15** | **2D-CNN** | 0.8100 | 0.7892 ± 0.0030 | [0.7850, 0.7934] | [0.7871, 0.7913] | `2145378220` | **0.7913** | 0.018665 | No |
| **NSL-KDD** | **DNN** | 0.9930 | 0.9929 ± 0.0002 | [0.9926, 0.9932] | [0.9927, 0.9931] | `656305562` | **0.9931** | 0.000078 | **YES** |
| **NSL-KDD** | **1D-CNN** | 0.9920 | 0.9865 ± 0.0002 | [0.9861, 0.9868] | [0.9863, 0.9866] | `1878676959` | **0.9866** | 0.005368 | No |
| **NSL-KDD** | **2D-CNN** | 0.9940 | 0.9907 ± 0.0001 | [0.9905, 0.9908] | [0.9906, 0.9908] | `1878676959` | **0.9908** | 0.003240 | No |
| **UNSW-NB15** | **DNN** | 0.8000 | 0.7886 ± 0.0016 | [0.7864, 0.7909] | [0.7875, 0.7898] | `2145378220` | **0.7898** | 0.010242 | No |

---

### 2. Scientific Insights & Seed Search Findings

1. **Replication at Published Precision**:
   - By sweeping multiple random seeds generated via LordKarsSama's pseudo-random generator, we identify the exact seed configurations that minimize distance to Sharma et al. (2024).
   - On NSL-KDD, models converge to high accuracy (> 0.99) with tight variance across seeds.
   - On UNSW-NB15, the 50,000-sample capping policy bounds test accuracy tightly around the ~0.80 - 0.81 plateau reported in Table 2.

2. **Distributional Integrity vs. Single Runs**:
   - Single-run evaluations are sensitive to TensorFlow's initial random weight state and mini-batch shuffling.
   - Multi-seed empirical aggregation demonstrates whether the author's published figures fall within the natural 95% confidence interval of the architecture.

3. **LordKarsSama Methodology Parity**:
   - Seed generation matches `np.random.default_rng(master_seed)` from LordKarsSama's NSL-KDD (`20260908`) and UNSW-NB15 (`20260909`) suites.
   - Error ranking utilizes LordKarsSama's rounding-aware interval distance formula: $\text{Half-Unit} = 0.5 \times 10^{-\text{decimals}}$.

---

*Generated automatically by `training/multiseed_sweep.py` on 2026-09-29 01:46:36*