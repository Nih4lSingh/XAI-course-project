# Multi-Seed Empirical Replication Framework

This directory contains the multi-seed empirical replication framework extending the methodology introduced by LordKarsSama across **all six canonical models** reported by Sharma et al. (2024).

---

## 1. Why Multi-Seed Evaluation?

Sharma et al. (2024) omitted several operational hyperparameters, including:
1. **Random Seed**: Initialization seed for model weights and train/val/test splits.
2. **Mini-Batch Size**: Omitted in the text (commonly set between 32 and 128).
3. **Weight Decay Semantics**: Adam with L2 regularizer vs. AdamW decoupled decay.

In single-run benchmarks, performance is sensitive to the fortuitous alignment of weight initialization and mini-batch ordering. To test the hypothesis that published metrics reflect specific favorable seeds (or to determine whether the architecture consistently delivers the published metrics across replications), this framework evaluates multiple independent seeds and provides rigorous statistical aggregations.

---

## 2. Supported Models & Benchmark Targets

| Dataset | Architecture | Paper Accuracy (Target) | Paper Decimals | Target Table in Sharma et al. (2024) |
| :--- | :--- | :---: | :---: | :--- |
| **NSL-KDD** | DNN | 0.9930 | 3 | Table 1 |
| **NSL-KDD** | 1D-CNN | 0.9920 | 3 | Table 1 |
| **NSL-KDD** | 2D-CNN | 0.9940 | 3 | Table 1 |
| **UNSW-NB15** | DNN | 0.8000 | 2 | Table 2 |
| **UNSW-NB15** | 1D-CNN | 0.8000 | 2 | Table 2 |
| **UNSW-NB15** | 2D-CNN | 0.8100 | 2 | Table 2 |

---

## 3. Mathematical Methodology

### A. Pseudo-Random Seed Generation
Seeds are generated using NumPy's modern PCG64 pseudo-random generator initialized with fixed master seeds (matching LordKarsSama's protocol):
- **NSL-KDD Master Seed**: `20260908`
- **UNSW-NB15 Master Seed**: `20260909`

$$S = \text{choice}(\{1, \dots, 2^{31}-2\}, N, \text{replace}=\text{False})$$

### B. Statistical Aggregation
For $N$ seed evaluations $\{x_1, x_2, \dots, x_N\}$:
- **Sample Mean**:
  $$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$$
- **Sample Standard Deviation**:
  $$s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}$$
- **95% Confidence Interval**:
  $$\text{CI}_{95\%} = \left[ \bar{x} - 1.96 \cdot \frac{s}{\sqrt{N}}, \; \bar{x} + 1.96 \cdot \frac{s}{\sqrt{N}} \right]$$

### C. Distance to Published Targets
- **Raw Error**: $e_{\text{raw}} = x_s - T$
- **Absolute Error**: $e_{\text{abs}} = |x_s - T|$
- **Rounding-Aware Error** (LordKarsSama distance):
  $$\text{half\_unit} = 0.5 \times 10^{-d}$$
  $$e_{\text{round}} = \begin{cases} (T - \text{half\_unit}) - x_s & \text{if } x_s < T - \text{half\_unit} \\ x_s - (T + \text{half\_unit}) & \text{if } x_s > T + \text{half\_unit} \\ 0 & \text{otherwise} \end{cases}$$
- **Printed Precision Match**: $\text{round}(x_s, d) == \text{round}(T, d)$

---

## 4. How to Run

### Quick Smoke-Test (2 seeds, 2 epochs)
```powershell
python run_multiseed.py --quick
```

### Run 10-Seed Sweep on All 6 Models
```powershell
python run_multiseed.py --all --num_seeds 10
```

### Run a Specific Model (e.g. NSL-KDD 2D-CNN)
```powershell
python run_multiseed.py --exp_id NSL_SELECTED_2DCNN --num_seeds 20 --batch_size 128
```

### Run with Specific Explicit Seeds
```powershell
python run_multiseed.py --exp_id NSL_SELECTED_DNN --seeds 42,20240901,1414324351
```

### Regenerate Master Comparison Report without Retraining
```powershell
python run_multiseed.py --report_only
```

---

## 5. Artifacts Produced

For each model `<exp_id>`:
- `results/multiseed/<exp_id>/seed_runs.csv`: Complete per-seed metrics (Accuracy, Loss, Macro/Weighted F1, Per-class metrics, Error metrics).
- `results/multiseed/<exp_id>/ranked_seeds.csv`: Seeds sorted by proximity to paper targets.
- `results/multiseed/<exp_id>/summary_statistics.json`: Machine-readable statistical summary.

Consolidated cross-model artifacts:
- `results/multiseed/multiseed_paper_comparison.csv`: Master side-by-side benchmark table across all models.
- `results/multiseed/multiseed_report.md`: Complete publication-ready Markdown report.

---

## 6. 64-Seed GPU Empirical Replication (1D-CNN & DNN)

Using `Sharma_2024_1D_CNN_Replication_GPU_128concurrent.py` (which ports the model matrix across 64 seeds concurrently on GPU streams), the empirical distribution across 64 independent weight initializations is:

| Architecture | Dataset | Paper Target | 64-Seed Mean ± Std | 95% Confidence Interval | Best Matching Seed | Best Seed Accuracy | Match at Printed Precision? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DNN** | NSL-KDD | **0.9930** | 0.9876 ± 0.0010 | [0.9873, 0.9879] | Seed 36 | **0.989200** | No |
| **DNN** | UNSW-NB15 | **0.8000** | 0.8122 ± 0.0017 | [0.8118, 0.8127] | Seed 13 | **0.805389** | No |
| **1D-CNN** | NSL-KDD | **0.9920** | 0.9823 ± 0.0024 | [0.9817, 0.9828] | Seed 41 | **0.985268** | No |
| **1D-CNN** | UNSW-NB15 | **0.8000** | 0.8052 ± 0.0044 | [0.8041, 0.8063] | Seed 34 | **0.799788** | **YES** (rounds to 0.80) |
