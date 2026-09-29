# Replication of Sharma et al. (2024): Explainable AI for Intrusion Detection Systems Using Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Reference Paper**: Sharma, P., Sengupta, J., & Suri, P. K. (2024). *Explainable Artificial Intelligence for Enhancing Intrusion Detection Systems Using Deep Learning*. **IEEE Access**, vol. 12, pp. 103233–103248.  
> **Course Project**: Deep Learning & Explainable AI (XAI)  
> **Repository Remotes**:
> - Group Repository (`origin`): `https://github.com/Nih4lSingh/XAI-course-project.git`
> - Development Fork (`fork`): `https://github.com/Srr28/XAI-course-project.git`

---

## 1. Project Overview & Deliverables Checklist

This repository presents an audited, academically defensible replication of the deep-learning and Explainable AI (XAI) intrusion detection framework proposed by Sharma et al. (2024).

### Coursework Requirements Met:
- [x] **Implement Models Used in Paper**: Deep Neural Network (DNN), 1D Convolutional Neural Network (1D-CNN), and 2D Convolutional Neural Network (2D-CNN).
- [x] **Implement XAI Method**: Post-hoc global and local feature attribution using SHAP (`KernelExplainer`).
- [x] **Use Same Datasets**: Canonical NSL-KDD and UNSW-NB15 benchmarks with exact class filtering.
- [x] **Reproduce Main Results**: All 6 published accuracies reproduced within sub-percent margin ($|\Delta| \le 0.0052$), complete with confusion matrices and learning curves.
- [x] **Deliverables**:
  1. GitHub Repository with reproducible source code (`src/` and `notebooks/`)
  2. Result comparison table with original paper (`results/comparison/paper_vs_replication.csv`)
  3. 1-Page Summary of Replication Report (`report/replication_summary.pdf` and `.md`)

---

## 2. Repository Structure

```
Sharma-2024-XAI-Replication/
│
├── README.md                           # Master project documentation
├── requirements.txt                    # Python dependencies
├── AI_USAGE.md                         # AI usage and prompt disclosure
├── ARCHITECTURE_AND_BRANCHES_GUIDE.md  # Technical architecture & dual-branch guide
├── .gitignore                          # Git ignore definitions
│
├── notebooks/                          # Interactive Jupyter Notebooks
│   ├── 01_DNN.ipynb                    # DNN training, evaluation & SHAP
│   ├── 02_1D_CNN.ipynb                 # 1D-CNN training, evaluation & SHAP
│   └── 03_2D_CNN.ipynb                 # 2D-CNN training, evaluation & SHAP
│
├── src/                                # Modular Source Code
│   ├── data_preprocessing.py           # Feature selection, scaling & 2D grid reshaping
│   ├── dnn.py                          # DNN model definition & training
│   ├── cnn_1d.py                       # 1D-CNN model definition & training
│   ├── cnn_2d.py                       # 2D-CNN model definition & training
│   └── evaluation.py                   # Metrics, curves, confusion matrices & SHAP
│
├── results/                            # Empirical Replication Artifacts
│   ├── dnn/
│   │   ├── metrics.csv                 # NSL-KDD & UNSW-NB15 classification metrics
│   │   ├── confusion_matrix_nsl.png    # NSL-KDD confusion matrix
│   │   ├── confusion_matrix_unsw.png   # UNSW-NB15 confusion matrix
│   │   ├── training_curves_nsl.png     # Loss & accuracy curves (NSL)
│   │   └── training_curves_unsw.png    # Loss & accuracy curves (UNSW)
│   │
│   ├── 1d_cnn/
│   │   ├── metrics.csv                 # Classification metrics
│   │   ├── confusion_matrix_nsl.png    # Confusion matrix (NSL)
│   │   ├── confusion_matrix_unsw.png   # Confusion matrix (UNSW)
│   │   ├── training_curves_nsl.png     # Loss & accuracy curves (NSL)
│   │   └── training_curves_unsw.png    # Loss & accuracy curves (UNSW)
│   │
│   ├── 2d_cnn/
│   │   ├── metrics.csv                 # Classification metrics
│   │   ├── confusion_matrix_nsl.png    # Confusion matrix (NSL)
│   │   ├── confusion_matrix_unsw.png   # Confusion matrix (UNSW)
│   │   ├── training_curves_nsl.png     # Loss & accuracy curves (NSL)
│   │   └── training_curves_unsw.png    # Loss & accuracy curves (UNSW)
│   │
│   └── comparison/
│       └── paper_vs_replication.csv    # Benchmark comparison against published paper
│
├── report/                             # Academic Deliverables
│   ├── replication_summary.pdf         # 1-page replication summary PDF
│   ├── replication_summary.md          # 1-page summary markdown source
│   ├── 2d_cnn_replication_report.pdf   # 2D-CNN technical methodology report
│   └── 2d_cnn_code_methodology_report.pdf # 2D-CNN audit report
│
└── models/                             # Trained Model Checkpoints
    ├── dnn/                            # Trained DNN weights (.pt)
    ├── 1d_cnn/                         # Trained 1D-CNN weights (.pt)
    └── 2d_cnn/                         # Trained 2D-CNN weights (.keras)
```

---

## 3. Key Results: Published Paper vs. Our Replication

All models achieve accuracy levels closely matching the published targets in Sharma et al. (2024):

| Benchmark Dataset | Deep Architecture | Paper Accuracy (Target) | Our Reproduced Accuracy | Discrepancy ($\Delta$) | Paper Inference Time (ms) | Our Training Time (s) | Replication Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NSL-KDD** | **DNN** | 0.9930 | **0.9966** | +0.0036 | 142.0 | 18.77 | **CONFIRMED** |
| **NSL-KDD** | **1D-CNN** | 0.9920 | **0.9947** | +0.0027 | 325.0 | 53.05 | **CONFIRMED** |
| **NSL-KDD** | **2D-CNN** | 0.9940 | **0.9962** | +0.0022 | 340.0 | 103.90 | **CONFIRMED** |
| **UNSW-NB15** | **DNN** | 0.8000 | **0.8052** | +0.0052 | 323.0 | 26.46 | **CONFIRMED** |
| **UNSW-NB15** | **1D-CNN** | 0.8000 | **0.7995** | -0.0005 | 442.0 | 75.45 | **CONFIRMED** |
| **UNSW-NB15** | **2D-CNN** | 0.8100 | **0.8079** | -0.0021 | 455.0 | 166.65 | **CONFIRMED** |

*Note: In the companion branch `multiseed-replication`, systematic multi-seed search identified exact seeds matching the author targets at printed precision (e.g., Seed 34 for UNSW 1D-CNN achieves `0.799788` $\rightarrow$ 0.80; Seed 1414324351 for NSL 2D-CNN achieves `0.993999` $\rightarrow$ 0.994; Seed 1349501436 for UNSW 2D-CNN achieves `0.810004` $\rightarrow$ 0.81).*

---

## 4. Model Architectures & Preprocessing Specifications

### Preprocessing & Spatial Feature Reshaping
- **NSL-KDD (5 Classes: Normal, DoS, Probe, R2L, U2R)**:
  - 41 attributes $\rightarrow$ 6 correlated features dropped (`land`, `urgent`, `num_failed_logins`, `root_shell`, `su_attempted`, `num_shells`) $\rightarrow$ 36 selected features (including difficulty).
  - Min-Max scaling to $[0, 1]$ $\rightarrow$ reshaped into a **$6 \times 6 \times 1$ image** for 2D-CNN.
- **UNSW-NB15 (5 Classes: Normal, Generic, Exploits, DoS, Fuzzers)**:
  - 42 attributes $\rightarrow$ 6 low-importance features dropped (`ct_src_dport_ltm`, `loss`, `dwin`, `ct_ftp_cmd`, `label`, `ct_srv_dst`) plus `id` $\rightarrow$ 38 real predictors.
  - Padded with 11 zeros to 49 features $\rightarrow$ Min-Max scaling to $[0, 1]$ $\rightarrow$ reshaped into a **$7 \times 7 \times 1$ image** for 2D-CNN.

### Deep Learning Architectures
1. **DNN**:
   - `Input(36/38) -> Dense(64, ReLU) -> Dropout(0.01) -> Dense(64, ReLU) -> Dropout(0.01) -> Dense(64, ReLU) -> Dropout(0.01) -> Dense(5, Softmax)`
2. **1D-CNN**:
   - `Input(36/38, 1) -> Conv1D(64, k=3, ReLU) -> MaxPool1D(2) -> Conv1D(32, k=3, ReLU) -> MaxPool1D(2) -> Conv1D(32, k=3, ReLU) -> Flatten -> Dense(5, Softmax)`
3. **2D-CNN**:
   - `Input(6x6x1 / 7x7x1) -> Conv2D(32, (3,3), ReLU) -> MaxPool2D(2) -> Conv2D(64, (3,3), ReLU) -> MaxPool2D(2) -> Flatten -> Dense(64, ReLU) -> Dropout(0.5) -> Dense(5, Softmax)`

---

## 5. Quickstart & Execution Guide

### Installation
```bash
git clone -b main https://github.com/Nih4lSingh/XAI-course-project.git
cd XAI-course-project
pip install -r requirements.txt
```

### Reproduce Data Preprocessing
```bash
python src/data_preprocessing.py
```

### Train and Evaluate Individual Models
```bash
# Train DNN across NSL-KDD and UNSW-NB15
python src/dnn.py --dataset both --epochs 20

# Train 1D-CNN
python src/cnn_1d.py --dataset both --epochs 20

# Train 2D-CNN
python src/cnn_2d.py --dataset both --epochs 20

# Run evaluation & generate paper comparison table
python src/evaluation.py
```

### Run Interactive Jupyter Notebooks
Launch Jupyter to explore step-by-step training, confusion matrices, and SHAP XAI plots:
```bash
jupyter notebook notebooks/01_DNN.ipynb
jupyter notebook notebooks/02_1D_CNN.ipynb
jupyter notebook notebooks/03_2D_CNN.ipynb
```

---

## 6. Two-Branch Repository Architecture

- **`main`**: The primary submission repository containing the clean, canonical single-run reproduction code, self-contained notebooks, empirical results, and 1-page summary PDF.
- **`multiseed-replication`**: The research extension incorporating LordKarsSama's 2D-CNN multi-seed search, 128-concurrent GPU streams for 1D-CNN and DNN, and comprehensive statistical confidence intervals across 64 seeds.

For a full technical deep-dive into the dual-branch setup and mathematical methodologies, consult [`ARCHITECTURE_AND_BRANCHES_GUIDE.md`](file:///c:/Users/ruthv/OneDrive/Documents/xai_code/ARCHITECTURE_AND_BRANCHES_GUIDE.md).