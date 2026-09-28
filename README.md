# Explainable AI — Sharma et al. (2024) Replication

Replication of the experiments from:

> Sharma et al. (2024), **"Explainable Artificial Intelligence for Intrusion Detection in IoT Networks: A Deep Learning Based Approach"**, *Expert Systems with Applications*.

This repository contains the Phase 1 replication pipeline for **NSL-KDD** and **UNSW-NB15**, including the DNN and 1D-CNN models and SHAP-based explainability.

---

## 1. Project Contents

The main components of the repository are:

```text
.
├── preprocessing.py
├── dnn_architecture.py
├── 1d_cnn_architecture.py
├── requirements.txt
├── AI_USAGE.md
├── README.md
└── notebooks/
    ├── DNN replication notebook
    └── 1D-CNN replication notebook
```

The preprocessing pipeline is shared by the DNN and 1D-CNN experiments.

---

## 2. Dataset

The experiments use:

- **NSL-KDD**
- **UNSW-NB15**

### NSL-KDD

Dataset information:

https://www.unb.ca/cic/datasets/nsl.html

The pipeline uses the `KDDTrain+` and `KDDTest+` files.

### UNSW-NB15

Dataset information:

https://research.unsw.edu.au/projects/unsw-nb15-dataset

The pipeline uses:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

> The datasets are not included in this repository. Download them from the respective sources and place the files at the paths specified in `preprocessing.py`.

---

## 3. Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- SHAP
- Matplotlib
- Seaborn
- Jupyter / IPython

---

## 4. Reproducibility

A fixed random seed is used throughout the preprocessing and experimental pipeline.

```text
Random seed = 42
```

The seed is used for Python/NumPy reproducibility and for the train/validation split.

The project uses:

```text
Train        : 60%
Validation   : 15%
Test         : 25%
```

The validation split is created from the original training portion using a fixed `random_state=42`.

---

## 5. Preprocessing

The same preprocessing pipeline is used for both the **DNN** and **1D-CNN** so that the model comparison is performed on the same processed data.

The main steps are:

1. Load the dataset.
2. Remove the columns specified for the respective dataset.
3. Encode categorical features.
4. Encode the five target classes.
5. Handle missing and infinite values.
6. Normalize the numerical feature values using Min-Max scaling.
7. Select the required number of features.
8. Split the data into training, validation, and test sets.

Feature dimensions used by the pipeline:

| Dataset | Selected Features |
|---|---:|
| NSL-KDD | 36 |
| UNSW-NB15 | 38 |

The exact feature-selection code used in this replication is documented as an implementation assumption because the paper does not provide the authors' complete feature-selection code.

---

## 6. Running the Pipeline

### Option 1 — Google Colab

1. Open the corresponding notebook in Google Colab.
2. Upload the required dataset files.
3. Make sure the files are available at:

```text
/content/KDDTrain+.txt
/content/KDDTest+.txt
/content/UNSW_NB15_training-set.csv
/content/UNSW_NB15_testing-set.csv
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

5. Run the notebook from top to bottom.

### Option 2 — Local Python Environment

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Update the dataset paths in `preprocessing.py` if necessary, then run the relevant training script/notebook.

---

## 7. Model Configurations

### DNN

The DNN uses three hidden layers:

```text
Input
  ↓
Dense(64, ReLU)
  ↓
Dense(64, ReLU)
  ↓
Dense(64, ReLU)
  ↓
Dense(5, Softmax)
```

Training configuration:

```text
Optimizer       : Adam
Learning rate   : 0.001
Weight decay    : 0.0001
Epochs          : 20
Batch size      : 128
Output classes  : 5
```

### 1D-CNN

The 1D-CNN uses:

```text
Input
  ↓
Conv1D(64, kernel=3, ReLU)
  ↓
MaxPooling1D(2)
  ↓
Conv1D(32, kernel=3, ReLU)
  ↓
MaxPooling1D(2)
  ↓
Conv1D(32, kernel=3, ReLU)
  ↓
MaxPooling1D(2)
  ↓
Flatten
  ↓
Dense(5, Softmax)
```

For the 1D-CNN, the preprocessed feature matrix is reshaped from:

```text
(samples, features)
```

to:

```text
(samples, features, 1)
```

---

## 8. Results Summary

The following are the results obtained from the current replication pipeline.

### Accuracy

| Dataset | Model | Paper Accuracy | Replication Accuracy | Difference |
|---|---|---:|---:|---:|
| NSL-KDD | DNN | 0.9930 | 0.976506 | -0.016494 |
| NSL-KDD | 1D-CNN | 0.9920 | 0.979289 | -0.012711 |
| UNSW-NB15 | DNN | 0.8000 | 0.908623 | +0.108623 |
| UNSW-NB15 | 1D-CNN | 0.8000 | 0.808734 | +0.008734 |

The differences are calculated as:

```text
Difference = Replication Accuracy - Paper Accuracy
```

### Additional Replication Metrics

| Dataset | Model | Precision (Macro) | Recall (Macro) | F1 (Macro) | F1 (Weighted) |
|---|---|---:|---:|---:|---:|
| NSL-KDD | DNN | 0.911678 | 0.912633 | 0.911299 | 0.976664 |
| NSL-KDD | 1D-CNN | 0.913875 | 0.896694 | 0.895328 | 0.980287 |
| UNSW-NB15 | DNN | 0.745989 | 0.758622 | 0.752017 | 0.909999 |
| UNSW-NB15 | 1D-CNN | 0.727713 | 0.692098 | 0.671600 | 0.781154 |

> The paper accuracy values above are the values used for the project's paper-vs-replication comparison. The replication values are from the executed project pipeline.

---

## 9. Explainability

SHAP is used as the explainability method.

The pipeline generates SHAP explanations to analyze the contribution of input features to model predictions.

The generated explainability outputs include SHAP summary visualizations for the evaluated model/dataset experiments.

---

## 10. Important Replication Note

This project is a **reimplementation/reconstruction**, not the original authors' source code.

Some implementation details are not fully specified in the paper. Where necessary, assumptions have been explicitly documented in the code and project documentation.

In particular, the exact feature-selection implementation is not provided by the paper, so the replication uses the documented Pearson-correlation-based implementation.

Therefore, differences between the reported paper results and the reproduced results can arise from differences in implementation details, preprocessing, feature selection, dataset handling, and execution environment.

---

## 11. AI Usage

The project includes an `AI_USAGE.md` file documenting:

- AI tool used
- Prompts used during development
- How AI-generated code/suggestions were modified
- How the project team reviewed and tested the generated output

AI assistance was used as a development aid. The reported experimental results were obtained by executing the project pipeline.

---


## 12. Paper

Sharma et al. (2024):

**Explainable Artificial Intelligence for Intrusion Detection in IoT Networks: A Deep Learning Based Approach**

The replication focuses on the neural-network intrusion-detection experiments and SHAP-based explainability described in the paper.
