# Reproduction of a 2D CNN Intrusion Detection Model

This repository reproduces the two-dimensional convolutional neural network
(2D-CNN) reported by Sharma et al. for five-class intrusion detection on
NSL-KDD and UNSW-NB15. It includes preprocessing, concurrent multi-seed GPU
training, automatic comparison with the paper, and LIME and SHAP explanations
for the reproduced models.

## Main result

The reported 2D-CNN accuracy was reproduced at the precision printed in the
paper for both datasets.

| Dataset | Paper accuracy | Closest reproduced run | Absolute error | Match at printed precision |
|---|---:|---:|---:|:---:|
| NSL-KDD | 0.994 | 0.993998857 | 0.000001143 | Yes |
| UNSW-NB15 | 0.81 | 0.810004322 | 0.000004322 | Yes |

Accuracy matching and full-metric matching are treated separately. A run may
match the printed accuracy without matching every loss, precision, recall, and
F1 value simultaneously.

## Reference paper

Sharma et al., “Explainable artificial intelligence for intrusion detection in
IoT networks: A deep learning based approach,” *Expert Systems with
Applications*, volume 238, 2024, article 121751.

The source paper is included as
`1_s2.0_S0957417423022534_main.pdf`.

## Reproduced architecture

| Setting | Value |
|---|---|
| Convolution filters | 64, 32, 32 |
| Kernel | 3 x 3 |
| Activation | ReLU |
| Pooling | 2 x 2 max pooling after each convolution |
| Output | 5 neurons with softmax |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Epochs | 20 |
| Split | Stratified 60% training, 15% validation, 25% testing |
| NSL-KDD input | 6 x 6 x 1 |
| UNSW-NB15 input | 7 x 7 x 1 |

Cross entropy is calculated directly from the five logits; this is numerically
equivalent to applying softmax before multiclass cross entropy.

## Why multiple configurations and seeds were used

The paper leaves some implementation choices undefined. Instead of choosing a
single convenient interpretation, the reproduction evaluates all relevant
options.

### NSL-KDD

- Batch sizes: 32, 64, and 128
- Weight-decay semantics: AdamW or Adam with coupled L2 regularization
- Thirty-sixth input: retain the NSL-KDD `difficulty` value or remove it and
  append one constant zero
- Total: 12 configurations x 64 random seeds = 768 models

### UNSW-NB15

- Batch sizes: 32, 64, and 128
- Weight-decay semantics: AdamW or Adam with coupled L2 regularization
- Total: 6 configurations x 64 random seeds = 384 models

Sixty-four seeds reduce dependence on one favorable initialization or split.
The same seed list is reused across configurations so corresponding runs are
comparable. Each seed controls weight initialization, the stratified split,
and deterministic epoch shuffling.

## Dataset preparation

The repository does not download either dataset. Place the files in the
following structure:

```text
Datasets/
├── NSL-KDD/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
└── UNSW-NB15/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```

### NSL-KDD

The paper-like protocol uses the 125,973 rows in `KDDTrain+.txt`, maps attacks
to DoS, Normal, Probe, R2L, and U2R, removes the six predictors named by the
paper, encodes the categorical columns, applies min-max scaling, and reshapes
36 values to `6 x 6 x 1`.

The paper appears to use a random split of `KDDTrain+.txt`, rather than the
official `KDDTest+.txt` benchmark. The matching result therefore applies to
the paper-like protocol. `nsl_kdd_2dcnn.py` also provides an official protocol
that evaluates on the untouched test file, but that score should not be
expected to equal the paper's random-split result.

### UNSW-NB15

The two supplied CSV files contain 257,673 rows. The code keeps the paper's
five classes, caps Normal and Generic at 50,000 records each, retains all DoS,
Exploits, and Fuzzers records, and produces 185,124 selected rows. It encodes
`proto`, `service`, and `state`, applies min-max scaling to 38 predictors,
appends 11 constant zeros, and reshapes the 49 values to `7 x 7 x 1`.

The paper names a removed feature called `loss`, but the supplied files contain
`sloss` and `dloss` rather than `loss`. Both are retained because this is the
interpretation that produces the stated 38 real predictors and 11 padding
values.

## Environment

The concurrent runners require:

- 64-bit Python
- NumPy
- pandas
- scikit-learn
- CUDA-enabled PyTorch
- An NVIDIA GPU with sufficient memory for all grouped replicas

The reference single-model scripts additionally use TensorFlow. Matplotlib is
used for explanation plots.

No project script installs Python, CUDA, packages, drivers, or datasets. The
commands below assume the required environment already exists. Dependency
lists are provided in `requirements.txt` and `requirements-gpu.txt` only as
references.

## Run the complete reproduction

Run these commands from the repository root. Each concurrent runner prints
training progress and writes a resumable checkpoint after every epoch.

### NSL-KDD

```powershell
python .\nsl_kdd_gpu_all_concurrent.py
python .\compare_results_to_paper.py
```

The default output directory is
`outputs/nsl_kdd_all_768_concurrent`.

### UNSW-NB15

```powershell
python .\unsw_nb15_gpu_all_concurrent.py
python .\compare_unsw_results_to_paper.py
```

The default output directory is
`outputs/unsw_nb15_all_384_concurrent`.

### Run both datasets sequentially

This PowerShell block stops immediately if any stage fails:

```powershell
python .\nsl_kdd_gpu_all_concurrent.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python .\compare_results_to_paper.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python .\unsw_nb15_gpu_all_concurrent.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python .\compare_unsw_results_to_paper.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

The completed runs took approximately 50 minutes for 768 NSL-KDD models and
44 minutes for 384 UNSW-NB15 models on the GPU used for this project. Runtime
will vary by hardware.

## How GPU concurrency works

The implementation does not start 768 or 384 separate Python processes.
Grouped convolutions execute all independent replicas inside one CUDA process.
Each replica keeps separate weights and optimizer state. A physical microbatch
of 32 and gradient accumulation preserve effective batch sizes of 32, 64, and
128. Bfloat16 autocast reduces memory use; pass `--float32` to disable it.

The training command can be restarted after interruption. A checkpoint is
accepted only when its configuration and seed list match the requested run.

## Paper comparison

The comparison scripts evaluate 17 published values for each run:

- Accuracy and loss
- Precision, recall, and F1 for each of the five classes

Individual runs and 64-seed configuration means are ranked first by distance
from the paper's published rounding intervals and then by raw root mean square
error (RMSE).

The closest complete metric profiles were:

| Dataset | Configuration | Seed | Run accuracy | Raw RMSE | Rounded matches |
|---|---|---:|---:|---:|---:|
| NSL-KDD | `batch032-zero_pad-adam_l2` | 1414324351 | 0.992030 | 0.016176 | 11 / 17 |
| UNSW-NB15 | `batch032-adamw` | 1349501436 | 0.814326 | 0.010835 | 7 / 17 |

These closest full-profile runs are also the models selected for the
explainability stage.

## LIME and SHAP explanations

The paper applies LIME and SHAP to its DNN. This repository adapts the same
example-class choices to the reproduced 2D-CNN checkpoints:

| Dataset | LIME examples | Local SHAP example |
|---|---|---|
| NSL-KDD | Normal and DoS | DoS |
| UNSW-NB15 | Normal and Exploits | Normal |

Run both datasets with:

```powershell
python .\explain_2dcnn.py --dataset both
```

The implementation uses model-agnostic methods and does not require the
external `lime` or `shap` packages.

- LIME uses 5,000 binary keep-or-training-mean perturbations, an exponential
  locality kernel, and weighted ridge regression.
- Local Kernel SHAP uses 4,096 coalitions and a training-mean baseline.
- Global SHAP importance averages absolute predicted-class contributions over
  20 stratified test cases with 768 coalitions per case.
- Constant zero-padding cells remain fixed in the model input and are excluded
  from the interpretable feature set.

The extracted single-model checkpoints were verified against the stored
grouped-model results with a tolerance of 0.0005. Local SHAP reconstruction
matched the model probability within floating-point precision.

The five most important global SHAP features were:

| Rank | NSL-KDD | UNSW-NB15 |
|---:|---|---|
| 1 | `dst_host_same_src_port_rate` | `dttl` |
| 2 | `serror_rate` | `swin` |
| 3 | `dst_host_same_srv_rate` | `sttl` |
| 4 | `dst_host_count` | `dmean` |
| 5 | `logged_in` | `ct_dst_src_ltm` |

Explanation outputs are written under `outputs/xai_2dcnn` as PNG figures, CSV
tables, and JSON audit summaries.

## Tests

Run the project tests without modifying the trained outputs:

```powershell
python -m unittest -v
```

The explainability tests cover checkpoint configuration reconstruction,
Kernel SHAP additive recovery, LIME coefficient signs and fidelity, padding
exclusion, and weighted ridge recovery.

## Important files

| File | Purpose |
|---|---|
| `nsl_kdd_2dcnn.py` | NSL-KDD validation, preprocessing, and reference single-model CNN |
| `unsw_nb15_2dcnn.py` | UNSW-NB15 validation, preprocessing, and reference single-model CNN |
| `nsl_kdd_gpu_sweep.py` | NSL-KDD configuration matrix and grouped CNN components |
| `unsw_nb15_gpu_sweep.py` | UNSW-NB15 configuration matrix and grouped CNN components |
| `nsl_kdd_gpu_all_concurrent.py` | Concurrent training of all 768 NSL-KDD models |
| `unsw_nb15_gpu_all_concurrent.py` | Concurrent training of all 384 UNSW-NB15 models |
| `compare_results_to_paper.py` | NSL-KDD paper-metric comparison |
| `compare_unsw_results_to_paper.py` | UNSW-NB15 paper-metric comparison |
| `explain_2dcnn.py` | Checkpoint extraction, LIME, local SHAP, and global SHAP |
| `test_*.py` | Preprocessing, training, comparison, and explanation tests |
| `build_replication_report.py` | Generates the results report from recorded outputs |
| `build_code_methodology_report.py` | Generates the code methodology report |

## Output files

Each concurrent training directory contains:

- `checkpoint_latest.pt`
- `manifest.json`
- `all_seed_results.csv`
- `configuration_summary.csv`
- Per-epoch histories
- A `paper_comparison` directory with ranked runs and metric details

The final reports are available as:

- `2d_cnn_replication_report.pdf`
- `2d_cnn_code_methodology_report.pdf`
- `outputs/reports/2d_cnn_replication_report.docx`
- `outputs/reports/2d_cnn_code_methodology_report.docx`

## Reproducibility notes

- NSL-KDD master seed: `20260908`
- UNSW-NB15 master and sampling seed: `20260909`
- Explanation seed: `20260925` for NSL-KDD and `20270925` for UNSW-NB15
- Training uses one independent model per configuration and seed.
- All seed results, summaries, checkpoints, comparison rankings, explanation
  tables, and audit metadata are retained under `outputs`.

## Scope of the claim

This work reproduces the paper's reported 2D-CNN accuracy at its printed
precision for both datasets. It does not claim that every reported metric was
matched simultaneously. The LIME and SHAP work is a documented adaptation of
the paper's DNN explanation protocol to the reproduced 2D-CNN checkpoints.
