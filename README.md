# Sharma 2024 XAI Replication

Reproducible Python project for replicating the model-comparison portion of Sharma et al. (2024), *Explainable artificial intelligence for intrusion detection in IoT networks: a deep learning based approach*.

## Research fidelity

This project implements the paper's multiclass protocol: five classes, Pearson correlation threshold `0.95`, label encoding, min-max normalization, a 60/15/25 random stratified split, 36 NSL-KDD features reshaped to `6x6`, and 38 UNSW-NB15 features zero-padded to `7x7`.

The previous binary experiment outputs in `results/` are not comparable to the paper. Rerun each model after this update to replace them with the five-class protocol's generated results.

Primary source: https://doi.org/10.1016/j.eswa.2023.121751

## Notebooks

Run from the repository root after configuring data paths:

- `notebooks/01_DNN.ipynb`
- `notebooks/02_1D_CNN.ipynb`
- `notebooks/03_2D_CNN.ipynb`

They call the same source modules as the CLI, so notebooks and scripts share preprocessing logic.

## Implementation assumptions and deviations

- NSL-KDD uses `Normal`, `DoS`, `Probe`, `R2L`, and `U2R`; UNSW-NB15 uses `Normal`, `Generic`, `Exploits`, `DoS`, and `Fuzzers`, with Normal and Generic sampled to 50,000 records each.
- Categorical values use label encoding, and all features use min-max normalization. The paper applies these transformations before its random split, so this project does the same to match its protocol.
- The paper does not report a random seed. The configuration uses `20240901` as a recorded replication choice; it is not a paper-reported seed.
- The UNSW-NB15 file contains `sloss` and `dloss`, whereas the paper says `loss`; both are retained to achieve its stated 38-feature input.
- The report and CSVs intentionally distinguish `paper_reported_value` from `replication_value`. Blank / `NOT_REPORTED_IN_PROJECT` means no value has been asserted.

## Explainability context

The paper's XAI phase should be reproduced only after selecting the target classifier and its documented explainer. This project saves feature names and run metadata needed for that step, and includes SHAP as an optional dependency. It does not claim a SHAP, LIME, or feature-attribution result without a completed model run and a paper-verified XAI protocol.

