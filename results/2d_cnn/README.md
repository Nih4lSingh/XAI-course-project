# 2D-CNN LIME and SHAP reproduction

The source paper used LIME and SHAP on its DNN. These outputs apply the same
paper example classes to the replicated 2D-CNN runs selected by the existing
17-metric paper-comparison scripts.

No external LIME or SHAP package is required. The implementation uses weighted
local keep-or-mean surrogate regression for LIME and model-agnostic Kernel SHAP coalition
regression for SHAP.

## Verified runs

| Dataset | Configuration | Seed | Stored accuracy | Verified accuracy |
|---|---|---:|---:|---:|
| NSL-KDD | batch032-zero_pad-adam_l2 | 1414324351 | 0.992030228 | 0.992157236 |
| UNSW-NB15 | batch032-adamw | 1349501436 | 0.814325843 | 0.814174589 |

## Files per dataset

- `lime_<class>.png`: class probabilities, local LIME weights, and instance values.
- `lime_explanations.csv`: all LIME feature weights and fidelity values.
- `shap_local_<class>.png`: local Kernel-SHAP contributions.
- `shap_local_explanation.csv`: all local SHAP values.
- `shap_global_importance.png`: mean absolute SHAP importance across a stratified test sample.
- `shap_global_importance.csv`: ranked global SHAP importance.
- `xai_summary.json`: full audit metadata and sampling settings.

Constant zero-padding cells are excluded from the interpretable feature set
because they cannot explain a prediction. They remain fixed in every model input.
