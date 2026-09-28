# Tool used

OpenAI Codex was used as the AI-assisted coding and documentation tool for the
2D-CNN reproduction. It was used to inspect the supplied paper and datasets,
write and revise Python code, design the multi-seed GPU experiments, interpret
the recorded results, implement the LIME and SHAP stage, write tests, and
prepare the reports and repository documentation.

# Prompts used

The prompts and follow-up instructions included the following requests:

- Verify the paper's 2D-CNN architecture, hyperparameters, preprocessing, and
  training procedure for NSL-KDD.
- Write the 2D-CNN reproduction code without downloading or installing Python,
  CUDA, datasets, or packages.
- When the paper does not define a parameter, test all relevant plausible
  options instead of selecting one arbitrary value.
- Train every configuration with 64 randomly selected seeds.
- Launch all models concurrently on the GPU with visible progress reporting.
- Run 12 configurations x 64 seeds for NSL-KDD and 6 configurations x 64 seeds
  for UNSW-NB15.
- Provide PowerShell commands that run the dataset experiments sequentially.
- Compare every individual run and each 64-seed configuration mean with the
  accuracy, loss, precision, recall, and F1 values reported by the paper.
- Identify the run closest to the paper's complete 17-metric profile.
- Reproduce the same process for UNSW-NB15.
- Apply LIME and SHAP to the selected reproduced 2D-CNN checkpoints, following
  the example classes used by the paper.
- Verify the implementation and explanation outputs with automated tests and
  numerical checks.
- Prepare concise reproduction, code-methodology, README, and upload-package
  documentation.

# How output was modified

The initial AI-generated code and documentation were not submitted unchanged.
They were revised through repeated user instructions and checks against the
paper, the supplied datasets, and the generated results. The user defined the
core experimental rules: cover all relevant undefined choices, use 64 seeds
per configuration, execute every model concurrently on the GPU, show progress,
avoid downloads and installers, compare results automatically, and reproduce
the LIME and SHAP stage.

Specific examples of user-directed corrections and modifications include:

- When the initial approach attempted to prepare a separate CUDA-enabled
  environment, the user stopped that action and clarified that only code should
  be written and that the existing Python and CUDA installation must be used.
  The final scripts therefore contain no installer, downloader, or environment
  setup action.
- When one value could have been chosen for a parameter not defined by the
  paper, the user explicitly required all relevant plausible options to be
  tested. This instruction produced the batch-size, optimizer-semantics, and
  NSL-KDD feature-mode configuration matrix.
- The initial multi-seed plan was refined by the user from ordinary repeated
  runs to 64 random seeds for every configuration, with all configurations and
  seeds launched concurrently on the GPU. The final NSL-KDD job therefore ran
  12 x 64 models together, and the UNSW-NB15 job ran 6 x 64 models together.
- The user required visible progress and a PowerShell sequence that completed
  one dataset and its comparison before starting the next. Progress reporting,
  resumable checkpoints, and failure-aware sequential commands were added in
  response.
- After obtaining the training results, the user requested a separate program
  to identify the run closest to all metrics in the paper rather than relying
  only on accuracy. The comparison was expanded to 17 metrics with
  rounding-aware and raw RMSE rankings for individual runs and configuration
  means.
- The original task covered model replication. The user later required the
  LIME and SHAP part of the paper to be reproduced for the selected 2D-CNN
  results. This added checkpoint extraction, local LIME, local Kernel SHAP,
  global SHAP importance, plots, CSV files, JSON audit summaries, and tests.

The implementation was modified to use grouped PyTorch convolutions so that
each replica retained independent parameters while all replicas ran in one
CUDA process. Gradient accumulation preserved batch sizes 32, 64, and 128.
Checkpoint validation, progress reporting, paper-metric ranking, rounding-aware
comparison, and CSV and JSON audit outputs were added during revision.

Dataset-specific preprocessing was adjusted to match the dimensions reported
by the paper. For NSL-KDD, both plausible interpretations of the thirty-sixth
input were tested. For UNSW-NB15, the supplied `sloss` and `dloss` fields were
retained because the files did not contain the paper's named `loss` column and
this choice produced the stated 38 predictors before zero-padding.

The user ran the GPU experiments and supplied the resulting comparison output.
The final accuracy claims were taken from those recorded results. The closest
complete metric-profile checkpoints were then used for LIME and Kernel SHAP.
The explanation code was modified to exclude constant padding cells from the
interpretable feature set, verify extracted checkpoint accuracy, report LIME
fidelity, and enforce SHAP probability reconstruction.

The final code was checked with unit tests and syntax validation. The reports
were generated from the saved CSV and JSON results and visually reviewed after
rendering. The documentation was revised to distinguish matching the paper's
printed 2D-CNN accuracy from matching every reported metric, and to state that
the paper's DNN explanation protocol was adapted to the reproduced 2D-CNN.
