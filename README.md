# Sharma et al. (2024): replication and Phase 2 XAI evaluation

This repository contains the source for a close reconstruction of Sharma et al.'s Phase 1 intrusion-detection experiments and a Phase 2 evaluation of explanation stability and faithfulness on the same frozen DNN. The Phase 2 result shows a **stability-fidelity trade-off**.

## Reproduction boundary

The paper does not provide executable source, a random seed, exact split indices, processed data, complete feature-selector artifacts, or a SHAP explainer/background configuration. Its data statement says the data are available on request, and it reports some inconsistent NSL-KDD accuracy values. Exact score reproduction is therefore not possible from the paper alone. This implementation uses seed **42**, records its choices in output manifests, and keeps published results separate from local measurements.

The paper describes label encoding, min-max scaling, feature removals, then a 60/15/25 split. This implementation follows that order. Because transformations use the full filtered dataset before the split, the results may be optimistic. No class weights, oversampling, or additional classifier are introduced.

## Datasets and saved-run provenance

Download the required files from the dataset custodians and put them under `work/datasets/` in the project root:

| Dataset | Required files | Custodian page |
|---|---|---|
| NSL-KDD | `KDDTrain+.txt` | [UNB / CIC NSL-KDD page](https://www.unb.ca/cic/datasets/nsl.html) |
| UNSW-NB15 | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` | [UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |

UNB currently says its original NSL-KDD download is unavailable. The paper's reported 125,972 rows match the standard NSL-KDD training file within one record, so the reconstruction uses `KDDTrain+.txt` only. For UNSW-NB15, the code combines the 175,341-row training and 82,332-row testing partitions, keeps the five paper categories, and caps Normal and Generic at 50,000 each using seed 42. Cite the dataset papers as required by UNSW.

The locally saved experiment used an NSL-KDD file and UNSW-NB15 files obtained from a public mirror. The exact UNSW mirror URL was not preserved in the run artifacts; fill in `[UNSW MIRROR URL USED]` before submission if it is known. The manifests record these SHA-256 checksums so replacement files can be checked:

| File | SHA-256 |
|---|---|
| `KDDTrain+.txt` | `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95` |
| `UNSW_NB15_training-set.csv` | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |
| `UNSW_NB15_testing-set.csv` | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |

The benchmark files and generated model/data artifacts are not included in this repository or its source ZIP.

## Environment

Use 64-bit Python 3.13. `requirements.txt` pins the experiment libraries (including TensorFlow 2.21.0, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.9.1, SHAP 0.54.1, LIME 0.2.0.1, SciPy 1.18.1, and Matplotlib 3.11.2). On native Windows, TensorFlow runs on CPU in this setup.

From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
New-Item -ItemType Directory -Force work\datasets
```

For a standalone clone, place the three dataset files in `work\datasets\`. In this full project workspace, the dataset folder is two levels above `outputs\source_code`, so use `..\..\work\datasets\` as shown below.

## Run Phase 1

Run all three architectures on both datasets:

```powershell
python src\replicate_paper.py --dataset both `
  --nsl-train work\datasets\KDDTrain+.txt `
  --unsw-train work\datasets\UNSW_NB15_training-set.csv `
  --unsw-test work\datasets\UNSW_NB15_testing-set.csv
```

For the current full project workspace, replace each `work\datasets\...` argument with `..\..\work\datasets\...`. Results, model weights, split indices, and preprocessing manifests are written under `work\results\phase1\` in the project root.

After Phase 1 has produced the DNN outputs, the optional figure and atlas scripts create local counterparts to the paper's figures and tables:

```powershell
python src\phase1_visuals.py
python src\build_phase1_atlas.py
```

The atlas contains counterparts for all 20 paper figures, including correlation and confusion-matrix heatmaps, plus the seven paper tables transcribed as published. The local plots come from local data, saved predictions, histories, and explanations. The original figure image files and unreleased prediction/explanation arrays are not available, so these are not pixel-for-pixel copies. Published table entries remain labeled as published values.

## Run Phase 2

Phase 2 requires the Phase 1 DNN artifacts:

```powershell
python src\phase2_xai.py --dataset nsl --queries 300 --neighbors 10 --lime-samples 1000
python src\phase2_xai.py --dataset unsw --queries 300 --neighbors 10 --lime-samples 1000
```

The comparison uses the same frozen DNN and 300 predicted-class-stratified test queries per dataset. The code samples up to 100 training references per predicted class, then splits them into disjoint aggregation and evaluation pools. The ten nearest neighbors come from the sampled same-predicted-class **aggregation pool** (about 50 references per class after the split, fewer when that class has fewer available rows). The separate evaluation pool is used to score agreement. Mean query-to-neighbor distance in the scaled-feature Euclidean space was 0.6242 for NSL-KDD and 0.4862 for UNSW-NB15.

The label **SC-SHAP** is retained so the method name maps to the saved output keys. “Stability-Calibrated” is only a project label: the code does not perform a calibration step. The operation is a distance-weighted average of neighboring SHAP vectors, so **Neighborhood-Averaged SHAP** is the more literal description. Because averaged same-class SHAP vectors are evaluated against other same-class SHAP vectors, higher agreement is partly expected by construction, despite the disjoint reference pools. The result is a neighborhood attribution profile, not an additive SHAP explanation guaranteed to be locally accurate for the query.

Outputs include top-five Jaccard, Spearman rank agreement, direction consistency, a perturbation faithfulness proxy, per-query metrics, paired bootstrap intervals, Wilcoxon tests, feature rankings, and model/prediction hashes. The faithfulness gap is the target-class probability drop from masking the top five features minus the drop from a random-five mask; higher values favor the explanation mask under this proxy. It is not causal evidence.

### Saved Phase 2 result summary

These values are from the saved `work/results/phase2/{nsl,unsw}/summary.json` files; no additional Phase 2 runs are implied here.

| Dataset | Top-5 Jaccard delta | Spearman delta | Faithfulness gap, SHAP -> SC-SHAP | Faithfulness paired difference (95% CI) |
|---|---:|---:|---:|---:|
| NSL-KDD | +0.03446 | +0.04742 | 0.08401 -> 0.00769 | -0.07632 [-0.10800, -0.04628] |
| UNSW-NB15 | +0.03482 | +0.04905 | 0.05953 -> 0.04471 | -0.01482 [-0.03087, +0.00141] |

On NSL-KDD, agreement increases while the faithfulness gap falls close to zero, near the random-masking contrast. On UNSW-NB15, the stability gains are similar, but the faithfulness difference is inconclusive because its confidence interval includes zero. Read these results as a stability-fidelity trade-off.

### Runtime interpretation

`summary.json`'s `sc_shap_posthoc_seconds_per_query_mean` is the measured aggregation time with cached SHAP references and **excludes** the one-time reference explanations, as stated in that file. The report's amortized figures add the measured reference explanation time (`reference_shap_seconds_per_training_reference * reference_count`) to the 300-query aggregation time, then divide by 300. That gives approximately 2.26 seconds of one-time reference explanation cost and 0.01676 seconds/query amortized for NSL-KDD, and 2.61 seconds and 0.01794 seconds/query for UNSW-NB15. The post-hoc-only value is about 0.00923 seconds/query on both datasets. These are local CPU timings.

## Limitations to report

- The project did not compare against class-mean SHAP, a random-neighbor average, or a sweep over neighbor counts `k`; the stability gain is not benchmarked against simple aggregation baselines.
- LIME used the library default `sample_around_instance=False`. Its perturbations are centered on the training mean rather than around the query instance, so the LIME baseline numbers may be pessimistic.
- SHAP `DeepExplainer` ran with `check_additivity=False`; the additivity error was not measured or reported.
- Phase 1 NSL-KDD DNN R2L recall is 0.90 versus 0.54 in the paper. UNSW-NB15 accuracies are about 0.7-1.6 percentage points above the paper's comparison values. The saved UNSW files came from a mirror; its exact URL is not recorded in the artifacts.
- The paper's described preprocessing order may leak test-distribution information. The paper reports 36 NSL-KDD predictors while the listed removals yield 35 in the standard schema; the listed UNSW `loss` feature is absent from the standard split CSV.
- The paper does not specify the SHAP variant, LIME perturbation settings, or original explanation examples. Rare-class support and results on benchmark data limit generalization to operational IoT traffic.

## GitHub preparation

The source folder is prepared as the repository root. `.gitignore` excludes datasets, processed data, model weights, caches, and generated result artifacts. The archive does not contain benchmark files or trained weights. No GitHub repository has been created or pushed by this project workflow. After creating an empty repository on GitHub, replace the URL placeholder and run these commands from this folder:

```powershell
git init
git add .
git commit -m "Prepare Sharma XAI replication and Phase 2 evaluation"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
git push -u origin main
```

## Files

- `src/data_pipeline.py`: dataset class mapping, sampling, encoding, scaling, and feature selection.
- `src/models.py`: DNN/CNN definitions, split, and training.
- `src/replicate_paper.py`: Phase 1 training and published-table comparison artifacts.
- `src/phase2_xai.py`: frozen-model SHAP/LIME/neighborhood-averaged SHAP evaluation.
- `src/phase1_visuals.py`: local counterparts for the paper's 20 figures.
- `src/build_phase1_atlas.py`: PDF atlas with local figure counterparts and published table transcriptions.
- `configs/replication.json`: experiment settings, including seed 42.
- `AI_USAGE.md`: AI tool disclosure and student-review fields.
