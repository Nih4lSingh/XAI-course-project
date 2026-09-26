# Replication Deviation & Ambiguity Log

This document meticulously records every ambiguity, inconsistency, missing detail, and explicit replication decision made during our reimplementation of:

**Sharma et al. (2024)**, *“Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach”*, *Expert Systems with Applications*, 238, 121751.

Each entry follows the strict scientific traceability format:
- **Issue:** Description of the ambiguity or discrepancy
- **Paper states:** Exact wording, table values, or citations from the original publication
- **Our implementation:** Concrete reproducible decision taken in this replication
- **Reason:** Scientific, mathematical, or architectural rationale
- **Effect on replication:** Impact on numerical results, model capacity, or runtime
- **Confidence:** High / Medium / Low

---

### Deviation / Ambiguity 1: DNN Dropout Rate Contradiction
* **Issue:** Contradiction between Table 1 and Section 4.1 text regarding DNN dropout.
* **Paper states:** Table 1 lists `"Dropout: 0"`. Section 4.1 text mentions setting `"dropout rate to 0.01 to prevent overfitting"`.
* **Our implementation:** We implemented a configurable `dropout_rate` hyperparameter. The primary replication runs with `dropout_rate = 0.00` (matching Table 1 specification), and a sensitivity experiment runs with `dropout_rate = 0.01` to directly quantify the impact of this ambiguity.
* **Reason:** In standard deep learning, dropout of 0.01 is an unusually low rate (typically 0.2–0.5 is used). Table 1 represents the explicit hyperparameter summary table. Testing both values guarantees scientific transparency without arbitrarily picking one.
* **Effect on replication:** Minimal difference in test accuracy expected due to small magnitude (0 vs 1%), but fully resolves the ambiguity.
* **Confidence:** High.

---

### Deviation / Ambiguity 2: NSL-KDD Feature Dimensionality (35 vs 36)
* **Issue:** Mathematical mismatch between original NSL-KDD features, 6 dropped features, and the paper's reported 36 selected features.
* **Paper states:** The paper states that 6 features (`srv_serror_rate`, `dst_host_srv_rerror_rate`, `num_root`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `srv_rerror_rate`) are dropped due to $|PCC| > 0.95$, and that the resulting dataset contains 36 features. It then reshapes the input into a $6 \times 6 = 36$ grid for 2D-CNN.
* **Our implementation:** Standard NSL-KDD contains 41 traffic predictors (plus 1 attack class column and 1 difficulty level column). Removing 6 predictors leaves exactly $41 - 6 = 35$ features. To construct the $6 \times 6 = 36$ input grid for 2D-CNN, we either append 1 deterministic zero-padding feature or investigate if the original authors retained `difficulty_level` before dropping. For target-safety and consistency, we maintain the 35 true network predictors for DNN and 1D-CNN, and zero-pad 1 element to yield the $6 \times 6$ grid for 2D-CNN.
* **Reason:** Preserves 100% integrity of network traffic predictors while satisfying the $6 \times 6$ geometric requirement of the 2D-CNN.
* **Effect on replication:** Allows flawless execution of the 2D-CNN without distorting feature semantics.
* **Confidence:** High.

---

### Deviation / Ambiguity 3: UNSW-NB15 "label" Column in Feature Selection Table
* **Issue:** The paper lists `label` as one of the 6 features removed by Pearson correlation.
* **Paper states:** Section 3.2 explicitly lists the removed features for UNSW-NB15: `ct_src_dport_ltm`, `loss`, `dwin`, `ct_ftp_cmd`, `label`, `ct_srv_dst`.
* **Our implementation:** In the UNSW-NB15 dataset, `label` is the ground-truth binary target column ($0 = \text{Normal}, 1 = \text{Attack}$), whereas `attack_cat` is the multi-class attack category. If `label` were included in the feature matrix $X$ prior to Pearson correlation, it would constitute target leakage. In our implementation, `label` and `attack_cat` are separated immediately upon ingestion into the target vector $y$. The 5 redundant predictors (`ct_src_dport_ltm`, `loss` [i.e. `sloss`/`dloss`], `dwin`, `ct_ftp_cmd`, `ct_srv_dst`) are dropped from the input feature set.
* **Reason:** Target leakage violates fundamental machine learning principles. The paper's authors likely computed a correlation matrix across all dataframe columns (including the binary target `label`) and observed high correlation with attack status.
* **Effect on replication:** Prevents artificial inflation of classification metrics and ensures scientific defensibility.
* **Confidence:** High.

---

### Deviation / Ambiguity 4: Incomplete 1D-CNN Architecture Specification
* **Issue:** 1D-CNN filter counts and layer depths are omitted from the text.
* **Paper states:** Section 4.2 specifies `kernel_size = 3`, `pool_size = 2`, `activation = 'relu'`, `optimizer = Adam`, `learning_rate = 0.001`, `weight_decay = 0.0001`, and `epochs = 20`. The number of convolutional layers and filter dimensions are unstated.
* **Our implementation:** We implemented an architecture consistent with the 2D-CNN pattern: Conv1D(64, kernel_size=3, padding='same', activation='relu') $\to$ MaxPool1D(pool_size=2) $\to$ Conv1D(32, kernel_size=3, padding='same', activation='relu') $\to$ Flatten $\to$ Dense(5, activation='softmax'). This is cataloged as `INFERRED PARAMETER`.
* **Reason:** Mirrors the 2D-CNN filter progression (64 $\to$ 32) while adhering strictly to the paper's explicit kernel (3) and pooling (2) parameters.
* **Effect on replication:** Fully functional 1D-CNN with comparable parameter scale to the paper's reported training times.
* **Confidence:** Medium.

---

### Deviation / Ambiguity 5: 2D-CNN Spatial Feature Ordering
* **Issue:** The exact spatial mapping (which feature goes to which $(i, j)$ cell in the $6 \times 6$ or $7 \times 7$ grid) is not specified.
* **Paper states:** NSL-KDD is reshaped to $6 \times 6 \times 1$; UNSW-NB15 is reshaped to $7 \times 7 \times 1$ with zero padding.
* **Our implementation:** We use deterministic row-major ordering based on the standardized column sequence after feature selection, zero-padding the final cells up to the target grid capacity. The mapping is saved to `feature_to_grid_mapping.json`.
* **Reason:** CNN spatial inductive bias depends on adjacency; without an explicit coordinate map, deterministic natural ordering is the only reproducible, non-arbitrary choice.
* **Effect on replication:** Guarantees 100% deterministic reproducibility across runs and environments.
* **Confidence:** High.

---

### Deviation / Ambiguity 6: Training Batch Size Not Specified
* **Issue:** Batch size for DNN, 1D-CNN, and 2D-CNN is omitted from the publication.
* **Paper states:** Section 4 does not report a batch size.
* **Our implementation:** We configure `batch_size = 64` (with support for 32). This is recorded as `INFERRED PARAMETER`.
* **Reason:** 64 is standard in Keras network intrusion detection benchmarks and aligns with GPU memory throughput.
* **Effect on replication:** Governs gradient update frequency per epoch; stable convergence across 20 epochs.
* **Confidence:** Medium.

---

### Deviation / Ambiguity 7: Training Time Reporting Context
* **Issue:** The paper reports training times of $\approx 142$ ms for NSL-KDD DNN, $\approx 325$ ms for 1D-CNN, etc., without specifying whether this represents per-epoch step time or total runtime, nor the exact CPU/GPU model.
* **Paper states:** Training times given as milliseconds (e.g. 142 ms, 325 ms, 340 ms).
* **Our implementation:** We record high-resolution timestamps (`time.perf_counter()`) for total training time, per-epoch average time, and per-sample inference latency, alongside complete system hardware and software specifications.
* **Reason:** Training time in milliseconds on modern hardware typically reflects single-epoch or single-batch execution rather than 20 full epochs over $>100,000$ samples.
* **Effect on replication:** Enables rigorous academic interpretation without claiming false hardware parity.
* **Confidence:** High.
