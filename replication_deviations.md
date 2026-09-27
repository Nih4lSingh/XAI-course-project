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

### Deviation / Ambiguity 2: NSL-KDD Feature Dimensionality (42 raw -> 36 selected)
* **Issue:** Mathematical alignment between raw NSL-KDD attributes, 6 dropped features, and the paper's reported 36 selected features.
* **Paper states:** The paper states that 6 features (`srv_serror_rate`, `dst_host_srv_rerror_rate`, `num_root`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `srv_rerror_rate`) are dropped due to $|PCC| > 0.95$, and that the resulting dataset contains 36 features. It reshapes the input into a $6 \times 6 = 36$ grid for 2D-CNN. However, the paper's feature table lists only 41 traffic attributes and omits `difficulty_level`.
* **Our implementation:** In the original NSL-KDD dataset, there are 41 traffic attributes, 1 binary/multi-class label, and 1 difficulty score attribute (43 columns total). To isolate the ground-truth target label and prevent leakage, target labels are removed. We chose to retain `difficulty_level` as an input predictor, yielding 42 raw predictors. Dropping the 6 collinear features yields exactly $42 - 6 = 36$ selected features, mapping into a $6 \times 6$ grid with **0 padding zeros**. In an alternative strict-traffic ablation where `difficulty_level` is discarded as non-network metadata, 41 raw predictors minus 6 yields 35 features, requiring 1 zero padding cell. Retaining `difficulty_level` to achieve the canonical 36 features is our **project reconstruction decision**, not a mechanical copy of a procedure specified in the paper.
* **Reason:** Reconciles the paper's reported 36-feature dimension and zero-padded $6 \times 6$ grid without introducing synthetic padding.
* **Effect on replication:** Guarantees exact agreement with the paper's canonical 36-feature model input shape.
* **Confidence:** High.

---

### Deviation / Ambiguity 3: UNSW-NB15 "label" and "loss" Columns in Feature Selection (42 raw -> 38 selected)
* **Issue:** The paper lists `label` and `loss` among the 6 features removed by Pearson correlation, but reports 38 selected features.
* **Paper states:** Section 3.2 lists 6 removed features for UNSW-NB15: `ct_src_dport_ltm`, `loss`, `dwin`, `ct_ftp_cmd`, `label`, `ct_srv_dst`. However, the paper does not present the arithmetic "$42 - 4 = 38$".
* **Our implementation:** In UNSW-NB15, `label` is the ground-truth binary classification target ($0 = \text{Normal}, 1 = \text{Attack}$) and `attack_cat` is the multi-class target. Retaining `label` inside the predictor set $X$ would cause catastrophic target leakage. In our pipeline, `label` and `attack_cat` are isolated immediately upon ingestion into the target vector $y$, leaving 42 input predictors. Furthermore, UNSW-NB15 contains no column named `loss`; it contains source packet loss (`sloss`) and destination packet loss (`dloss`). Dropping the 4 unambiguous collinear traffic predictors (`ct_src_dport_ltm`, `dwin`, `ct_ftp_cmd`, `ct_srv_dst`) while retaining `sloss` and `dloss` leaves exactly **38 selected features**. For the 2D-CNN, the 38 features are reshaped into a $7 \times 7 = 49$ grid with **exactly 11 trailing zero-padding elements**. Isolating `label` to avoid leakage and retaining `sloss`/`dloss` to yield 38 selected features are **project reconstruction decisions**, not paper arithmetic.
* **Reason:** Strictly prevents target leakage while maintaining architectural compatibility with the paper's 38 selected features and $7 \times 7$ grid.
* **Effect on replication:** Prevents artificial target leakage and achieves the canonical 38 input features with 11 padding zeros.
* **Confidence:** High.

---

### Deviation / Ambiguity 4: 2D-CNN Fig. 5 Topology and Spatial Padding on Small Grids
* **Issue:** Paper Fig. 5 illustrates 3 Conv2D layers and 3 MaxPooling2D layers. On small $6 \times 6$ and $7 \times 7$ grids, standard valid pooling leads to negative or zero spatial dimensions.
* **Paper states:** Figure 5 illustrates Conv2D(64, 3x3) $\to$ MaxPool2D(2x2) $\to$ Conv2D(32, 3x3) $\to$ MaxPool2D(2x2) $\to$ Conv2D(32, 3x3) $\to$ MaxPool2D(2x2) $\to$ Flatten $\to$ Dense(5, softmax).
* **Our implementation:** We implement the exact 3-Conv / 3-Pool architecture with deterministic `padding='same'` on all Conv2D and MaxPooling2D layers. For NSL-KDD ($6 \times 6$), the spatial transitions proceed: $6 \times 6 \to 3 \times 3 \to 2 \times 2 \to 1 \times 1$, flattening to 32 units. For UNSW-NB15 ($7 \times 7$), the spatial transitions proceed: $7 \times 7 \to 4 \times 4 \to 2 \times 2 \to 1 \times 1$, flattening to 32 units.
* **Reason:** Preserves every single convolutional and pooling layer depicted in the paper's canonical Figure 5 without dropping any pooling operations.
* **Effect on replication:** Fully adheres to the paper's diagrammatic architecture.
* **Confidence:** High.

---

### Deviation / Ambiguity 5: 1D-CNN Inferred Filter Progression
* **Issue:** 1D-CNN filter counts and layer depths are omitted from the text.
* **Paper states:** Section 4.2 specifies `kernel_size = 3`, `pool_size = 2`, `activation = 'relu'`, `optimizer = Adam`, `learning_rate = 0.001`, `weight_decay = 0.0001`, and `epochs = 20`. The number of convolutional layers and filter dimensions are unstated.
* **Our implementation:** We implemented an architecture consistent with the 2D-CNN pattern: Conv1D(64, kernel_size=3, padding='same', activation='relu') $\to$ MaxPool1D(pool_size=2) $\to$ Conv1D(32, kernel_size=3, padding='same', activation='relu') $\to$ Flatten $\to$ Dense(5, activation='softmax'). This is cataloged as `INFERRED PARAMETER`.
* **Reason:** Mirrors the 2D-CNN filter progression (64 $\to$ 32) while adhering strictly to the paper's explicit kernel (3) and pooling (2) parameters.
* **Effect on replication:** Fully functional 1D-CNN with comparable parameter scale to the paper's reported training times.
* **Confidence:** Medium.

---

### Deviation / Ambiguity 6: DNN Architecture Standardization
* **Issue:** Discrepancy between paper specification and third-party references (e.g. 128 -> 64 -> 32).
* **Paper states:** Section 4.1 specifies three dense layers with 64 units each and ReLU activation, followed by a 5-unit Softmax classification layer ($64 \to 64 \to 64 \to 5$).
* **Our implementation:** We audited and strictly enforce Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(5, Softmax). All legacy scripts or documentation mentioning 128 $\to$ 64 $\to$ 32 were purged.
* **Reason:** Full fidelity to Section 4.1 text.
* **Effect on replication:** Exact structural alignment with Sharma et al. (2024).
* **Confidence:** High.

---

### Deviation / Ambiguity 7: Training Batch Size Not Specified
* **Issue:** Batch size for DNN, 1D-CNN, and 2D-CNN is omitted from the publication.
* **Paper states:** Section 4 does not report a batch size.
* **Our implementation:** We configure `batch_size = 64` (with support for 32). This is recorded as `INFERRED PARAMETER`.
* **Reason:** 64 is standard in Keras network intrusion detection benchmarks and aligns with GPU memory throughput.
* **Effect on replication:** Governs gradient update frequency per epoch; stable convergence across 20 epochs.
* **Confidence:** Medium.

---

### Deviation / Ambiguity 8: Training Time Terminology & Comparability
* **Issue:** The paper reports 142/325/340 ms (NSL-KDD) and 323/442/455 ms (UNSW-NB15) as "training times" without specifying execution hardware, epoch breakdown, or profiling scope.
* **Paper states:** Table 3 and accompanying text report these figures strictly as "training time" in milliseconds.
* **Our implementation:** We label the published figures strictly as `"Paper-reported training time"` and our measured times as `"Our measured total 20-epoch wall-clock training time"`. In comparative tables, headers are formatted as `Paper Training Time (ms)` and `Our Total Training Time (s)`. We record high-resolution timestamps (`time.perf_counter()`) for total training time, per-epoch average, and inference latency. We add the explicit caveat: *"The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons."* We do NOT describe the paper's figures as inference latency, per-sample latency, or per-batch inference.
* **Reason:** Adheres strictly to the paper's own terminology while ensuring transparent, scientifically rigorous reporting.
* **Effect on replication:** Eliminates mischaracterization of published metrics while recording empirical execution times faithfully.
* **Confidence:** High.

---

### Deviation / Ambiguity 9: UNSW-NB15 SHAP Top Feature 'data' Inconsistency
* **Issue:** Sharma et al. cite `data` as the #1 most important feature for UNSW-NB15 Normal global SHAP attribution, but `data` does not exist in the UNSW-NB15 dataset.
* **Paper states:** Figure 7 and Section 5.2 describe `data` as the top-ranking feature driving Normal flow classification in UNSW-NB15.
* **Our implementation:** We document that a feature named `data` is neither in the UNSW-NB15 dataset schema nor in the paper's own feature table (Table 2). In our empirical replication, we compute SHAP strictly on the canonical 38-feature set and faithfully report the actual top-ranking feature: `dttl` (destination time to live, mean |SHAP| = 0.15098), followed by `swin`, `sttl`, `ct_dst_sport_ltm`, and `ct_state_ttl`.
* **Reason:** Zero fabrication: we report empirical outputs from the real dataset rather than inserting or inventing non-existent features.
* **Effect on replication:** Completely transparent audit trail resolving a published textual inconsistency.
* **Confidence:** High.
