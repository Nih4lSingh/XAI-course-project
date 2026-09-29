# AI Usage Disclosure Statement
## Academic Integrity & AI-Assisted Replication Framework

**Project**: Replication of Sharma et al. (2024), *Explainable Artificial Intelligence for Enhancing Intrusion Detection Systems Using Deep Learning*, IEEE Access.  
**Course**: Explainable AI / Deep Learning Project  
**Date**: September 2026  

---

### 1. Overview of AI Assistance
In accordance with academic integrity guidelines, this document provides a comprehensive and transparent record of generative AI tools used during this replication project. Generative AI tools (including Google Antigravity and Anthropic Claude) were utilized as pair programming and technical documentation assistants. All AI-generated suggestions, code snippets, and mathematical derivations were subjected to rigorous manual verification against the published paper, audited for empirical reproducibility, and validated through deterministic unit testing.

---

### 2. Specific Tasks Where AI Was Consulted

| Pipeline Stage | AI Tool Utilized | Nature of Assistance | Human Verification & Modifications Made |
| :--- | :--- | :--- | :--- |
| **Dataset Preprocessing** | Google Antigravity | Automated translation of feature filtering rules and spatial grid reshaping (6x6 for NSL-KDD, 7x7 for UNSW-NB15). | Verified exact dropped column lists against paper Section III; checked that UNSW zero-padding preserves feature ordering. |
| **Model Architectures** | Google Antigravity | Scaffolded PyTorch model definitions for DNN, 1D-CNN, and 2D-CNN. | Ensured layer dimensions, kernel sizes, pool strides, and dropout rates matched Table 1 & Table 2 of Sharma et al. (2024) exactly. |
| **Multi-Seed Engine** | Claude / Antigravity | Implementation of LordKarsSama's rounding-aware ranking formula and concurrent GPU execution streams. | Verified mathematical correctness of half-unit rounding intervals and deterministic PRNG seeding. |
| **Explainable AI (SHAP)** | Google Antigravity | Boilerplate for SHAP `KernelExplainer` background sample selection and plotting routines. | Verified that background samples represent the true data distribution and validated top-feature rankings against paper Figure 7. |
| **Documentation & Reports** | Google Antigravity | Formatting Markdown reports and drafting publication-style summary PDF scripts. | Reviewed all text for factual precision, technical rigor, and consistency with empirical findings. |

---

### 3. Exemplary Prompts & Dialogue Structure

#### Example Prompt 1 (Feature Selection & Grid Reshaping)
> *"Implement a standalone Python preprocessing module for Sharma et al. (2024) NSL-KDD and UNSW-NB15 benchmarks. NSL-KDD must drop 6 correlated predictors ('land', 'urgent', 'num_failed_logins', 'root_shell', 'su_attempted', 'num_shells') and retain difficulty to form 36 features reshaped to a 6x6 grid. UNSW-NB15 must drop 6 predictors ('ct_src_dport_ltm', 'loss', 'dwin', 'ct_ftp_cmd', 'label', 'ct_srv_dst') plus 'id', and pad the 38 real predictors with 11 zeros to form 49 features reshaped to a 7x7 grid. Both must use stratified 60/15/25 splitting and Min-Max scaling."*

#### Example Prompt 2 (Architectural Faithfulness)
> *"Construct the exact deep architectures reported in Table 1 and Table 2 of Sharma et al. (2024): DNN with 3 hidden layers of 64 units and p=0.01 dropout; 1D-CNN with Conv1D(64, k=3) -> MaxPool(2) -> Conv1D(32, k=3) -> MaxPool(2) -> Conv1D(32, k=3) and p=0.0 dropout; 2D-CNN with Conv2D(32, 3x3) -> MaxPool(2) -> Conv2D(64, 3x3) -> MaxPool(2) -> Dense(64) -> Dropout(0.5) -> Dense(5). Use AdamW optimizer with learning rate 0.001 and weight decay 0.0001."*

---

### 4. Human Verification, Testing & Defensibility

1. **Independent Execution**: Every script and notebook in this repository was executed independently in clean runtime environments (both local CPU/GPU and Google Colab).
2. **Numerical Validation**: All metric computations (accuracy, precision, recall, macro-F1) were computed using standard `scikit-learn` implementations rather than custom unverified logic.
3. **Absence of Data Leakage**: Scalers and encoders are strictly fitted on the training split only and applied transform-only on validation and test sets.
4. **Final Responsibility**: The student authors accept full responsibility for all code, results, analysis, and conclusions presented in this repository.
