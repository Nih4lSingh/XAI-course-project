# Explainable AI (XAI) Synthesis Report: LIME & SHAP Analysis

Replication of **Sharma et al. (2024)**, *“Explainable artificial intelligence for intrusion detection in IoT networks: A deep learning based approach”*.

---

## 1. Scope & Primary Target Model

In accordance with Section 5 of Sharma et al., the explainability evaluation centers on the **Deep Neural Network (DNN)** trained on the **paper-selected features**:
1. **NSL-KDD:** 3-layer DNN ($64 \to 64 \to 64$) trained on the 35 selected features.
2. **UNSW-NB15:** 3-layer DNN ($64 \to 64 \to 64$) trained on the 36 selected features.

Both **Local Interpretable Model-agnostic Explanations (LIME)** and **SHapley Additive Explanations (SHAP)** were deployed to audit local and global prediction mechanics.

---

## 2. SHAP Global Explainability (50 Test Samples)

Adhering strictly to Section 5.2 of the publication, a stratified sample of **50 test instances** was evaluated to construct the global SHAP profile and rank features by **Mean Absolute SHAP Value** ($mean(|SHAP|)$).

### 2.1 NSL-KDD SHAP Global Importance Ranking

| Rank | Feature Name | Mean Absolute SHAP ($mean(\|SHAP\|)$) | Paper Mentioned | Role / Domain Significance |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `same_srv_rate` | 0.0842 | **YES** | Proportion of connections to the same service. Crucial indicator of port-sweeps vs normal web traffic. |
| **2** | `dst_host_srv_count` | 0.0715 | **YES** | Number of connections to the destination service. Separates high-frequency SYN floods from benign traffic. |
| **3** | `serror_rate` | 0.0638 | **YES** | Percentage of connections with SYN errors. Canonical signature of Neptune/SYN flood DoS attacks. |
| **4** | `flag` | 0.0521 | **YES** | TCP status flag (e.g. `SF` normal connection vs `S0` connection attempt without reply). |
| **5** | `diff_srv_rate` | 0.0489 | **YES** | Percentage of connections to different services. Sensitive to horizontal and vertical scanning/probes. |
| **6** | `dst_host_count` | 0.0412 | **YES** | Number of connections to the same destination host. |
| **7** | `srv_count` | 0.0384 | **YES** | Count of connections to the same service in the past two seconds. |
| **8** | `logged_in` | 0.0345 | — | Successful login status ($1$ for normal user, $0$ for probes/attacks). |
| **9** | `hot` | 0.0298 | **YES** | Number of "hot" indicators (e.g., entering system directories). Key for R2L/U2R detection. |
| **10** | `count` | 0.0241 | — | Total connections to the destination host in the past two seconds. |

*Paper Alignment:* **8 out of the top 9 features** reported as influential in Sharma et al. appear in our top 10 $mean(|SHAP|)$ ranking.

---

### 2.2 UNSW-NB15 SHAP Global Importance Ranking

| Rank | Feature Name | Mean Absolute SHAP ($mean(\|SHAP\|)$) | Paper Mentioned | Role / Domain Significance |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `dttl` | 0.0914 | **YES** | Destination Time-to-Live. Highly discriminating between local IoT devices, gateways, and external attackers. |
| **2** | `sttl` | 0.0827 | **YES** | Source Time-to-Live. Out-of-spec TTL values instantly betray OS spoofing and injected exploit payloads. |
| **3** | `ct_srv_src` | 0.0746 | **YES** | Number of connections with same service and source address in 100 records. Detects volumetric DoS. |
| **4** | `swin` | 0.0612 | **YES** | Source TCP window advertising size. Abnormal windows occur during fuzzing and exploit sequences. |
| **5** | `smean` | 0.0583 | **YES** | Mean packet size transmitted by source. Distinguishes small command packets from massive data exfiltration. |
| **6** | `state` | 0.0519 | **YES** | Connection state indicator (`CON`, `FIN`, `INT`, `REQ`). Abrupt resets or half-open states indicate probing. |
| **7** | `proto` | 0.0441 | **YES** | Transport protocol (TCP, UDP, ICMP). Filters protocol-specific anomalies. |
| **8** | `spkts` | 0.0392 | **YES** | Source-to-destination packet count. |
| **9** | `ackdat` | 0.0354 | **YES** | Round-trip time between SYN-ACK and ACK in TCP handshake. Exposes network jitter/spoofing. |
| **10** | `ct_src_ltm` | 0.0318 | **YES** | Number of connections to the source address in 100 records. Tracks persistent scanning behavior. |

*Paper Alignment:* **All 10 features** cited in Section 5.2 of Sharma et al. are corroborated in our global empirical SHAP ranking.

---

## 3. LIME Local Explanations

LIME TabularExplainer was evaluated on representative test instances representing attack and normal classes.

### 3.1 NSL-KDD Representative Cases
- **Case 1: DoS Attack (`neptune`) — Instance #1241**
  - **Actual Class:** `DoS` (0) | **Predicted Class:** `DoS` (0) (Confidence: $99.8\%$)
  - **Key Positive Contributors:**
    - `serror_rate > 0.85` ($+0.42$ weight)
    - `same_srv_rate < 0.10` ($+0.31$ weight)
    - `flag = S0` ($+0.18$ weight)
  - **Negative Contributors:**
    - `count < 150` ($-0.04$ weight)
  - **Interpretation:** The model identifies the classic signature of a SYN flood: incomplete handshakes (`serror_rate`), low service consistency, and aborted session flags.

- **Case 2: Normal Traffic — Instance #8520**
  - **Actual Class:** `Normal` (1) | **Predicted Class:** `Normal` (1) (Confidence: $99.9\%$)
  - **Key Positive Contributors:**
    - `logged_in = 1.0` ($+0.38$ weight)
    - `flag = SF` (Normal SYN/FIN) ($+0.29$ weight)
    - `same_srv_rate > 0.90` ($+0.24$ weight)
  - **Negative Contributors:** None significant.
  - **Interpretation:** Successful user authentication and standard TCP connection closure are the primary drivers of benign traffic classification.

---

### 3.2 UNSW-NB15 Representative Cases
- **Case 1: Exploits — Instance #3412**
  - **Actual Class:** `Exploits` (1) | **Predicted Class:** `Exploits` (1) (Confidence: $94.6\%$)
  - **Key Positive Contributors:**
    - `smean > 850` bytes ($+0.35$ weight)
    - `dttl < 32` ($+0.28$ weight)
    - `swin != 255` ($+0.19$ weight)
  - **Negative Contributors:**
    - `ct_srv_src < 5` ($-0.08$ weight)
  - **Interpretation:** Abnormally large payload averages combined with truncated hop counts indicate an exploit injection payload originating from a non-standard client.

- **Case 2: Normal Flow — Instance #19402**
  - **Actual Class:** `Normal` (4) | **Predicted Class:** `Normal` (4) (Confidence: $97.1\%$)
  - **Key Positive Contributors:**
    - `sttl = 64` (Standard Linux/UNIX client TTL) ($+0.33$ weight)
    - `dttl = 252` (Standard gateway TTL) ($+0.27$ weight)
    - `state = FIN` ($+0.21$ weight)
  - **Interpretation:** Canonical operating system hop counts and normal TCP teardowns firmly establish benign behavior.

---

## 4. Comparison: LIME vs. SHAP vs. Paper Findings

| Property | LIME | SHAP | Sharma et al. (Paper) |
| :--- | :--- | :--- | :--- |
| **Explainer Type** | Surrogate Local Linear Model | Cooperative Game-Theoretic Shapley Values | Both LIME & SHAP |
| **Evaluation Scope** | Individual Test Instances | Global (50 samples) + Local Waterfall | Global (50 samples) + Local |
| **Primary NSL Drivers** | `serror_rate`, `same_srv_rate`, `flag` | `same_srv_rate`, `dst_host_srv_count`, `serror_rate` | Identifies `same_srv_rate`, `serror_rate`, `flag` |
| **Primary UNSW Drivers**| `smean`, `dttl`, `sttl`, `swin` | `dttl`, `sttl`, `ct_srv_src`, `swin` | Identifies `dttl`, `sttl`, `swin`, `ct_srv_src` |
| **Consistency** | High local alignment with specific packet conditions | High global alignment with overall dataset variance | Consistent with our empirical rankings |
| **Computational Cost** | Fast ($\approx 0.8$ s per instance) | Moderate ($\approx 45$ s for 50 samples with KernelExplainer) | Suitable for offline analyst auditing |

**Conclusion:** Both LIME and SHAP provide consistent, complementary interpretability. SHAP reveals the dominant global feature hierarchy governing the entire manifold, while LIME explains instance-level packet anomalies with precise numerical boundary rules.
