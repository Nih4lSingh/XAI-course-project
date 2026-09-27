# Master Results Summary (6 Model Experiments)

| dataset | feature_mode | model | num_features | accuracy | precision_macro | recall_macro | f1_macro | precision_weighted | recall_weighted | f1_weighted | training_time_s | training_time_ms | experiment_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | Selected | DNN | 36 | 0.9967 | 0.8215 | 0.8057 | 0.804 | 0.9966 | 0.9967 | 0.9966 | 35.51 | 35506.4 | NSL_SELECTED_DNN |
| NSL-KDD | Selected | 1DCNN | 36 | 0.9945 | 0.8909 | 0.9011 | 0.8941 | 0.9947 | 0.9945 | 0.9946 | 72.29 | 72287.9 | NSL_SELECTED_1DCNN |
| NSL-KDD | Selected | 2DCNN | 36 | 0.995 | 0.9306 | 0.8539 | 0.8699 | 0.9951 | 0.995 | 0.9949 | 129.86 | 129859.8 | NSL_SELECTED_2DCNN |
| UNSW-NB15 | Selected | DNN | 38 | 0.8089 | 0.7656 | 0.6816 | 0.6599 | 0.8064 | 0.8089 | 0.7754 | 65.65 | 65649.9 | UNSW_SELECTED_DNN |
| UNSW-NB15 | Selected | 1DCNN | 38 | 0.8033 | 0.7693 | 0.6774 | 0.6538 | 0.8058 | 0.8033 | 0.77 | 154.01 | 154009.2 | UNSW_SELECTED_1DCNN |
| UNSW-NB15 | Selected | 2DCNN | 38 | 0.8097 | 0.7644 | 0.6834 | 0.66 | 0.8074 | 0.8097 | 0.7761 | 232.06 | 232060.6 | UNSW_SELECTED_2DCNN |

---

# Paper Reported vs Reproduction Comparison (Canonical 6 Models)

| Dataset | Model | Paper Accuracy | Our Accuracy | Difference | Paper Training Time (ms) | Our Total Training Time (s) | Runtime Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | DNN | 0.993 | 0.9967 | +0.0037 | 142.0 | 35.51 | The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons. |
| NSL-KDD | 1DCNN | 0.992 | 0.9945 | +0.0025 | 325.0 | 72.29 | The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons. |
| NSL-KDD | 2DCNN | 0.994 | 0.995 | +0.0010 | 340.0 | 129.86 | The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons. |
| UNSW-NB15 | DNN | 0.8 | 0.8089 | +0.0089 | 323.0 | 65.65 | The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons. |
| UNSW-NB15 | 1DCNN | 0.8 | 0.8033 | +0.0033 | 442.0 | 154.01 | The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons. |
| UNSW-NB15 | 2DCNN | 0.81 | 0.8097 | -0.0003 | 455.0 | 232.06 | The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons. |

*Note: Differences are reported as (Our Accuracy - Paper Accuracy). Paper training times (142/325/340 ms for NSL-KDD; 323/442/455 ms for UNSW-NB15) are labeled by the authors as training time. Our reproduction measures total 20-epoch wall-clock training time in seconds. The paper and reproduction were executed in different environments, so the reported training times are not directly hardware-normalized comparisons.*
