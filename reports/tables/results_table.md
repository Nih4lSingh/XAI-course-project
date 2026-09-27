# Master Results Summary (12 Experiments + Sensitivity)

| dataset | feature_mode | model | num_features | accuracy | precision_macro | recall_macro | f1_macro | precision_weighted | recall_weighted | f1_weighted | training_time_s | training_time_ms | experiment_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | Selected | DNN | 36 | 0.9967 | 0.8215 | 0.8057 | 0.804 | 0.9966 | 0.9967 | 0.9966 | 35.51 | 35506.4 | NSL_SELECTED_DNN |
| NSL-KDD | Selected | 1DCNN | 36 | 0.9945 | 0.8909 | 0.9011 | 0.8941 | 0.9947 | 0.9945 | 0.9946 | 72.29 | 72287.9 | NSL_SELECTED_1DCNN |
| NSL-KDD | Selected | 2DCNN | 36 | 0.995 | 0.9306 | 0.8539 | 0.8699 | 0.9951 | 0.995 | 0.9949 | 129.86 | 129859.8 | NSL_SELECTED_2DCNN |
| UNSW-NB15 | Selected | DNN | 38 | 0.8089 | 0.7656 | 0.6816 | 0.6599 | 0.8064 | 0.8089 | 0.7754 | 65.65 | 65649.9 | UNSW_SELECTED_DNN |
| UNSW-NB15 | Selected | 1DCNN | 38 | 0.8033 | 0.7693 | 0.6774 | 0.6538 | 0.8058 | 0.8033 | 0.77 | 154.01 | 154009.2 | UNSW_SELECTED_1DCNN |
| UNSW-NB15 | Selected | 2DCNN | 38 | 0.8097 | 0.7644 | 0.6834 | 0.66 | 0.8074 | 0.8097 | 0.7761 | 232.06 | 232060.6 | UNSW_SELECTED_2DCNN |
| NSL-KDD | All | DNN | 42 | 0.9966 | 0.8555 | 0.8379 | 0.8402 | 0.9967 | 0.9966 | 0.9966 | 35.46 | 35464.6 | NSL_ALL_DNN |
| NSL-KDD | All | 1DCNN | 42 | 0.9953 | 0.9435 | 0.8995 | 0.9128 | 0.9954 | 0.9953 | 0.9953 | 80.61 | 80608.2 | NSL_ALL_1DCNN |
| NSL-KDD | All | 2DCNN | 42 | 0.9939 | 0.9645 | 0.8686 | 0.8884 | 0.9941 | 0.9939 | 0.9939 | 149.28 | 149279.2 | NSL_ALL_2DCNN |
| UNSW-NB15 | All | DNN | 42 | 0.8105 | 0.7743 | 0.6877 | 0.6654 | 0.8129 | 0.8105 | 0.7788 | 62.87 | 62871.2 | UNSW_ALL_DNN |
| UNSW-NB15 | All | 1DCNN | 42 | 0.8046 | 0.7962 | 0.6784 | 0.6543 | 0.8178 | 0.8046 | 0.7706 | 142.94 | 142936.4 | UNSW_ALL_1DCNN |
| UNSW-NB15 | All | 2DCNN | 42 | 0.8089 | 0.7691 | 0.686 | 0.6605 | 0.8108 | 0.8089 | 0.776 | 242.43 | 242429.6 | UNSW_ALL_2DCNN |
| NSL-KDD | Selected (Dropout=0.01) | DNN | 36 | 0.9971 | 0.9477 | 0.8794 | 0.9015 | 0.9971 | 0.9971 | 0.9971 | 51.54 | 51538.8 | NSL_SELECTED_DNN_DROPOUT_001 |
| UNSW-NB15 | Selected (Dropout=0.01) | DNN | 38 | 0.8095 | 0.7501 | 0.6863 | 0.6619 | 0.801 | 0.8095 | 0.7771 | 74.92 | 74921.1 | UNSW_SELECTED_DNN_DROPOUT_001 |

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
