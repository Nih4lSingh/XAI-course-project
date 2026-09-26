# Master Results Summary (12 Experiments + Sensitivity)

| dataset | feature_mode | model | num_features | accuracy | precision_macro | recall_macro | f1_macro | precision_weighted | recall_weighted | f1_weighted | training_time_s | training_time_ms | experiment_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | Selected | DNN | 36 | 0.997 | 0.9816 | 0.8947 | 0.9242 | 0.997 | 0.997 | 0.997 | 87.01 | 87009.1 | NSL_SELECTED_DNN |
| NSL-KDD | Selected | 1DCNN | 36 | 0.9944 | 0.9736 | 0.8921 | 0.9188 | 0.9945 | 0.9944 | 0.9944 | 85.99 | 85992.8 | NSL_SELECTED_1DCNN |
| NSL-KDD | Selected | 2DCNN | 36 | 0.9967 | 0.957 | 0.8738 | 0.9035 | 0.9967 | 0.9967 | 0.9967 | 90.35 | 90353.9 | NSL_SELECTED_2DCNN |
| UNSW-NB15 | Selected | DNN | 38 | 0.8089 | 0.7656 | 0.6816 | 0.6599 | 0.8064 | 0.8089 | 0.7754 | 40.54 | 40542.2 | UNSW_SELECTED_DNN |
| UNSW-NB15 | Selected | 1DCNN | 38 | 0.8033 | 0.7693 | 0.6774 | 0.6538 | 0.8058 | 0.8033 | 0.77 | 91.23 | 91226.7 | UNSW_SELECTED_1DCNN |
| UNSW-NB15 | Selected | 2DCNN | 38 | 0.8353 | 0.7366 | 0.6764 | 0.6741 | 0.8194 | 0.8353 | 0.8139 | 161.35 | 0.0 | UNSW_SELECTED_2DCNN |
| NSL-KDD | All | DNN | 42 | 0.9971 | 0.9843 | 0.8409 | 0.8759 | 0.9971 | 0.9971 | 0.997 | 80.88 | 80880.8 | NSL_ALL_DNN |
| NSL-KDD | All | 1DCNN | 42 | 0.9953 | 0.9708 | 0.9004 | 0.9212 | 0.9954 | 0.9953 | 0.9953 | 86.57 | 86574.3 | NSL_ALL_1DCNN |
| NSL-KDD | All | 2DCNN | 42 | 0.9865 | 0.919 | 0.8693 | 0.8875 | 0.9866 | 0.9865 | 0.9865 | 104.96 | 104955.4 | NSL_ALL_2DCNN |
| UNSW-NB15 | All | DNN | 42 | 0.8352 | 0.7324 | 0.6745 | 0.6694 | 0.8178 | 0.8352 | 0.8125 | 149.69 | 149688.2 | UNSW_ALL_DNN |
| UNSW-NB15 | All | 1DCNN | 42 | 0.8321 | 0.7381 | 0.6717 | 0.6647 | 0.8201 | 0.8321 | 0.8092 | 152.51 | 152507.7 | UNSW_ALL_1DCNN |
| UNSW-NB15 | All | 2DCNN | 42 | 0.8374 | 0.7529 | 0.6748 | 0.6601 | 0.8274 | 0.8374 | 0.811 | 166.3 | 166297.3 | UNSW_ALL_2DCNN |
| NSL-KDD | Selected (Dropout=0.01) | DNN | 36 | 0.9971 | 0.9477 | 0.8794 | 0.9015 | 0.9971 | 0.9971 | 0.9971 | 30.9 | 30897.6 | NSL_SELECTED_DNN_DROPOUT_001 |
| UNSW-NB15 | Selected (Dropout=0.01) | DNN | 38 | 0.8095 | 0.7501 | 0.6863 | 0.6619 | 0.801 | 0.8095 | 0.7771 | 44.62 | 44617.0 | UNSW_SELECTED_DNN_DROPOUT_001 |

---

# Paper Reported vs Reproduction Comparison (Canonical 6 Models)

| Dataset | Model | Paper Accuracy | Our Accuracy | Difference | Paper Reported Time (ms) | Our Training Time (s) | Runtime Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | DNN | 0.993 | 0.997 | +0.0040 | 142.0 | 87.01 | Paper reports inference/step time in ms; our time is total wall-clock training for 20 epochs |
| NSL-KDD | 1DCNN | 0.992 | 0.9944 | +0.0024 | 325.0 | 85.99 | Paper reports inference/step time in ms; our time is total wall-clock training for 20 epochs |
| NSL-KDD | 2DCNN | 0.994 | 0.9967 | +0.0027 | 340.0 | 90.35 | Paper reports inference/step time in ms; our time is total wall-clock training for 20 epochs |
| UNSW-NB15 | DNN | 0.8 | 0.8089 | +0.0089 | 323.0 | 40.54 | Paper reports inference/step time in ms; our time is total wall-clock training for 20 epochs |
| UNSW-NB15 | 1DCNN | 0.8 | 0.8033 | +0.0033 | 442.0 | 91.23 | Paper reports inference/step time in ms; our time is total wall-clock training for 20 epochs |
| UNSW-NB15 | 2DCNN | 0.81 | 0.8353 | +0.0253 | 455.0 | 161.35 | Paper reports inference/step time in ms; our time is total wall-clock training for 20 epochs |

*Note: Differences are reported as (Our Accuracy - Paper Accuracy). Runtime comparison reflects distinct quantities: the paper's table presents per-sample or per-batch inference latency in milliseconds, whereas our reproduction records end-to-end training epoch duration on GPU/CPU.*
