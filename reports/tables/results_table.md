# Master Results Summary (6 Model Experiments)

| dataset | feature_mode | model | num_features | accuracy | precision_macro | recall_macro | f1_macro | precision_weighted | recall_weighted | f1_weighted | training_time_s | training_time_ms | experiment_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | Selected | DNN | 36 | 0.9966 | 0.9659 | 0.8056 | 0.8058 | 0.9968 | 0.9966 | 0.9965 | 18.77 | 18773.3 | NSL_SELECTED_DNN |
| NSL-KDD | Selected | 1DCNN | 36 | 0.9947 | 0.9107 | 0.8978 | 0.9017 | 0.9948 | 0.9947 | 0.9947 | 53.05 | 53052.5 | NSL_SELECTED_1DCNN |
| NSL-KDD | Selected | 2DCNN | 36 | 0.9962 | 0.9711 | 0.8051 | 0.8085 | 0.9963 | 0.9962 | 0.9961 | 103.9 | 103904.9 | NSL_SELECTED_2DCNN |
| UNSW-NB15 | Selected | DNN | 38 | 0.8052 | 0.751 | 0.6848 | 0.6627 | 0.8028 | 0.8052 | 0.7754 | 26.46 | 26455.8 | UNSW_SELECTED_DNN |
| UNSW-NB15 | Selected | 1DCNN | 38 | 0.7995 | 0.7796 | 0.6684 | 0.6454 | 0.806 | 0.7995 | 0.7639 | 75.45 | 75451.7 | UNSW_SELECTED_1DCNN |
| UNSW-NB15 | Selected | 2DCNN | 38 | 0.8079 | 0.7578 | 0.6845 | 0.6595 | 0.8067 | 0.8079 | 0.7755 | 166.65 | 166647.7 | UNSW_SELECTED_2DCNN |

# Paper Reported vs Reproduction Comparison

| Dataset | Model | Paper Accuracy | Our Accuracy | Difference | Paper Time (ms) | Our Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | DNN | 0.993 | 0.9966 | 0.0036 | 142.0 | 18.77 |
| NSL-KDD | 1DCNN | 0.992 | 0.9947 | 0.0027 | 325.0 | 53.05 |
| NSL-KDD | 2DCNN | 0.994 | 0.9962 | 0.0022 | 340.0 | 103.9 |
| UNSW-NB15 | DNN | 0.8 | 0.8052 | 0.0052 | 323.0 | 26.46 |
| UNSW-NB15 | 1DCNN | 0.8 | 0.7995 | -0.0005 | 442.0 | 75.45 |
| UNSW-NB15 | 2DCNN | 0.81 | 0.8079 | -0.0021 | 455.0 | 166.65 |