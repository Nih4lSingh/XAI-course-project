# Master Results Summary (12 Model Experiments)

| dataset   | feature_mode            | model   |   num_features |   accuracy |   precision_macro |   recall_macro |   f1_macro |   precision_weighted |   recall_weighted |   f1_weighted |   training_time_s |   training_time_ms | experiment_id                 |
|:----------|:------------------------|:--------|---------------:|-----------:|------------------:|---------------:|-----------:|---------------------:|------------------:|--------------:|------------------:|-------------------:|:------------------------------|
| NSL-KDD   | Selected                | DNN     |             36 |     0.997  |            0.9816 |         0.8947 |     0.9242 |               0.997  |            0.997  |        0.997  |             85.14 |            85137.7 | NSL_SELECTED_DNN              |
| NSL-KDD   | Selected                | 1DCNN   |             36 |     0.9944 |            0.9736 |         0.8921 |     0.9188 |               0.9945 |            0.9944 |        0.9944 |             85.99 |            85992.8 | NSL_SELECTED_1DCNN            |
| NSL-KDD   | Selected                | 2DCNN   |             36 |     0.9967 |            0.957  |         0.8738 |     0.9035 |               0.9967 |            0.9967 |        0.9967 |             90.35 |            90353.9 | NSL_SELECTED_2DCNN            |
| NSL-KDD   | All                     | DNN     |             42 |     0.9971 |            0.9843 |         0.8409 |     0.8759 |               0.9971 |            0.9971 |        0.997  |             80.88 |            80880.8 | NSL_ALL_DNN                   |
| NSL-KDD   | All                     | 1DCNN   |             42 |     0.9953 |            0.9708 |         0.9004 |     0.9212 |               0.9954 |            0.9953 |        0.9953 |             86.57 |            86574.3 | NSL_ALL_1DCNN                 |
| NSL-KDD   | All                     | 2DCNN   |             49 |     0.9865 |            0.919  |         0.8693 |     0.8875 |               0.9866 |            0.9865 |        0.9865 |            104.96 |           104955   | NSL_ALL_2DCNN                 |
| UNSW-NB15 | Selected                | DNN     |             36 |     0.8312 |            0.7485 |         0.6562 |     0.6447 |               0.8174 |            0.8312 |        0.8011 |            153.11 |           153107   | UNSW_SELECTED_DNN             |
| UNSW-NB15 | Selected                | 1DCNN   |             36 |     0.8312 |            0.7378 |         0.6658 |     0.6533 |               0.8187 |            0.8312 |        0.805  |            156.94 |           156941   | UNSW_SELECTED_1DCNN           |
| UNSW-NB15 | Selected                | 2DCNN   |             49 |     0.8353 |            0.7366 |         0.6764 |     0.6741 |               0.8194 |            0.8353 |        0.8139 |            161.35 |           161346   | UNSW_SELECTED_2DCNN           |
| UNSW-NB15 | All                     | DNN     |             42 |     0.8352 |            0.7324 |         0.6745 |     0.6694 |               0.8178 |            0.8352 |        0.8125 |            149.69 |           149688   | UNSW_ALL_DNN                  |
| UNSW-NB15 | All                     | 1DCNN   |             42 |     0.8321 |            0.7381 |         0.6717 |     0.6647 |               0.8201 |            0.8321 |        0.8092 |            152.51 |           152508   | UNSW_ALL_1DCNN                |
| UNSW-NB15 | All                     | 2DCNN   |             49 |     0.8374 |            0.7529 |         0.6748 |     0.6601 |               0.8274 |            0.8374 |        0.811  |            166.3  |           166297   | UNSW_ALL_2DCNN                |
| NSL-KDD   | Selected (Dropout=0.01) | DNN     |             36 |     0.997  |            0.9816 |         0.8947 |     0.9242 |               0.997  |            0.997  |        0.997  |             87.01 |            87009.1 | NSL_SELECTED_DNN_DROPOUT_001  |
| UNSW-NB15 | Selected (Dropout=0.01) | DNN     |             38 |     0.8093 |            0.765  |         0.692  |     0.6664 |               0.8131 |            0.8093 |        0.7792 |            122.32 |           122324   | UNSW_SELECTED_DNN_DROPOUT_001 |

# Paper Reported vs Reproduction Comparison

| Dataset   | Model   |   Paper Accuracy |   Our Accuracy |   Difference |   Paper Time (ms) |   Our Time (s) |
|:----------|:--------|-----------------:|---------------:|-------------:|------------------:|---------------:|
| NSL-KDD   | DNN     |            0.993 |         0.997  |       0.004  |               142 |          85.14 |
| NSL-KDD   | 1DCNN   |            0.992 |         0.9944 |       0.0024 |               325 |          85.99 |
| NSL-KDD   | 2DCNN   |            0.994 |         0.9967 |       0.0027 |               340 |          90.35 |
| UNSW-NB15 | DNN     |            0.8   |         0.8312 |       0.0312 |               323 |         153.11 |
| UNSW-NB15 | 1DCNN   |            0.8   |         0.8312 |       0.0312 |               442 |         156.94 |
| UNSW-NB15 | 2DCNN   |            0.81  |         0.8353 |       0.0253 |               455 |         161.35 |