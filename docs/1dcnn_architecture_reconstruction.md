# 1D-CNN Architecture Reconstruction: Sharma et al. (2024)

This document provides complete traceability for the 1D Convolutional Neural Network (1D-CNN) implementation in the replication of **Sharma et al. (2024)**.

---

## 1. Specification Breakdown: Explicit vs. Inferred

The published paper explicitly defines certain 1D-CNN hyperparameters in Section 4.2, while omitting the exact filter count, layer depth, and tensor padding dimensions. To ensure scientific rigor without inventing details silently, every parameter is classified below:

| Parameter | Implemented Value | Paper Status | Source / Rationale |
| :--- | :---: | :---: | :--- |
| **Kernel Size** | 3 | `EXPLICIT` | Section 4.2: *"kernel size of 3 is used"* |
| **Pooling Size** | 2 | `EXPLICIT` | Section 4.2: *"max pooling with pool size 2"* |
| **Activation Function** | ReLU | `EXPLICIT` | Section 4.2: *"ReLU activation function"* |
| **Optimizer** | Adam | `EXPLICIT` | Section 4.2: *"Adam optimizer"* |
| **Learning Rate** | 0.001 | `EXPLICIT` | Section 4.2: *"learning rate 0.001"* |
| **Weight Decay** | 0.0001 | `EXPLICIT` | Section 4.2: *"weight decay 0.0001"* |
| **Epochs** | 20 | `EXPLICIT` | Section 4.2: *"trained for 20 epochs"* |
| **Output Layer** | Dense(5, Softmax) | `EXPLICIT` | Section 4.2: 5 canonical attack classes |
| **Convolutional Filters** | 64 $\to$ 32 | `INFERRED` | Inferred to mirror the 2D-CNN filter progression (64 $\to$ 32) |
| **Convolutional Layers** | 2 layers | `INFERRED` | Standard minimal hierarchical feature extractor |
| **Padding** | `'same'` | `INFERRED` | Prevents premature sequence collapse before pooling |
| **Stride** | 1 | `INFERRED` | Standard convolution stride |
| **Batch Size** | 64 | `INFERRED` | Standard DL batch size matching GPU memory alignment |

---

## 2. Canonical Keras Computational Graph

```text
Input Tensor: (batch_size, num_features, 1)
  │
  ▼
Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=L2(1e-4))
  │
  ▼
MaxPooling1D(pool_size=2)
  │
  ▼
Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=L2(1e-4))
  │
  ▼
Flatten()
  │
  ▼
Dense(units=5, activation='softmax')
```

---

## 3. Scientific Impact of Inferred Architecture
Because the filter count was not stated in the paper text, using 64 $\to$ 32 filters provides a balanced capacity that matches the empirical runtime and training progression reported by Sharma et al., without overfitting on small tabular sequences.
