"""
Deep Neural Network (DNN) Architecture
Replication of Sharma et al. (2024)

Paper Specification (Section 4.1):
- Input Layer: matches feature dimension (Selected or All features)
- Hidden Layer 1: Dense(64, activation='relu', kernel_regularizer=L2(1e-4))
- (Optional Dropout): configurable (0.00 vs 0.01) to evaluate paper ambiguity
- Hidden Layer 2: Dense(64, activation='relu', kernel_regularizer=L2(1e-4))
- (Optional Dropout): configurable (0.00 vs 0.01)
- Hidden Layer 3: Dense(64, activation='relu', kernel_regularizer=L2(1e-4))
- Output Layer: Dense(5, activation='softmax')

Optimizer: Adam(learning_rate=0.001)
Weight Decay: 0.0001
Loss: Sparse Categorical Cross-Entropy
Epochs: 20
"""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_dnn_model(
    input_dim: int,
    num_classes: int = 5,
    dropout_rate: float = 0.0,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    name: str = "Sharma_DNN"
) -> keras.Model:
    """
    Constructs and compiles the paper-faithful 3-layer DNN.
    """
    l2_reg = regularizers.l2(weight_decay) if weight_decay > 0 else None

    inputs = keras.Input(shape=(input_dim,), name="feature_input")
    
    # Layer 1
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2_reg, name="dense_1")(inputs)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="dropout_1")(x)
        
    # Layer 2
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2_reg, name="dense_2")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="dropout_2")(x)
        
    # Layer 3
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2_reg, name="dense_3")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="dropout_3")(x)
        
    # Output Layer
    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
