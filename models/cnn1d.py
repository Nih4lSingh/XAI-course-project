"""
1D Convolutional Neural Network (1D-CNN) Architecture
Replication of Sharma et al. (2024)

Paper Specification (Section 4.2):
- Kernel size = 3 (EXPLICIT)
- Max Pooling size = 2 (EXPLICIT)
- Activation = ReLU (EXPLICIT)
- Optimizer = Adam(learning_rate=0.001) (EXPLICIT)
- Weight Decay = 0.0001 (EXPLICIT)
- Output Layer = Dense(5, activation='softmax') (EXPLICIT)
- Epochs = 20 (EXPLICIT)

Inferred Architecture (documented in replication_deviations.md):
- Layer 1: Conv1D(filters=64, kernel_size=3, padding='same', activation='relu') [INFERRED]
- MaxPool1D(pool_size=2) [EXPLICIT]
- Layer 2: Conv1D(filters=32, kernel_size=3, padding='same', activation='relu') [INFERRED]
- Flatten [INFERRED]
- Dense(5, activation='softmax') [EXPLICIT]
"""

from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_cnn1d_model(
    input_dim: int,
    num_classes: int = 5,
    filters_conv1: int = 64,
    filters_conv2: int = 32,
    kernel_size: int = 3,
    pool_size: int = 2,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    name: str = "Sharma_1DCNN"
) -> keras.Model:
    """
    Constructs and compiles the 1D-CNN according to explicit paper parameters and inferred filters.
    """
    l2_reg = regularizers.l2(weight_decay) if weight_decay > 0 else None

    # Input shape: (feature_dim, 1)
    inputs = keras.Input(shape=(input_dim, 1), name="sequence_input")
    
    # Conv1D Layer 1
    x = layers.Conv1D(
        filters=filters_conv1,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv1d_1"
    )(inputs)
    
    # Max Pooling 1D
    x = layers.MaxPooling1D(pool_size=pool_size, name="maxpool1d_1")(x)
    
    # Conv1D Layer 2
    x = layers.Conv1D(
        filters=filters_conv2,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv1d_2"
    )(x)
    
    # Flatten
    x = layers.Flatten(name="flatten")(x)
    
    # Output Layer
    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
