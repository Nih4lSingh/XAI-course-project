"""
Model architectures package for Sharma et al. (2024) replication.
"""

from models.feature_to_grid import FeatureGridMapper

def build_dnn_model(*args, **kwargs):
    from models.dnn import build_dnn_model as _build
    return _build(*args, **kwargs)

def build_cnn1d_model(*args, **kwargs):
    from models.cnn1d import build_cnn1d_model as _build
    return _build(*args, **kwargs)

def build_cnn2d_model(*args, **kwargs):
    from models.cnn2d import build_cnn2d_model as _build
    return _build(*args, **kwargs)

__all__ = [
    "build_dnn_model",
    "build_cnn1d_model",
    "build_cnn2d_model",
    "FeatureGridMapper"
]
