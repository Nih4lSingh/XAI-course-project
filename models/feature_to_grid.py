"""
Feature-to-Grid Transformation Module
Sharma et al. (2024) Replication Master Pipeline

Maps 1D tabular feature vectors into 2D spatial grid representations for Conv2D:
- NSL-KDD: 6x6x1 (36 elements, 0 padding if 36 features retained)
- UNSW-NB15: 7x7x1 (49 elements, 38 features + 11 zeros padding)
Row-major deterministic assignment.
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_selection.feature_to_grid_mapping import (
    FeatureGridMapper,
    get_nsl_kdd_grid_mapper,
    get_unsw_nb15_grid_mapper
)

__all__ = [
    "FeatureGridMapper",
    "get_nsl_kdd_grid_mapper",
    "get_unsw_nb15_grid_mapper"
]
