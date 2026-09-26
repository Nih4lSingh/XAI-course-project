"""
Source package for Sharma et al. (2024) replication.
Exposes all submodules: preprocessing, models, training, evaluation, xai, feature_selection, data.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
