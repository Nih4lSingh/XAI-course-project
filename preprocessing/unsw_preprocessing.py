"""
UNSW-NB15 Preprocessing Module Alias / Compatibility Wrapper
Forwards to preprocessing/unsw_nb15.py
"""

from preprocessing.unsw_nb15 import (
    UNSW_NON_PREDICTORS,
    UNSW_CATEGORICAL_COLS,
    UNSW_PAPER_REMOVED_PREDICTORS,
    load_raw_unsw_nb15,
    apply_sampling_policy,
    process_unsw_nb15
)

__all__ = [
    "UNSW_NON_PREDICTORS",
    "UNSW_CATEGORICAL_COLS",
    "UNSW_PAPER_REMOVED_PREDICTORS",
    "load_raw_unsw_nb15",
    "apply_sampling_policy",
    "process_unsw_nb15"
]

if __name__ == "__main__":
    process_unsw_nb15()
