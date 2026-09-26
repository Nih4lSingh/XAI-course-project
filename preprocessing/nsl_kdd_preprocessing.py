"""
NSL-KDD Preprocessing Module Alias / Compatibility Wrapper
Forwards to preprocessing/nsl_kdd.py
"""

from preprocessing.nsl_kdd import (
    NSL_KDD_COLUMNS,
    NSL_KDD_CATEGORICAL_COLS,
    NSL_KDD_PAPER_REMOVED_FEATURES,
    load_raw_nsl_kdd,
    process_nsl_kdd
)

__all__ = [
    "NSL_KDD_COLUMNS",
    "NSL_KDD_CATEGORICAL_COLS",
    "NSL_KDD_PAPER_REMOVED_FEATURES",
    "load_raw_nsl_kdd",
    "process_nsl_kdd"
]

if __name__ == "__main__":
    process_nsl_kdd()
