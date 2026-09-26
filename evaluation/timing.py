"""
High-Resolution Timing & Hardware Profiler
Replication of Sharma et al. (2024)

Tracks:
- Total training duration (seconds & milliseconds)
- Average per-epoch step time
- Per-sample inference latency
- Full system hardware & software specifications
"""

import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np


class PerformanceTimer:
    """
    High-resolution timer for training and inference benchmarking.
    """
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_seconds: float = 0.0
        self.total_ms: float = 0.0
        self.epoch_times: list = []

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.total_seconds = self.end_time - self.start_time
            self.total_ms = self.total_seconds * 1000.0
        return self.total_ms

    def record_epoch(self, epoch_seconds: float):
        self.epoch_times.append(epoch_seconds)

    @staticmethod
    def get_hardware_environment() -> Dict:
        """Collects OS, CPU, RAM, and GPU hardware profile."""
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        gpu_names = [gpu.name for gpu in gpus] if gpus else ["None (CPU execution)"]

        return {
            "os": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "gpu_devices": gpu_names,
            "has_gpu": len(gpus) > 0
        }

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "total_training_time_seconds": round(self.total_seconds, 4),
            "total_training_time_ms": round(self.total_ms, 2),
            "num_epochs_recorded": len(self.epoch_times),
            "avg_epoch_time_seconds": round(float(np.mean(self.epoch_times)), 4) if self.epoch_times else None,
            "hardware_environment": self.get_hardware_environment()
        }

    def save(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
