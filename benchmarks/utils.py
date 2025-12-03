"""
Benchmarking utilities for feature_blocks performance analysis.

Provides tools for:
- Memory usage tracking (peak and current)
- Execution time measurement
- Resource monitoring
- Results storage and export
"""

from __future__ import annotations
import time
import psutil
import os
import json
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
from pathlib import Path

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Tuple, Any
import json
import numpy as np
import pandas as pd

@dataclass
class BenchmarkResults:
    """Unified benchmark result + suite container."""

    # Suite metadata
    suite_name: str = "feature_blocks_benchmarks"
    results: List[BenchmarkResults] = field(default_factory=list, repr=False)

    # Individual benchmark fields
    name: Optional[str] = None
    image_size: Optional[Tuple[int, int]] = None
    n_chunks: Optional[int] = None
    block_size: Optional[int] = None
    chunk_size: Optional[int] = None
    shard_size: Optional[int] = None
    n_workers: Optional[int] = None
    batch_size: Optional[int] = None
    backend: Optional[str] = None  # e.g. "dask", "sequential"
    model_name: Optional[str] = None
    method: Optional[str] = None  # e.g. "zarr_dask", "gpu_batch", "in_memory"

    # Performance metrics
    total_time: Optional[float] = None
    inference_time: Optional[float] = None
    io_time: Optional[float] = None
    setup_time: Optional[float] = None

    # Memory metrics (in MB)
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0

    # Throughput metrics
    regions_per_second: float = 0.0
    pixels_per_second: float = 0.0

    # Metadata
    n_features: int = 0
    gpu_available: bool = False
    device: str = "cpu"

    def add_result(self, result: BenchmarkResults):
        """Add another BenchmarkResults (as a result) to this suite."""
        if not isinstance(result, BenchmarkResults):
            raise TypeError("Expected a BenchmarkResults instance.")
        self.results.append(result)

    def to_dict(self):
        """Convert to dict recursively."""
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d

    def save(self, path: str):
        """Save suite or single result to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved to {path}")

    def save_csv(self, path: str, append: bool = True):
        """Save results to CSV, optionally appending to existing file.

        Args:
            path: Path to CSV file
            append: If True, append to existing file; if False, overwrite
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to DataFrame
        if self.results:
            rows = []
            for r in self.results:
                row = asdict(r)
                row.pop('results', None)
                rows.append(row)
            df = pd.DataFrame(rows)
        else:
            row = asdict(self)
            row.pop('results', None)
            df = pd.DataFrame([row])

        if append and path.exists():
            df.to_csv(path, mode='a', header=False, index=False)
        else:
            df.to_csv(path, index=False)

        print(f"Saved {len(df)} result(s) to {path}")

    @classmethod
    def load(cls, path: str) -> BenchmarkResults:
        """Load suite or result from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        def from_dict(d: dict) -> BenchmarkResults:
            results = [from_dict(r) for r in d.pop("results", [])]
            obj = cls(**d)
            obj.results = results
            if obj.image_size and isinstance(obj.image_size, list):
                obj.image_size = tuple(obj.image_size)
            return obj

        return from_dict(data)


class MemoryTracker:
    """Track memory usage during execution."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_mb = self._get_memory_mb()
        self.peak_mb = self.baseline_mb
        self.samples = []

    def _get_memory_mb(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def sample(self):
        """Take a memory sample."""
        current_mb = self._get_memory_mb()
        self.samples.append(current_mb)
        self.peak_mb = max(self.peak_mb, current_mb)
        return current_mb

    def get_stats(self):
        """Get memory statistics."""
        if not self.samples:
            self.sample()
        return {
            "peak_mb": self.peak_mb,
            "avg_mb": np.mean(self.samples),
            "baseline_mb": self.baseline_mb,
            "delta_peak_mb": self.peak_mb - self.baseline_mb,
            "delta_avg_mb": np.mean(self.samples) - self.baseline_mb
        }


def estimate_n_chunks(image_size: tuple, block_size: int, block_method: str = "block") -> int:
    """
    Estimate number of chunks/tasks for given parameters.

    Args:
        image_size: (C, Z, H, W) or (H, W)
        block_size: Size of processing blocks
        block_method: "block" or "centroid"

    Returns:
        Estimated number of chunks
    """
    if len(image_size) == 4:
        _, _, H, W = image_size
    elif len(image_size) == 2:
        H, W = image_size
    else:
        raise ValueError(f"Invalid image_size: {image_size}")

    if block_method == "block":
        n_h = int(np.ceil(H / block_size))
        n_w = int(np.ceil(W / block_size))
        return n_h * n_w
    else:
        # For centroid method, this would depend on segmentation
        # Return approximate based on uniform distribution
        return int((H * W) / (block_size * block_size))


def format_size(size_bytes: int) -> str:
    """Format byte size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
