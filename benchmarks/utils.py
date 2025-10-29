"""
Benchmarking utilities for feature_blocks performance analysis.

Provides tools for:
- Memory usage tracking (peak and current)
- Execution time measurement
- Resource monitoring
- Results storage and export
"""

import time
import psutil
import os
import json
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    image_size: tuple
    n_chunks: int
    block_size: int
    n_workers: int
    model_name: str
    method: str  # "zarr_dask", "gpu_batch", "in_memory"

    # Performance metrics
    total_time: float
    inference_time: float
    io_time: Optional[float] = None
    setup_time: Optional[float] = None

    # Memory metrics (in MB)
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0

    # Throughput metrics
    regions_per_second: float = 0.0
    pixels_per_second: float = 0.0

    # Additional metadata
    n_features: int = 0
    gpu_available: bool = False
    device: str = "cpu"

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self):
        return (
            f"{self.name}: {self.method} | "
            f"Size: {self.image_size} | "
            f"Chunks: {self.n_chunks} | "
            f"Time: {self.total_time:.2f}s | "
            f"Memory: {self.peak_memory_mb:.1f}MB | "
            f"Throughput: {self.regions_per_second:.1f} regions/s"
        )


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


@contextmanager
def benchmark_context(name: str = "benchmark"):
    """
    Context manager for benchmarking code execution.

    Usage:
        with benchmark_context("my_test") as tracker:
            # code to benchmark
            pass

        stats = tracker.get_stats()
        print(f"Time: {stats['elapsed']:.2f}s")
        print(f"Peak memory: {stats['memory']['peak_mb']:.1f}MB")
    """
    tracker = {
        "name": name,
        "start_time": time.time(),
        "memory": MemoryTracker()
    }

    def get_stats():
        elapsed = time.time() - tracker["start_time"]
        mem_stats = tracker["memory"].get_stats()
        return {
            "name": name,
            "elapsed": elapsed,
            "memory": mem_stats
        }

    tracker["get_stats"] = get_stats

    try:
        yield tracker
    finally:
        tracker["memory"].sample()


class BenchmarkSuite:
    """Manage a collection of benchmark results."""

    def __init__(self, name: str = "feature_blocks_benchmarks"):
        self.name = name
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)

    def save(self, path: str):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "n_results": len(self.results),
            "results": [r.to_dict() for r in self.results]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.results)} results to {path}")

    @classmethod
    def load(cls, path: str):
        """Load results from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        suite = cls(name=data["name"])
        for r_dict in data["results"]:
            # Convert tuple strings back to tuples
            if isinstance(r_dict["image_size"], list):
                r_dict["image_size"] = tuple(r_dict["image_size"])
            suite.results.append(BenchmarkResult(**r_dict))

        return suite

    def filter(self, **kwargs) -> List[BenchmarkResult]:
        """Filter results by attributes."""
        filtered = self.results
        for key, value in kwargs.items():
            filtered = [r for r in filtered if getattr(r, key) == value]
        return filtered

    def summary(self) -> str:
        """Generate summary statistics."""
        if not self.results:
            return "No results"

        lines = [f"Benchmark Suite: {self.name}", f"Total runs: {len(self.results)}", ""]

        # Group by method
        methods = set(r.method for r in self.results)
        for method in sorted(methods):
            method_results = self.filter(method=method)
            avg_time = np.mean([r.total_time for r in method_results])
            avg_mem = np.mean([r.peak_memory_mb for r in method_results])
            lines.append(f"{method}:")
            lines.append(f"  Runs: {len(method_results)}")
            lines.append(f"  Avg time: {avg_time:.2f}s")
            lines.append(f"  Avg peak memory: {avg_mem:.1f}MB")
            lines.append("")

        return "\n".join(lines)


def measure_function(func: Callable, *args, track_memory: bool = True, **kwargs):
    """
    Measure execution time and memory usage of a function.

    Args:
        func: Function to measure
        *args: Function arguments
        track_memory: Whether to track memory usage
        **kwargs: Function keyword arguments

    Returns:
        Tuple of (result, elapsed_time, memory_stats)
    """
    if track_memory:
        memory_tracker = MemoryTracker()

    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time

    memory_stats = memory_tracker.get_stats() if track_memory else None

    return result, elapsed, memory_stats


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
