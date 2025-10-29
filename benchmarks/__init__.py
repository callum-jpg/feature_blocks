"""
Feature_blocks benchmarking suite.

Comprehensive performance analysis for zarr+dask feature extraction.
"""

from benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSuite,
    MemoryTracker,
    benchmark_context
)

from benchmarks.synthetic_data import (
    create_test_scenario,
    create_scaling_scenarios
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "MemoryTracker",
    "benchmark_context",
    "create_test_scenario",
    "create_scaling_scenarios",
]
