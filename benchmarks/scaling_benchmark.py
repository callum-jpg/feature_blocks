"""
Scaling benchmark for feature_blocks zarr+dask approach.

Tests performance across different image sizes and worker counts to identify:
1. Overhead of zarr+dask for small images
2. Scaling efficiency for large images
3. Optimal worker count
"""

import time
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import shutil
import traceback
import dask.array
import tempfile
import os

from feature_blocks.features._extract import extract
from utils import (
    BenchmarkResult,
    BenchmarkSuite,
    MemoryTracker,
    estimate_n_chunks
)
from synthetic_data import create_test_scenario, create_scaling_scenarios


def benchmark_zarr_dask_extraction(
    input_zarr_path: str,
    output_zarr_path: str,
    model_name: str,
    block_size: int,
    n_workers: int,
    image_size: Tuple[int, int, int, int],
    block_method: str = "block",
    segmentations_path: Optional[str] = None,
    backend: str = "local",
    track_memory: bool = True
) -> BenchmarkResult:
    """
    Benchmark feature_blocks extraction with zarr+dask.

    Args:
        input_zarr_path: Path to input zarr
        output_zarr_path: Path to output zarr
        model_name: Name of feature extraction model
        block_size: Size of processing blocks
        n_workers: Number of dask workers
        image_size: Image dimensions (C, Z, H, W)
        block_method: "block" or "centroid"
        segmentations_path: Path to segmentations GeoJSON (for centroid method)
        backend: Dask backend ("local", "slurm", etc.)
        track_memory: Whether to track memory usage

    Returns:
        BenchmarkResult object
    """
    if track_memory:
        mem_tracker = MemoryTracker()

    # Estimate number of chunks
    n_chunks = estimate_n_chunks(image_size, block_size, block_method)

    print(f"\nBenchmarking zarr+dask extraction:")
    print(f"  Model: {model_name}")
    print(f"  Image size: {image_size}")
    print(f"  Block size: {block_size}")
    print(f"  Block method: {block_method}")
    print(f"  Workers: {n_workers}")
    print(f"  Expected chunks: {n_chunks}")

    # Run extraction
    start_time = time.time()

    try:
        extract(
            input_zarr_path=input_zarr_path,
            feature_extraction_method=model_name,
            block_size=block_size,
            output_zarr_path=output_zarr_path,
            n_workers=n_workers,
            block_method=block_method,
            segmentations=segmentations_path,
            # backend=backend
        )

        total_time = time.time() - start_time

        if track_memory:
            mem_tracker.sample()

        print(f"✓ Completed in {total_time:.2f}s")

        # Get memory stats
        memory_stats = mem_tracker.get_stats() if track_memory else {}

        # Calculate throughput
        _, _, H, W = image_size
        pixels_total = H * W
        regions_per_second = n_chunks / total_time
        pixels_per_second = pixels_total / total_time

        # Infer number of features from output
        output_z = dask.array.from_zarr(input_zarr_path, component=0)
        n_features = output_z.shape[0] if block_method == "block" else output_z.shape[1]

        result = BenchmarkResult(
            name=f"zarr_dask_{model_name}",
            image_size=image_size,
            n_chunks=n_chunks,
            block_size=block_size,
            n_workers=n_workers,
            model_name=model_name,
            method="zarr_dask",
            total_time=total_time,
            inference_time=total_time,  # Approximation
            io_time=None,
            setup_time=None,
            peak_memory_mb=memory_stats.get("peak_mb", 0.0),
            avg_memory_mb=memory_stats.get("avg_mb", 0.0),
            regions_per_second=regions_per_second,
            pixels_per_second=pixels_per_second,
            n_features=n_features,
            gpu_available=False,
            device="cpu"
        )

        return result

    except Exception as e:
        traceback.print_exc()
        raise


def run_scaling_benchmark(
    scenarios: List[dict],
    model_name: str = "dummy",
    block_size: int = 112,
    worker_counts: List[int] = [1, 2, 4, 8],
    output_dir: Optional[str] = None
) -> BenchmarkSuite:
    """
    Run scaling benchmarks across multiple scenarios and worker counts.

    Args:
        scenarios: List of test scenarios from create_test_scenario()
        model_name: Model to use for benchmarking
        block_size: Block size
        worker_counts: List of worker counts to test
        output_dir: Directory for output zarr files

    Returns:
        BenchmarkSuite with all results
    """
    suite = BenchmarkSuite(name="zarr_dask_scaling")

    output_dir = Path(output_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running Zarr+Dask Scaling Benchmark")
    print(f"Model: {model_name}")
    print(f"Block size: {block_size}")
    print(f"Worker counts: {worker_counts}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"{'='*60}\n")

    for scenario in scenarios:
        scenario_name = scenario["name"]
        input_zarr = os.path.abspath(scenario["image_path"])
        image_size = scenario["image_size"]

        print(f"\n--- Scenario: {scenario_name} ---")
        print(f"Image size: {image_size}")
        print(f"Input zarr: {input_zarr}")

        for n_workers in worker_counts:
            output_zarr = output_dir / f"{scenario_name}_w{n_workers}.zarr"

            # Remove output if exists
            if output_zarr.exists():
                shutil.rmtree(output_zarr)

            try:
                result = benchmark_zarr_dask_extraction(
                    input_zarr_path=input_zarr,
                    output_zarr_path=str(output_zarr),
                    model_name=model_name,
                    block_size=block_size,
                    n_workers=n_workers,
                    image_size=image_size,
                    block_method="block",
                    backend="local"
                )

                suite.add_result(result)
                print(f"  → {result}")

            except Exception as e:
                print(f"  ✗ Failed with {n_workers} workers: {e}")

            # Cleanup output
            if output_zarr.exists():
                shutil.rmtree(output_zarr)

    print(f"\n{'='*60}")
    print(f"Benchmark completed: {len(suite.results)} results")
    print(f"{'='*60}\n")

    return suite


def run_worker_scaling_analysis(
    image_size: Tuple[int, int, int, int],
    model_name: str = "dummy",
    block_size: int = 112,
    max_workers: int = 16,
    temp_dir: Optional[str] = None
) -> BenchmarkSuite:
    """
    Analyze worker scaling for a single image size.

    Tests with 1, 2, 4, 8, ... up to max_workers to find optimal parallelism.

    Args:
        image_size: Image dimensions (C, Z, H, W)
        model_name: Model name
        block_size: Block size
        max_workers: Maximum number of workers to test
        temp_dir: Temporary directory for test data

    Returns:
        BenchmarkSuite with results
    """
    # Create test scenario
    scenario = create_test_scenario(
        name=f"scaling_{image_size[2]}x{image_size[3]}",
        image_size=image_size,
        base_dir=temp_dir
    )

    # Generate worker counts: 1, 2, 4, 8, 16, ...
    worker_counts = []
    n = 1
    while n <= max_workers:
        worker_counts.append(n)
        n *= 2

    # Run benchmark
    suite = run_scaling_benchmark(
        scenarios=[scenario],
        model_name=model_name,
        block_size=block_size,
        worker_counts=worker_counts
    )

    return suite


def compare_block_sizes(
    image_size: Tuple[int, int, int, int],
    model_name: str = "dummy",
    block_sizes: List[int] = [64, 112, 224, 512],
    n_workers: int = 4,
    temp_dir: Optional[str] = None
) -> BenchmarkSuite:
    """
    Compare performance across different block sizes.

    Args:
        image_size: Image dimensions
        model_name: Model name
        block_sizes: List of block sizes to test
        n_workers: Number of workers
        temp_dir: Temporary directory

    Returns:
        BenchmarkSuite with results
    """
    suite = BenchmarkSuite(name="block_size_comparison")

    # Create test scenario
    scenario = create_test_scenario(
        name=f"blocksize_{image_size[2]}x{image_size[3]}",
        image_size=image_size,
        base_dir=temp_dir
    )

    output_dir = Path(tempfile.mkdtemp(prefix="benchmark_blocksize_"))

    print(f"\n{'='*60}")
    print(f"Comparing Block Sizes")
    print(f"Image size: {image_size}")
    print(f"Block sizes: {block_sizes}")
    print(f"Workers: {n_workers}")
    print(f"{'='*60}\n")

    for block_size in block_sizes:
        output_zarr = output_dir / f"blocks_{block_size}.zarr"

        if output_zarr.exists():
            shutil.rmtree(output_zarr)

        try:
            result = benchmark_zarr_dask_extraction(
                input_zarr_path=scenario["image_path"],
                output_zarr_path=str(output_zarr),
                model_name=model_name,
                block_size=block_size,
                n_workers=n_workers,
                image_size=image_size,
                block_method="block",
                backend="local"
            )

            suite.add_result(result)
            print(f"  Block size {block_size}: {result.total_time:.2f}s")

        except Exception as e:
            print(f"  ✗ Failed with block_size={block_size}: {e}")

        if output_zarr.exists():
            shutil.rmtree(output_zarr)

    # Cleanup
    shutil.rmtree(output_dir)

    return suite


if __name__ == "__main__":
    # Example: Test worker scaling on medium-sized image
    print("Running worker scaling analysis...")

    BLOCK_SIZE = 128

    scenarios = create_scaling_scenarios(base_dir="./data/benchmarking", block_size=BLOCK_SIZE)

    # Test with multiple worker counts
    suite = run_scaling_benchmark(
        scenarios=scenarios,
        model_name="dummy",
        block_size=BLOCK_SIZE,
        worker_counts=[32],
        output_dir = "data/benchmarking"
    )

    # Save results
    suite.save("data/benchmarking/image_size_scaling_benchmark_v2.json")
