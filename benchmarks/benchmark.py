import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

from synthetic_data import create_scenarios
from utils import BenchmarkResults, MemoryTracker, estimate_n_chunks

from feature_blocks.features import get_model
from feature_blocks.features._extract import extract


def run_benchmark(
    model_name: str,
    block_size: Tuple[int] | int,
    worker_counts: Tuple[int] | int,
    image_size: Tuple[int, int, int, int],
    output_dir: str,
    block_method: str = "block",
    segmentations_path: Optional[str] = None,
    n_regions=None,
    backend: str = "local",
    track_memory: bool = True,
    batch_size: Tuple[int] | int = 1,
) -> BenchmarkResults:

    scenarios = create_scenarios(
        base_dir="./data/benchmarking",
        block_size=block_size,
        image_size=image_size,
    )

    if isinstance(worker_counts, int):
        worker_counts = [worker_counts]

    if isinstance(batch_size, int):
        batch_sizes = [batch_size]
    else:
        batch_sizes = batch_size

    benchmark_results = BenchmarkResults()

    for scenario in scenarios:
        scenario_name = scenario["name"]
        input_zarr = os.path.abspath(scenario["image_path"])
        image_size = scenario["image_size"]
        blk_size = scenario["block_size"]

        for n_workers in worker_counts:
            for batch_sz in batch_sizes:
                output_zarr = Path(output_dir) / f"{scenario_name}_w{n_workers}_b{batch_sz}.zarr"

                # Remove output if exists
                if output_zarr.exists():
                    shutil.rmtree(output_zarr)

                try:
                    result = benchmark_extract(
                        input_zarr_path=input_zarr,
                        output_zarr_path=str(output_zarr),
                        model_name=model_name,
                        block_size=blk_size,
                        n_workers=n_workers,
                        image_size=image_size,
                        block_method="block",
                        batch_size=batch_sz,
                        # backend="local"
                    )

                    benchmark_results.add_result(result)

                except Exception as e:
                    print(f"Failed with {n_workers} workers and batch_size {batch_sz}: {e}")

                # Cleanup output
                if output_zarr.exists():
                    shutil.rmtree(output_zarr)

    return benchmark_results


def benchmark_extract(
    input_zarr_path: str,
    output_zarr_path: str,
    model_name: str,
    block_size: int,
    n_workers: int,
    image_size: Tuple[int, int, int, int],
    block_method: str = "block",
    segmentations_path: Optional[str] = None,
    backend: str = "local",
    track_memory: bool = True,
    batch_size: int = 1,
) -> BenchmarkResults:
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
        batch_size: Number of regions to process per task (default: 1)

    Returns:
        BenchmarkResult object
    """
    if track_memory:
        mem_tracker = MemoryTracker()

    # Estimate number of chunks
    n_chunks = estimate_n_chunks(image_size, block_size, block_method)

    print("\nBenchmarking zarr+dask extraction:")
    print(f"  Model: {model_name}")
    print(f"  Image size: {image_size}")
    print(f"  Block size: {block_size}")
    print(f"  Block method: {block_method}")
    print(f"  Workers: {n_workers}")
    print(f"  Batch size: {batch_size}")
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
            batch_size=batch_size,
            # backend=backend
        )

        total_time = time.time() - start_time

        if track_memory:
            mem_tracker.sample()

        print(f"Completed in {total_time:.2f}s")

        # Get memory stats
        memory_stats = mem_tracker.get_stats() if track_memory else {}

        # Calculate throughput
        _, _, H, W = image_size
        pixels_total = H * W
        regions_per_second = n_chunks / total_time
        pixels_per_second = pixels_total / total_time

        n_features = get_model(model_name).n_features

        result = BenchmarkResults(
            name=f"zarr_dask_{model_name}",
            image_size=image_size,
            n_chunks=n_chunks,
            block_size=block_size,
            n_workers=n_workers,
            batch_size=batch_size,
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
            device="cpu",
        )

        return result

    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("Running scaling analysis...")

    results = run_benchmark(
        model_name="dummy",
        block_size=[224],
        worker_counts=[50, 100, 150, 200],
        image_size=[
            # (3, 1, 256, 256), 
            # (3, 1, 512, 512)
            (3, 1, 32000, 32000),
        ],
        block_method="block",
        segmentations_path=None,
        n_regions=None,
        track_memory=False,
        output_dir="data/benchmarking",
        batch_size=[1, 5, 10], 
    )

    # Save results
    results.save("data/benchmarking/image_size_scaling_benchmark_v5.json")
