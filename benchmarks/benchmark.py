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
    shard_size: Tuple[int] | int = None,
    chunk_size: Tuple[int] | int = None,
    n_regions=None,
    backend: Tuple[str] | str = "dask",
    python_path: str = None,
    track_memory: bool = True,
    batch_size: Tuple[int] | int = 1,
    csv_path: Optional[str] = None,
) -> BenchmarkResults:
    """
    Run benchmarks across multiple scenarios and settings.

    Args:
        model_name: Feature extraction model to use
        block_size: Block size(s) to test
        worker_counts: Number(s) of workers to test
        image_size: Image size(s) to test
        output_dir: Directory for temporary output files
        block_method: "block" or "centroid"
        segmentations_path: Path to segmentations (for centroid method)
        shard_size: Shard size(s) for zarr
        chunk_size: Chunk size(s) for zarr (defaults to block_size if None)
        n_regions: Number of regions (for synthetic data)
        backend: Backend(s) to test - "dask", "sequential", or list of both
        track_memory: Whether to track memory usage
        batch_size: Batch size(s) to test
        csv_path: Path to save incremental CSV results

    Returns:
        BenchmarkResults with all results
    """
    scenarios = create_scenarios(
        base_dir="./data/benchmarking",
        block_size=block_size,
        shard_size=shard_size,
        chunk_size=chunk_size,
        image_size=image_size,
    )

    if isinstance(worker_counts, int):
        worker_counts = [worker_counts]

    if isinstance(batch_size, int):
        batch_size = [batch_size]

    if isinstance(backend, str):
        backend = [backend]

    benchmark_results = BenchmarkResults()

    for scenario in scenarios:
        scenario_name = scenario["name"]
        input_zarr = os.path.abspath(scenario["image_path"])
        image_size = scenario["image_size"]
        blk_size = scenario["block_size"]
        chk_size = scenario["chunk_size"]
        shrd_size = scenario["shard_size"]

        for bkend in backend:
            for n_workers in worker_counts:
                for batch_sz in batch_size:
                    output_zarr = Path(output_dir) / "benchmark_output.zarr"

                    # Remove output if exists
                    if output_zarr.exists():
                        shutil.rmtree(output_zarr)

                    try:
                        result = benchmark_extract(
                            input_zarr_path=input_zarr,
                            output_zarr_path=str(output_zarr),
                            model_name=model_name,
                            block_size=blk_size,
                            chunk_size=chk_size,
                            shard_size=shrd_size,
                            n_workers=n_workers,
                            image_size=image_size,
                            block_method="block",
                            batch_size=batch_sz,
                            backend=bkend,
                        )

                        benchmark_results.add_result(result)

                        # Incrementally save to CSV
                        if csv_path:
                            result.save_csv(csv_path, append=True)

                    except Exception as e:
                        print(f"Failed with backend={bkend}, {n_workers} workers, batch_size={batch_sz}: {e}")

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
    shard_size: int = None,
    chunk_size: int = None,
    block_method: str = "block",
    segmentations_path: Optional[str] = None,
    backend: str = "dask",
    track_memory: bool = True,
    batch_size: int = 1,
    python_path: str = None,
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
        shard_size: Size of shards for zarr
        chunk_size: Size of zarr chunks (defaults to block_size if None)
        block_method: "block" or "centroid"
        segmentations_path: Path to segmentations GeoJSON (for centroid method)
        backend: Dask backend ("dask" or "sequential")
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
    print(f"  Chunk size: {chunk_size if chunk_size else block_size}")
    print(f"  Shard size: {shard_size}")
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
            chunk_size=chunk_size,
            output_zarr_path=output_zarr_path,
            n_workers=n_workers,
            block_method=block_method,
            segmentations=segmentations_path,
            batch_size=batch_size,
            backend=backend,
            python_path=python_path,
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
            name=f"{backend}_{model_name}",
            image_size=image_size,
            n_chunks=n_chunks,
            block_size=block_size,
            chunk_size=chunk_size if chunk_size else block_size,
            shard_size=shard_size,
            n_workers=n_workers,
            batch_size=batch_size,
            backend=backend,
            model_name=model_name,
            method=backend,
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

    OUTPUT_DIR = "data/benchmarking"

    # base_img = (3, 1, 256, 256)
    # LEVELS = 8

    base_img = (3, 1, 32768, 32768)
    LEVELS = 0

    image_sizes = [
        (base_img[0], base_img[1], base_img[2] * 2**i, base_img[3] * 2**i)
        for i in range(LEVELS + 1)
    ]

    # results = run_benchmark(
    #     model_name="dummy",
    #     block_size=[200],
    #     shard_size=[200, 1000, 2000],
    #     # worker_counts=[1, 2, 4],
    #     worker_counts=[50, 100, 200, 400, 600],
    #     image_size=image_sizes,
    #     batch_size=[1, 4, 8, 16], 
    #     block_method="block",
    #     segmentations_path=None,
    #     n_regions=None,
    #     track_memory=False,
    #     output_dir=OUTPUT_DIR,
    #     backend="sequential",
    #     csv_path="benchmarking_results/image_benchmarking_SEQUENTIAL.csv"
    # )

    # # Save results
    # results.save(f"benchmarking_results/image_benchmarking_SEQUENTIAL.json")

    results = run_benchmark(
        model_name="dummy",
        block_size=[200],
        chunk_size=[200, 400],
        # shard_size=[200, 1000, 2000],
        shard_size=[200],
        worker_counts=[200],
        # worker_counts=[50, 100, 200, 400, 600],
        image_size=image_sizes,
        # batch_size=[1, 4, 8, 16],
        batch_size=[4], 
        block_method="block",
        segmentations_path=None,
        n_regions=None,
        track_memory=False,
        output_dir=OUTPUT_DIR,
        backend="dask",
        # python_path="/nfs/research/uhlmann/callum/feature_blocks/.venv/bin/python",
        csv_path="benchmarking_results/image_benchmarking_v3.csv"
    )

    # Save results
    results.save(f"benchmarking_results/image_benchmarking_v3.json")

