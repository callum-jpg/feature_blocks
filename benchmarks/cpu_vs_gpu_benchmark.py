"""
CPU (zarr+dask) vs GPU (batched inference) comparison benchmark.

Compares regions per second throughput between CPU and GPU approaches
across different scenarios.
"""

import os
import traceback
import shutil
from pathlib import Path
from typing import Tuple, Optional

from utils import BenchmarkResults
from synthetic_data import create_scenarios
from benchmark import benchmark_extract
from baseline_gpu import benchmark_gpu_inference


def run_cpu_gpu_benchmark(
    model_name: str,
    block_size: Tuple[int] | int,
    image_size: Tuple[int, int, int, int],
    output_dir: str,
    n_workers: int = 4,
    gpu_batch_size: int = 32,
    cpu_batch_size: int = 1,
    device: str = "cuda",
    csv_path: Optional[str] = None,
    shard_size: Tuple[int] | int = None,
) -> BenchmarkResults:
    """
    Run CPU vs GPU comparison benchmark.

    Args:
        model_name: Model name (must be GPU-compatible)
        block_size: Block size(s) to test
        image_size: Image size(s) to test
        output_dir: Output directory for temporary files
        n_workers: Number of CPU workers for zarr+dask
        gpu_batch_size: Batch size for GPU inference
        cpu_batch_size: Batch size for CPU inference
        device: "cuda" or "cpu" for GPU baseline
        csv_path: Path to save incremental CSV results
        shard_size: Shard size(s) for zarr

    Returns:
        BenchmarkResults with all CPU and GPU results
    """
    scenarios = create_scenarios(
        base_dir="./data/benchmarking",
        block_size=block_size,
        shard_size=shard_size,
        image_size=image_size,
    )

    benchmark_results = BenchmarkResults(name="cpu_vs_gpu_benchmark")

    for scenario in scenarios:
        scenario_name = scenario["name"]
        input_zarr = os.path.abspath(scenario["image_path"])
        image_size = scenario["image_size"]
        blk_size = scenario["block_size"]
        shrd_size = scenario["shard_size"]

        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"Image size: {image_size}")
        print(f"{'='*60}\n")

        output_zarr = Path(output_dir) / "benchmark_output.zarr"

        # GPU benchmark
        print("GPU Batched Inference")
        print("-" * 40)

        if output_zarr.exists():
            shutil.rmtree(output_zarr)

        try:
            gpu_result = benchmark_gpu_inference(
                zarr_path=input_zarr,
                model_name=model_name,
                block_size=blk_size,
                image_size=image_size,
                output_zarr_path=str(output_zarr),
                batch_size=gpu_batch_size,
                device=device,
            )
            # Mark as GPU result
            gpu_result.device = device
            gpu_result.method = "gpu_batch"

            benchmark_results.add_result(gpu_result)
            print(f"GPU: {gpu_result.regions_per_second:.1f} regions/s")

            if csv_path:
                gpu_result.save_csv(csv_path, append=True)

        except Exception as e:
            print(f"GPU failed: {e}")
            traceback.print_exc()

        # Cleanup
        if output_zarr.exists():
            shutil.rmtree(output_zarr)

        # CPU benchmark
        print(f"\nCPU Zarr+Dask ({n_workers} workers)")
        print("-" * 40)

        try:
            cpu_result = benchmark_extract(
                input_zarr_path=input_zarr,
                output_zarr_path=str(output_zarr),
                model_name=model_name,
                block_size=blk_size,
                shard_size=shrd_size,
                n_workers=n_workers,
                image_size=image_size,
                block_method="block",
                batch_size=cpu_batch_size,
            )
            # Mark as CPU result
            cpu_result.device = "cpu"
            cpu_result.method = "zarr_dask"

            benchmark_results.add_result(cpu_result)
            print(f"CPU: {cpu_result.regions_per_second:.1f} regions/s")

            if csv_path:
                cpu_result.save_csv(csv_path, append=True)

        except Exception as e:
            print(f"CPU failed: {e}")
            traceback.print_exc()

        # Cleanup
        if output_zarr.exists():
            shutil.rmtree(output_zarr)

    return benchmark_results


if __name__ == "__main__":
    OUTPUT_DIR = "benchmarking_results"

    results = run_cpu_gpu_benchmark(
        model_name="tiny_vit",
        block_size=200,
        shard_size=200,
        image_size=[
            # (3, 1, 1024, 1024)
            (3, 1, 32768, 32768)
            ],
        n_workers=200,
        gpu_batch_size=32,
        cpu_batch_size=4,
        device="cuda",
        output_dir=OUTPUT_DIR,
        csv_path=f"{OUTPUT_DIR}/cpu_vs_gpu_results.csv",
    )

    # Also save JSON
    results.save(f"{OUTPUT_DIR}/cpu_vs_gpu_results.json")
