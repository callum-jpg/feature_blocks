"""
CPU (zarr+dask) vs GPU (batched inference) comparison benchmark.

Answers the key question: At what scale does zarr+dask CPU inference become
more efficient than GPU batched inference?

This comparison is particularly relevant for:
- Small images: GPU may be faster due to batching and parallel computation
- Large images: CPU zarr+dask may win due to memory constraints and transfer overhead
"""

import os
import traceback
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import torch

from utils import BenchmarkResults
from synthetic_data import create_scenarios
from benchmark import benchmark_extract
from baseline_gpu import benchmark_gpu_inference


def compare_cpu_gpu_single_scenario(
    scenario: dict,
    model_name: str,
    block_size: int,
    output_dir: str,
    n_workers: int = 4,
    gpu_batch_size: int = 32,
    device: str = "cuda"
) -> Tuple[BenchmarkResults, dict]:
    """
    Compare CPU and GPU approaches for a single scenario.

    Args:
        scenario: Test scenario dictionary
        model_name: Model name (must be GPU-compatible, e.g., "uni", "dinov2")
        block_size: Block size
        n_workers: Number of CPU workers for zarr+dask
        gpu_batch_size: Batch size for GPU inference
        device: "cuda" or "cpu"

    Returns:
        Tuple of (BenchmarkResults, comparison_dict)
    """
    suite = BenchmarkResults(name=f"cpu_vs_gpu_{scenario['name']}")

    input_zarr = scenario["image_path"]
    image_size = scenario["image_size"]

    print(f"\n{'='*60}")
    print(f"CPU vs GPU Comparison: {scenario['name']}")
    print(f"Image size: {image_size}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # 1. GPU batch inference
    print("1. GPU Batched Inference")
    print("-" * 40)

    gpu_available = torch.cuda.is_available()
    if not gpu_available and device.casefold() == "cuda":
        raise ValueError("⚠ CUDA not available.")

    output_zarr_gpu = Path(output_dir) / "output_gpu.zarr"

    try:
        gpu_result = benchmark_gpu_inference(
            zarr_path=input_zarr,
            model_name=model_name,
            block_size=block_size,
            image_size=image_size,
            output_zarr_path=str(output_zarr_gpu),
            batch_size=gpu_batch_size,
            device=device
        )
        suite.add_result(gpu_result)
        print(f"✓ GPU completed in {gpu_result.total_time:.2f}s")
        print(f"  Memory: {gpu_result.peak_memory_mb:.1f}MB")
        print(f"  Throughput: {gpu_result.regions_per_second:.1f} regions/s")

    except Exception as e:
        traceback.print_exc()
        gpu_result = None

    # 2. CPU zarr+dask
    print(f"\n2. CPU Zarr+Dask ({n_workers} workers)")
    print("-" * 40)

    output_zarr_cpu = Path(output_dir) / "output_cpu.zarr"

    try:
        cpu_result = benchmark_extract(
            input_zarr_path=input_zarr,
            output_zarr_path=str(output_zarr_cpu),
            model_name=model_name,
            block_size=block_size,
            n_workers=n_workers,
            image_size=image_size,
            block_method="block",
            backend="local"
        )
        suite.add_result(cpu_result)
        print(f"✓ CPU completed in {cpu_result.total_time:.2f}s")
        print(f"  Memory: {cpu_result.peak_memory_mb:.1f}MB")
        print(f"  Throughput: {cpu_result.regions_per_second:.1f} regions/s")

    except Exception as e:
        print(f"✗ CPU failed: {e}")
        cpu_result = None

    # 3. Comparison
    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")

    comparison = {
        "scenario": scenario["name"],
        "image_size": image_size,
        "model": model_name,
        "gpu_available": gpu_available
    }

    if gpu_result and cpu_result:
        speedup = gpu_result.total_time / cpu_result.total_time
        winner = "GPU" if speedup < 1 else "CPU"

        comparison.update({
            "gpu_time": gpu_result.total_time,
            "cpu_time": cpu_result.total_time,
            "speedup": speedup,
            "winner": winner,
            "gpu_memory_mb": gpu_result.peak_memory_mb,
            "cpu_memory_mb": cpu_result.peak_memory_mb,
            "gpu_throughput": gpu_result.regions_per_second,
            "cpu_throughput": cpu_result.regions_per_second
        })

        print(f"GPU time: {gpu_result.total_time:.2f}s")
        print(f"CPU time: {cpu_result.total_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x ({winner} is faster)")
        print(f"GPU memory: {gpu_result.peak_memory_mb:.1f}MB")
        print(f"CPU memory: {cpu_result.peak_memory_mb:.1f}MB")

    else:
        print("Could not complete comparison (one method failed)")
        comparison["status"] = "incomplete"

    # Cleanup
    shutil.rmtree(output_dir)

    return suite, comparison


def find_crossover_point(
    model_name: str,
    block_size: Tuple[int] | int,
    image_size: Tuple[int, int, int, int],
    n_workers: int, 
    gpu_batch_size: int,
    device: str, 
    output_dir: str,
    temp_dir: Optional[str] = None,
) -> Tuple[BenchmarkResults, List[dict]]:
    """
    Find the crossover point where zarr+dask becomes faster than GPU batching.

    Tests multiple image sizes to identify the transition point.

    Args:
        model_name: Model name
        block_size: Block size
        n_workers: CPU workers
        gpu_batch_size: GPU batch size
        temp_dir: Temporary directory

    Returns:
        Tuple of (BenchmarkResults, comparisons_list)
    """
    print(f"\n{'#'*60}")
    print("FINDING CPU vs GPU CROSSOVER POINT")
    print(f"{'#'*60}\n")

    # Create scenarios of increasing size
    scenarios = create_scenarios(
        base_dir="./data/benchmarking",
        block_size=block_size,
        image_size=image_size,
    )

    suite = BenchmarkResults(name="cpu_gpu_crossover")
    comparisons = []

    for scenario in scenarios:
        scenario_name = scenario["name"]
        input_zarr = os.path.abspath(scenario["image_path"])
        image_size = scenario["image_size"]
        blk_size = scenario["block_size"]

        scenario_suite, comparison = compare_cpu_gpu_single_scenario(
            scenario=scenario,
            model_name=model_name,
            block_size=blk_size,
            n_workers=n_workers,
            output_dir=output_dir,
            gpu_batch_size=gpu_batch_size,
            device=device
        )

        # Add results to main suite
        for result in scenario_suite.results:
            suite.add_result(result)

        comparisons.append(comparison)

        # Print interim summary
        if "winner" in comparison:
            print(f"{scenario['name']}: {comparison['winner']} wins "
                  f"({comparison['speedup']:.2f}x)\n")

    # Final summary
    print(f"\n{'='*60}")
    print("CROSSOVER ANALYSIS")
    print(f"{'='*60}\n")

    cpu_wins = [c for c in comparisons if c.get("winner") == "CPU"]
    gpu_wins = [c for c in comparisons if c.get("winner") == "GPU"]

    print(f"GPU faster: {len(gpu_wins)} scenarios")
    print(f"CPU faster: {len(cpu_wins)} scenarios")

    # Identify crossover
    if cpu_wins and gpu_wins:
        # Find smallest image where CPU wins
        cpu_sizes = [(c["scenario"], c["image_size"]) for c in cpu_wins]
        cpu_sizes.sort(key=lambda x: x[1][2] * x[1][3])  # Sort by H*W

        print(f"Crossover point:")
        print(f"CPU becomes faster at: {cpu_sizes[0][0]}")
        print(f"  Image size: {cpu_sizes[0][1]}")

    return suite, comparisons


if __name__ == "__main__":
    suite, comparisons = find_crossover_point(
        model_name="tiny_vit",
        block_size=224,
        image_size=(
            (3, 1, 1024, 1024),
            # (3, 1, 8_000, 8_000),
            # (3, 1, 16_000, 16_000),
            # (3, 1, 32_000, 32_000),
        ),
        n_workers=1000,
        gpu_batch_size=128,
        device="cpu",
        output_dir="data",
    )

    # Save results
    suite.save("benchmark_results_cpu_vs_gpu.json")
