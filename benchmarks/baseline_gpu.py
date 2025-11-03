import numpy as np
import torch
import zarr
import time
from typing import List, Tuple, Optional, Callable
from pathlib import Path
import geopandas as gpd

from utils import MemoryTracker, BenchmarkResults, estimate_n_chunks


class GPUBatchInference:
    """GPU batch inference baseline."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str = "cuda"
    ):
        """
        Initialize GPU batch inference.

        Args:
            model_name: Name of model to use (e.g., "uni", "dinov2")
            batch_size: Number of regions to process in each batch
            device: "cuda" or "cpu"
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        # Check GPU availability
        self.gpu_available = torch.cuda.is_available() and device == "cuda"
        if not self.gpu_available and device == "cuda":
            raise ValueError("Warning: CUDA not available")

        self.model = None
        self.n_features = 0

    def _load_model(self):
        """Load the model."""
        if self.model is not None:
            return

        from feature_blocks.models import available_models

        if self.model_name not in available_models:
            raise ValueError(f"Unknown model: {self.model_name}")

        print(f"Loading model {self.model_name} on {self.device}...")
        self.model = available_models[self.model_name]()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.n_features = self.model.n_features

        print(f"Model loaded. Features: {self.n_features}")

    def extract_from_zarr_blocks(
        self,
        zarr_path: str,
        block_size: int,
        track_memory: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Extract features from zarr using block method with GPU batching.

        Args:
            zarr_path: Path to input zarr array
            block_size: Size of blocks to extract
            track_memory: Whether to track memory usage

        Returns:
            Tuple of (features_array, timing_dict)
        """
        self._load_model()

        if track_memory:
            mem_tracker = MemoryTracker()

        # Load zarr
        z = zarr.open(zarr_path, mode="r")[0]
        C, Z, H, W = z.shape

        # Generate block indices
        n_h = int(np.ceil(H / block_size))
        n_w = int(np.ceil(W / block_size))
        n_blocks = n_h * n_w

        print(f"Processing {n_blocks} blocks of size {block_size}x{block_size}")

        # Pre-allocate output
        features = np.zeros((self.n_features, 1, n_h, n_w), dtype=np.float32)

        # Extract features in batches
        timing = {
            "io_time": 0.0,
            "inference_time": 0.0,
            "total_time": 0.0
        }

        total_start = time.time()

        batch_blocks = []
        batch_indices = []

        for h_idx in range(n_h):
            for w_idx in range(n_w):
                # Read block
                io_start = time.time()

                h_start = h_idx * block_size
                h_end = min((h_idx + 1) * block_size, H)
                w_start = w_idx * block_size
                w_end = min((w_idx + 1) * block_size, W)

                block = z[:, :, h_start:h_end, w_start:w_end]

                # Convert to tensor (C, H, W) - take first Z slice
                block_tensor = torch.from_numpy(block[:, 0, :, :]).float()

                # Pad if needed
                if block_tensor.shape[1] < block_size or block_tensor.shape[2] < block_size:
                    padded = torch.zeros(C, block_size, block_size)
                    padded[:, :block_tensor.shape[1], :block_tensor.shape[2]] = block_tensor
                    block_tensor = padded

                batch_blocks.append(block_tensor)
                batch_indices.append((h_idx, w_idx))

                timing["io_time"] += time.time() - io_start

                if track_memory:
                    mem_tracker.sample()

                # Process batch if full
                if len(batch_blocks) >= self.batch_size:
                    inf_start = time.time()

                    # Stack into batch
                    batch = torch.stack(batch_blocks).to(self.device)

                    # Inference
                    with torch.no_grad():
                        batch_features = self.model(batch)

                    # Store results
                    batch_features = batch_features
                    for i, (h_i, w_i) in enumerate(batch_indices):
                        features[:, 0, h_i, w_i] = batch_features[i]

                    timing["inference_time"] += time.time() - inf_start

                    # Clear batch
                    batch_blocks = []
                    batch_indices = []

                    if track_memory:
                        mem_tracker.sample()

        # Process remaining blocks
        if batch_blocks:
            inf_start = time.time()

            batch = torch.stack(batch_blocks).to(self.device)

            with torch.no_grad():
                batch_features = self.model(batch)

            batch_features = batch_features
            for i, (h_i, w_i) in enumerate(batch_indices):
                features[:, 0, h_i, w_i] = batch_features[i]

            timing["inference_time"] += time.time() - inf_start

        timing["total_time"] = time.time() - total_start

        print(f"Completed in {timing['total_time']:.2f}s")
        print(f"  I/O: {timing['io_time']:.2f}s")
        print(f"  Inference: {timing['inference_time']:.2f}s")

        stats = {
            "timing": timing,
            "n_blocks": n_blocks
        }

        if track_memory:
            stats["memory"] = mem_tracker.get_stats()

        return features, stats

    def benchmark(
        self,
        zarr_path: str,
        block_size: int,
        image_size: Tuple[int, int, int, int],
        method: str = "block"
    ) -> BenchmarkResults:
        """
        Run benchmark and return results.

        Args:
            zarr_path: Path to input zarr
            block_size: Block size
            image_size: Image dimensions
            method: "block" or "centroid"

        Returns:
            BenchmarkResults object
        """
        

        n_chunks = estimate_n_chunks(image_size, block_size, method)

        if method == "block":
            features, stats = self.extract_from_zarr_blocks(
                zarr_path, block_size, track_memory=True
            )
        else:
            raise NotImplementedError(f"Method {method} not yet implemented")

        timing = stats["timing"]
        memory = stats.get("memory", {})

        _, _, H, W = image_size
        pixels_total = H * W

        result = BenchmarkResults(
            name=f"gpu_batch_{self.model_name}",
            image_size=image_size,
            n_chunks=n_chunks,
            block_size=block_size,
            n_workers=1,  # Single GPU
            model_name=self.model_name,
            method="gpu_batch",
            total_time=timing["total_time"],
            inference_time=timing["inference_time"],
            io_time=timing["io_time"],
            setup_time=0.0,
            peak_memory_mb=memory.get("peak_mb", 0.0),
            avg_memory_mb=memory.get("avg_mb", 0.0),
            regions_per_second=n_chunks / timing["total_time"],
            pixels_per_second=pixels_total / timing["total_time"],
            n_features=self.n_features,
            gpu_available=self.gpu_available,
            device=self.device
        )

        return result


def benchmark_gpu_inference(
    zarr_path: str,
    model_name: str,
    block_size: int,
    image_size: Tuple[int, int, int, int],
    batch_size: int = 32,
    device: str = "cuda"
) -> BenchmarkResults:
    """
    Convenience function to benchmark GPU inference.

    Args:
        zarr_path: Path to input zarr
        model_name: Name of model
        block_size: Block size
        image_size: Image dimensions
        batch_size: Batch size for GPU
        device: "cuda" or "cpu"

    Returns:
        BenchmarkResults
    """
    inference = GPUBatchInference(
        model_name=model_name,
        batch_size=batch_size,
        device=device
    )

    return inference.benchmark(zarr_path, block_size, image_size)
