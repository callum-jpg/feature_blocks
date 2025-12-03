import logging
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy
import torch
import zarr
from tqdm import tqdm

log = logging.getLogger(__name__)


def detect_device(device: str = "auto") -> str:
    """
    Detect and validate the compute device.

    Args:
        device: "auto", "cuda", or "cpu"

    Returns:
        Validated device string ("cuda" or "cpu")
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            log.info("Auto-detected CUDA GPU")
        else:
            device = "cpu"
            log.info("No CUDA GPU found, using CPU")
    elif device == "cuda":
        if not torch.cuda.is_available():
            log.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"

    return device


class SequentialProcessor:
    """
    Sequential (non-distributed) processor for feature extraction.

    Processes regions sequentially with batch inference for GPU efficiency.
    Supports both block and centroid methods.
    """

    def __init__(
        self,
        model_identifier: str | Callable,
        device: str = "auto",
        batch_size: int = 32,
    ):
        """
        Initialize sequential processor.

        Args:
            model_identifier: Model name string or callable
            device: "auto", "cuda", or "cpu"
            batch_size: Number of regions to batch for inference
        """
        self.model_identifier = model_identifier
        self.device = detect_device(device)
        self.batch_size = batch_size

        self.model = None
        self.n_features = None

    def _load_model(self):
        """Load the model onto the specified device."""
        if self.model is not None:
            return

        from feature_blocks.models import available_models

        # Get model instance
        if isinstance(self.model_identifier, str):
            if self.model_identifier not in available_models:
                raise ValueError(
                    f"Unknown model: {self.model_identifier}. "
                    f"Available models: {', '.join(available_models.keys())}"
                )
            log.info(f"Loading model '{self.model_identifier}' on {self.device}...")
            self.model = available_models[self.model_identifier]()
        else:
            log.info(f"Loading custom model on {self.device}...")
            self.model = self.model_identifier

        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        self.n_features = self.model.n_features

        log.info(f"Model loaded successfully. Features: {self.n_features}")

    def _read_block(self, input_data, region, block_size: int) -> torch.Tensor:
        """
        Read a block from input zarr and convert to tensor.

        Args:
            input_data: Zarr array
            region: Region tuple (slices)
            block_size: Expected block size

        Returns:
            Tensor of shape (C, H, W)
        """
        # Read block from zarr
        block = input_data[region]  # Shape: (C, Z, H, W)

        # Take first Z slice and convert to numpy
        block_np = block[:, 0, :, :]  # Shape: (C, H, W)

        # Convert to tensor
        block_tensor = torch.from_numpy(block_np).float()

        # Pad if needed (for edge blocks)
        C, H, W = block_tensor.shape
        if H < block_size or W < block_size:
            padded = torch.zeros(C, block_size, block_size)
            padded[:, :H, :W] = block_tensor
            block_tensor = padded

        return block_tensor

    def _read_centroid_region(self, input_data, region, block_size: int) -> torch.Tensor:
        """
        Read a centroid region from input zarr and convert to tensor.

        Args:
            input_data: Zarr array
            region: Region tuple (centroid_id, slices) or (centroid_id, slices, mask_index)
            block_size: Expected block size

        Returns:
            Tensor of shape (C, H, W)
        """
        if len(region) == 2:
            # Standard centroid (no mask)
            _, slices = region
            block = input_data[slices]  # Shape: (C, Z, H, W)
            block_np = block[:, 0, :, :]  # Shape: (C, H, W)
        elif len(region) == 3:
            # CellProfiler with mask (not yet implemented)
            raise NotImplementedError("CellProfiler mask mode not yet supported in sequential backend")
        else:
            raise ValueError(f"Invalid region tuple length: {len(region)}")

        # Convert to tensor
        block_tensor = torch.from_numpy(block_np).float()

        # Pad if needed
        C, H, W = block_tensor.shape
        if H < block_size or W < block_size:
            padded = torch.zeros(C, block_size, block_size)
            padded[:, :H, :W] = block_tensor
            block_tensor = padded

        return block_tensor

    def _infer_batch(self, batch_tensors: list[torch.Tensor]) -> numpy.ndarray:
        """
        Run inference on a batch of tensors.

        Args:
            batch_tensors: List of tensors, each of shape (C, H, W)

        Returns:
            Numpy array of shape (batch_size, n_features)
        """
        # Stack into batch: (B, C, H, W)
        batch = torch.stack(batch_tensors).to(self.device)

        # Run inference
        with torch.no_grad():
            features = self.model(batch)

        if isinstance(features, numpy.ndarray):
            # CPU-based methods do not return tensors
            return features
        else:
            # Convert to numpy and return
            return features.cpu().numpy()

    def process_blocks(
        self,
        input_zarr_path: str,
        output_zarr_path: str,
        regions: list,
        block_size: int,
        chunk_size: int,
        output_chunks: tuple,
    ):
        """
        Process regions using block method with batch inference.

        Args:
            input_zarr_path: Path to input zarr
            output_zarr_path: Path to output zarr
            regions: List of region tuples (slices)
            block_size: Size of blocks for processing
            chunk_size: Size of input zarr chunks
            output_chunks: Output chunk shape
        """
        self._load_model()

        # Open zarr stores
        input_data = zarr.open(input_zarr_path, mode="r")["0"]
        output_data = zarr.open(output_zarr_path, mode="r+")["0"]

        # Process in batches
        batch_tensors = []
        batch_regions = []

        log.info(f"Processing {len(regions)} blocks with batch size {self.batch_size}...")

        for region in tqdm(regions, desc="Processing blocks"):
            # Read block
            block_tensor = self._read_block(input_data, region, block_size)
            batch_tensors.append(block_tensor)
            batch_regions.append(region)

            # Process batch when full
            if len(batch_tensors) >= self.batch_size:
                features = self._infer_batch(batch_tensors)

                # Write results
                self._write_block_features(
                    output_data, features, batch_regions, chunk_size, output_chunks
                )

                # Clear batch
                batch_tensors = []
                batch_regions = []

        # Process remaining blocks
        if batch_tensors:
            features = self._infer_batch(batch_tensors)
            self._write_block_features(
                output_data, features, batch_regions, chunk_size, output_chunks
            )

        log.info("Block processing complete")

    def _write_block_features(
        self,
        output_data,
        features: numpy.ndarray,
        batch_regions: list,
        chunk_size: int,
        output_chunks: tuple,
    ):
        """Write block features to output zarr."""
        from feature_blocks.slice import normalize_slices

        for i, region in enumerate(batch_regions):
            # Build output region
            output_region = [
                slice(0, self.n_features, None),
                slice(0, 1, None),
            ]
            # Calculate output position based on input chunk_size and output downsampling
            output_step = chunk_size // output_chunks[2]
            output_region.extend(normalize_slices(region[-2:], output_step))

            # Write features
            output_data[tuple(output_region)] = features[i].reshape(self.n_features, 1, 1, 1)

    def process_centroids(
        self,
        input_zarr_path: str,
        output_zarr_path: str,
        regions: list,
        block_size: int,
        mask_store_path: Optional[str] = None,
    ):
        """
        Process regions using centroid method with batch inference.

        Args:
            input_zarr_path: Path to input zarr
            output_zarr_path: Path to output zarr
            regions: List of region tuples (centroid_id, slices)
            block_size: Size of regions
            mask_store_path: Optional path to mask store (for CellProfiler)
        """
        self._load_model()

        # Open zarr stores
        input_data = zarr.open(input_zarr_path, mode="r")[0]
        output_data = zarr.open(output_zarr_path, mode="r+")[0]

        # Process in batches
        batch_tensors = []
        batch_centroid_ids = []

        log.info(f"Processing {len(regions)} centroids with batch size {self.batch_size}...")

        for region in tqdm(regions, desc="Processing centroids"):
            # Extract centroid ID
            centroid_id = region[0]

            # Read region
            region_tensor = self._read_centroid_region(input_data, region, block_size)
            batch_tensors.append(region_tensor)
            batch_centroid_ids.append(centroid_id)

            # Process batch when full
            if len(batch_tensors) >= self.batch_size:
                features = self._infer_batch(batch_tensors)

                # Write results
                for i, cid in enumerate(batch_centroid_ids):
                    output_data[cid, :] = features[i]

                # Clear batch
                batch_tensors = []
                batch_centroid_ids = []

        # Process remaining regions
        if batch_tensors:
            features = self._infer_batch(batch_tensors)
            for i, cid in enumerate(batch_centroid_ids):
                output_data[cid, :] = features[i]

        log.info("Centroid processing complete")


def run_sequential_backend(
    regions: list,
    model_identifier: str | Callable,
    input_zarr_path: str,
    output_zarr_path: str,
    block_method: str,
    block_size: int,
    output_chunks: tuple,
    device: str = "auto",
    batch_size: int = 32,
    mask_store_path: Optional[str] = None,
    chunk_size: int | None = None,
):
    """
    Run feature extraction using sequential backend.

    This is a Dask-free approach that processes regions sequentially
    with batch inference for GPU efficiency.

    Args:
        regions: List of region tuples
        model_identifier: Model name string or callable
        input_zarr_path: Path to input zarr
        output_zarr_path: Path to output zarr
        block_method: "block" or "centroid"
        block_size: Size of blocks/regions for processing
        output_chunks: Output chunk shape
        device: "auto", "cuda", or "cpu"
        batch_size: Number of regions to batch for inference
        mask_store_path: Optional path to mask store
        chunk_size: Size of input zarr chunks (defaults to block_size if None)
    """
    # Default chunk_size to block_size for backward compatibility
    if chunk_size is None:
        chunk_size = block_size
    start_time = time.time()

    # Create processor
    processor = SequentialProcessor(
        model_identifier=model_identifier,
        device=device,
        batch_size=batch_size,
    )

    # Process based on method
    if block_method == "block":
        processor.process_blocks(
            input_zarr_path=input_zarr_path,
            output_zarr_path=output_zarr_path,
            regions=regions,
            block_size=block_size,
            chunk_size=chunk_size,
            output_chunks=output_chunks,
        )
    elif block_method == "centroid":
        processor.process_centroids(
            input_zarr_path=input_zarr_path,
            output_zarr_path=output_zarr_path,
            regions=regions,
            block_size=block_size,
            mask_store_path=mask_store_path,
        )
    else:
        raise ValueError(f"Unknown block_method: {block_method}")

    elapsed = time.time() - start_time
    log.info(f"Sequential backend completed in {elapsed:.2f}s")
