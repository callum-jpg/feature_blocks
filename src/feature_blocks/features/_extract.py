import logging
import math
import time
import typing
from datetime import timedelta

import dask.array
import numpy
import skimage
import zarr
from tqdm import tqdm

from feature_blocks.backend import run_dask_backend, run_sequential_backend
from feature_blocks.image import tissue_detection
from feature_blocks.models import available_models
from feature_blocks.slice import (filter_slices_by_mask,
                                  generate_centroid_slices,
                                  generate_nd_slices, normalize_slices)
from feature_blocks.io import save_ome_zarr
from feature_blocks.task import process_region

from zarr.codecs import BloscCodec

log = logging.getLogger(__name__)


def extract(
    input_zarr_path: str,
    feature_extraction_method: str,
    block_size: int,
    output_zarr_path: str,
    backend: str = "dask",
    device: str = "auto",
    batch_size: int = 1,
    n_workers: int | None = None,
    python_path: str = None,
    memory: str = "16GB",
    block_method: list["block", "centroid"] = "block",
    segmentations: typing.Optional["geopandas.GeoDataFrame"] = None,
    calculate_mask: bool = False,
    mask_downsample: int = 1,
    masked_block_value=numpy.nan,
    masking_kwargs: typing.Dict[str, int] = None,
    chunk_size: int | None = None,
    mask_batch_size: int = 256,
):

    # Default chunk_size to block_size if not specified
    if chunk_size is None:
        chunk_size = block_size

    # component=0 to read the high resolution image
    # We mostly load input_data here to get it's shape. If required,
    # it will be computed for mask creation.
    input_data = dask.array.from_zarr(input_zarr_path, component="0")

    if segmentations is not None:
        # Check if any segmentations are not valid
        if not segmentations.is_valid.all():
            log.warning(
                "Invalid geometries detected in segmentations, attempting to make valid."
            )
            # Use the buffer trick to make geometries valid
            segmentations["geometry"] = segmentations["geometry"].buffer(0)

            assert (
                segmentations.is_valid.all()
            ), "Unable to make segmentation geometries valid using buffer trick."

    assert (
        input_data.ndim == 4
    ), f"Expected zarr store to have 4 dimensions (C, Z, H, W). Got {input_data.ndim }."

    # Load the zarr store to be processed
    feature_extract_fn = get_model(feature_extraction_method)

    if block_method.casefold() == "block":
        # For each dimension, (channels, z, y, x) construct a list of
        # slice objects that will be used to index the zarr store.
        # XY slices are have shape (SIZE, SIZE) due to the slice object
        # step size being defined as SIZE
        regions = generate_nd_slices(input_data.shape, block_size, [2, 3])

        # For the zarr file to be saved, determine how many chunks
        # there will be in Y and X.
        num_chunks_y = math.ceil(input_data.shape[-2] / block_size)
        num_chunks_x = math.ceil(input_data.shape[-1] / block_size)

        # We need to know a priori what the output will be, which we can gather
        # from the feature extraction function
        output_chunks = feature_extract_fn.output_shape  # (C, Z, Y, X)

        assert (
            output_chunks[2] == output_chunks[3]
        ), f"Output chunksize defined in feature_extract_fn should be equal. Got {output_chunks[2], output_chunks[3]}"

        # Define the total size of the output zarr
        output_shape = (
            feature_extract_fn.n_features,
            1,
            num_chunks_y * output_chunks[2],
            num_chunks_x * output_chunks[3],
        )  # (C, Z, Y, X)

        placeholder_output_data = numpy.empty(output_shape)
        placeholder_output_data.fill(numpy.nan)

        save_ome_zarr(
            array=placeholder_output_data,
            output_path=output_zarr_path,
            chunks=output_chunks,
            axes="czyx",
            compression="zstd",
        )

        # No mask store needed for block method
        mask_store_path = None

    elif block_method.casefold() == "centroid":
        # Check if this is a CellProfiler model that needs mask data
        is_cellprofiler = (
            hasattr(feature_extract_fn, "__class__")
            and feature_extract_fn.__class__.__name__ == "CellProfiler"
        )

        if is_cellprofiler:
            # Check if CellProfiler is using bounding box mode
            use_bounding_box = getattr(feature_extract_fn, "use_bounding_box", False)

            if use_bounding_box:
                # Bounding box mode: use standard centroid slices (like ViT)
                log.info(
                    "CellProfiler in bounding box mode - using centroid regions without masks"
                )
                regions = generate_centroid_slices(
                    input_data.shape, size=block_size, segmentations=segmentations
                )
                mask_store_path = None
            else:
                # Mask mode: stream masks directly to zarr to avoid OOM
                # This approach keeps only ONE mask in memory at a time
                log.info(
                    "CellProfiler in mask mode - streaming masks to zarr (memory-efficient)"
                )

                from feature_blocks.utility._rasterize import rasterize_batch

                # First pass: gather slice info and find max dimensions
                # We avoid storing mask arrays here - just metadata
                regions = []
                slice_info = []  # (centroid_id, slc, x_min, y_min, x_max, y_max, geometry)
                max_h, max_w = 0, 0
                half_size = block_size // 2

                log.info("Pass 1: Computing slice bounds and max dimensions...")
                for idx, row in enumerate(segmentations.itertuples()):
                    # Get centroid ID
                    centroid_id = row.Index

                    # Calculate centroid and bounds
                    y = round(row.geometry.centroid.y)
                    x = round(row.geometry.centroid.x)

                    y_min = max(0, y - half_size)
                    y_max = min(input_data.shape[2], y + half_size)
                    x_min = max(0, x - half_size)
                    x_max = min(input_data.shape[3], x + half_size)

                    slc = (
                        slice(None),
                        slice(None),
                        slice(y_min, y_max),
                        slice(x_min, x_max),
                    )

                    region_h = y_max - y_min
                    region_w = x_max - x_min
                    max_h = max(max_h, region_h)
                    max_w = max(max_w, region_w)

                    # Store metadata only (no mask data yet)
                    slice_info.append(
                        (centroid_id, slc, x_min, y_min, x_max, y_max, row.geometry)
                    )
                    regions.append((centroid_id, slc, idx))

                # Create mask zarr store with known dimensions
                mask_store_path = f"{output_zarr_path}_masks.zarr"
                log.info(
                    f"Creating mask store for {len(regions)} masks (max size: {max_h}x{max_w})"
                )

                mask_store = zarr.create(
                    shape=(len(regions), max_h, max_w),
                    chunks=(1, max_h, max_w),
                    dtype=numpy.int32,
                    store=mask_store_path,
                    overwrite=True,
                    fill_value=0,
                )

                # Second pass: stream masks in batches to zarr
                # Batching significantly reduces I/O overhead
                n_batches = math.ceil(len(slice_info) / mask_batch_size)
                log.info(
                    f"Pass 2: Streaming masks to zarr in {n_batches} batches "
                    f"(batch_size={mask_batch_size})..."
                )

                for batch_idx in tqdm(range(n_batches), desc="Rasterizing mask batches"):
                    batch_start = batch_idx * mask_batch_size
                    batch_end = min(batch_start + mask_batch_size, len(slice_info))

                    # Prepare batch info with indices
                    batch_info = [
                        (idx, *slice_info[idx])
                        for idx in range(batch_start, batch_end)
                    ]

                    # Rasterize entire batch
                    batch_masks = rasterize_batch(batch_info, max_h, max_w)

                    # Write entire batch to zarr in one operation
                    mask_store[batch_start:batch_end, :, :] = batch_masks

                    # Explicitly delete to free memory
                    del batch_masks

                # Free the metadata list
                del slice_info
                log.info("Masks stored successfully")
        else:
            # Use the standard centroid slices without mask data
            regions = generate_centroid_slices(
                input_data.shape, size=block_size, segmentations=segmentations
            )
            mask_store_path = None

        output_shape = (len(regions), feature_extract_fn.n_features)

        output_chunks = (1, feature_extract_fn.n_features)

        placeholder_output_data = numpy.empty(output_shape)
        placeholder_output_data.fill(numpy.nan)

        save_ome_zarr(
            array=placeholder_output_data,
            output_path=output_zarr_path,
            chunks=output_chunks,
            axes="cy",  # c (observations), y (features)
            compression="zstd",
        )
    else:
        raise ValueError(f"block_method '{block_method}' not recognised.")

    if calculate_mask:
        log.info("Calculating mask...")
        if masking_kwargs:
            log.info(f"Using masking kawrgs: {', '.join(f'{k}={v}' for k, v in masking_kwargs.items())}")
        else:
            masking_kwargs = {}
        
        mask = tissue_detection(
            input_data[:, 0, ::mask_downsample, ::mask_downsample]
            .compute()
            .transpose(1, 2, 0),
            **masking_kwargs
        )
        # Mask has shape (Y, X). Resize this to the input image
        mask = skimage.transform.resize(
            mask, order=0, output_shape=input_data.shape[-2:]
        )
        mask = mask[numpy.newaxis, numpy.newaxis]  # Add C an Z dimensions back

        log.info(f"Before masking there were {len(regions)} possible regions")
        regions, _background_slices = filter_slices_by_mask(regions, mask)
        log.info(f"After masking there were {len(regions)} regions inside the mask")

        assert (
            len(regions) > 0
        ), "No foreground regions found after masking. Adjust or disable masking."

    # Validate backend selection
    if backend not in ["dask", "sequential"]:
        raise ValueError(f"backend must be 'dask' or 'sequential', got '{backend}'")

    # Determine model identifier to pass (string if possible for efficiency)
    if isinstance(feature_extraction_method, str):
        model_identifier = feature_extraction_method
    else:
        # If a callable was passed directly, use it (less efficient but supported)
        model_identifier = feature_extract_fn

    start_time = time.time()

    if backend == "dask":
        # Use Dask distributed backend
        log.info(f"Preparing to process {len(regions)} regions using Dask backend...")

        run_dask_backend(
            process_region,
            regions,
            n_workers=n_workers,
            python_path=python_path,
            memory=memory,
            batch_size=batch_size,
            input_zarr_path=input_zarr_path,
            output_zarr_path=output_zarr_path,
            mask_store_path=mask_store_path,
            function_kwargs={
                "input_zarr_path": input_zarr_path,
                "model_identifier": model_identifier,
                "n_features": feature_extract_fn.n_features,
                "output_zarr_path": output_zarr_path,
                "block_size": block_size,
                "chunk_size": chunk_size,
                "output_chunks": output_chunks,
                "mask_store_path": mask_store_path,
            },
        )

    elif backend == "sequential":
        # Use sequential backend (Dask-free, with GPU/CPU batching)
        log.info(f"Preparing to process {len(regions)} regions using sequential backend on {device}...")

        run_sequential_backend(
            regions=regions,
            model_identifier=model_identifier,
            input_zarr_path=input_zarr_path,
            output_zarr_path=output_zarr_path,
            block_method=block_method,
            block_size=block_size,
            chunk_size=chunk_size,
            output_chunks=output_chunks,
            device=device,
            batch_size=batch_size,
            mask_store_path=mask_store_path,
        )

    elapsed = time.time() - start_time
    log.info(f"Analysis time: {str(timedelta(seconds=round(elapsed)))}")


def get_model(model: typing.Callable | str) -> "torch.nn.Module":
    if isinstance(model, str):
        assert (
            model in available_models
        ), f"'{model}' is not a valid model name. Valid names are: {', '.join(list(available_models.keys()))}"

        return available_models[model]()

    return model
