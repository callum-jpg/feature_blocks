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
                                  generate_centroid_slices_with_single_masks,
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
):

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
                # Mask mode: use mask data for each segmentation
                log.info(
                    "CellProfiler in mask mode - generating masks for each segmentation"
                )
                regions_with_masks = generate_centroid_slices_with_single_masks(
                    input_data.shape, size=block_size, segmentations=segmentations
                )

                # Store masks in a temporary zarr to avoid embedding them in the Dask graph
                mask_store_path = f"{output_zarr_path}_masks.zarr"
                log.info(
                    f"Storing {len(regions_with_masks)} masks to temporary zarr: {mask_store_path}"
                )

                # Extract mask data and create regions list without masks
                regions = []
                masks = []
                for idx, (centroid_id, slc, mask_data) in enumerate(regions_with_masks):
                    regions.append(
                        (centroid_id, slc, idx)
                    )  # Store index instead of mask_data
                    masks.append(mask_data)

                # Find max mask shape to create uniform zarr array
                max_h = max(mask.shape[0] for mask in masks)
                max_w = max(mask.shape[1] for mask in masks)

                # Create mask zarr store
                mask_store = zarr.create(
                    shape=(len(masks), max_h, max_w),
                    chunks=(1, max_h, max_w),
                    dtype=numpy.int32,
                    store=mask_store_path,
                    overwrite=True,
                    fill_value=0,
                    # compressor=BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),
                )

                # Write all masks to zarr in parallel using numpy array operations
                log.info("Writing masks to zarr in parallel...")

                # Stack all masks into a single array with padding
                padded_masks = numpy.zeros((len(masks), max_h, max_w), dtype=numpy.int32)
                for idx, mask_data in enumerate(masks):
                    h, w = mask_data.shape
                    padded_masks[idx, :h, :w] = mask_data

                # Single write operation - much faster than individual writes
                mask_store[:] = padded_masks

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

        log.info(f"Before masking there was {len(regions)} possible regions")
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
