import logging
import math
import time
import typing
from datetime import timedelta

import dask.array
import numpy
import skimage
import zarr

from feature_blocks.backend import run_dask_backend
from feature_blocks.image import tissue_detection
from feature_blocks.models import available_models
from feature_blocks.slice import (filter_slices_by_mask,
                                  generate_centroid_slices,
                                  generate_centroid_slices_with_single_masks,
                                  generate_nd_slices, normalize_slices)
from feature_blocks.task import infer, read, read_with_mask, write

log = logging.getLogger(__name__)


def extract(
    input_zarr_path: str,
    feature_extraction_method: str,
    block_size: int,
    output_zarr_path: str,
    n_workers: int | None = None,
    python_path: str = "python",
    memory: str = "16GB",
    block_method: list["block", "centroid"] = "block",
    segmentations: typing.Optional["geopandas.GeoDataFrame"] = None,
    calculate_mask: bool = False,
    image_downsample: int = 1,
    masked_block_value=numpy.nan,
):

    input_data = dask.array.from_zarr(input_zarr_path)

    if segmentations is not None:
        # Check if any segmentations are not valid
        if not segmentations.is_valid.all():
            log.warning(f"Invalid geometries detected in segmentations, attempting to make valid.")
            # Use the buffer trick to make geometries valid
            segmentations["geometry"] = segmentations["geometry"].buffer(0)

            assert segmentations.is_valid.all(), f"Unable to make segmentation geometries valid using buffer trick."

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

        # Prepare output zarr file with synchronizer for safe parallel writes
        # and compression to reduce I/O bottleneck
        synchronizer = zarr.ProcessSynchronizer(f"{output_zarr_path}.sync")
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)

        output_data = zarr.create(
            shape=output_shape,
            chunks=output_chunks,
            dtype=numpy.float32,
            store=output_zarr_path,
            overwrite=True,
            fill_value=numpy.nan,  # Value to use for empty chunks
            synchronizer=synchronizer,
            compressor=compressor,
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
            # Use the specialized function that includes mask data for each segmentation
            regions_with_masks = generate_centroid_slices_with_single_masks(
                input_data.shape, size=block_size, segmentations=segmentations
            )

            # Store masks in a temporary zarr to avoid embedding them in the Dask graph
            # This prevents massive graph sizes (10+ GB) for whole slide images
            mask_store_path = f"{output_zarr_path}_masks.zarr"
            log.info(f"Storing {len(regions_with_masks)} masks to temporary zarr: {mask_store_path}")

            # Extract mask data and create regions list without masks
            regions = []
            mask_shapes = []
            for idx, (centroid_id, slc, mask_data) in enumerate(regions_with_masks):
                regions.append((centroid_id, slc, idx))  # Store index instead of mask_data
                mask_shapes.append(mask_data.shape)

            # Find max mask shape to create uniform zarr array
            max_h = max(shape[0] for shape in mask_shapes)
            max_w = max(shape[1] for shape in mask_shapes)

            # Create mask zarr store
            mask_store = zarr.create(
                shape=(len(regions_with_masks), max_h, max_w),
                chunks=(1, max_h, max_w),
                dtype=numpy.int32,
                store=mask_store_path,
                overwrite=True,
                fill_value=0,
                compressor=zarr.Blosc(cname="zstd", clevel=1),
            )

            # Write all masks to zarr
            for idx, (_, _, mask_data) in enumerate(regions_with_masks):
                h, w = mask_data.shape
                mask_store[idx, :h, :w] = mask_data

            log.info(f"Masks stored successfully")
        else:
            # Use the standard centroid slices without mask data
            regions = generate_centroid_slices(
                input_data.shape, size=block_size, segmentations=segmentations
            )
            mask_store_path = None

        output_shape = (len(regions), feature_extract_fn.n_features)

        output_chunks = (1, feature_extract_fn.n_features)

        # Prepare output zarr file with synchronizer for safe parallel writes
        # and compression to reduce I/O bottleneck
        synchronizer = zarr.ProcessSynchronizer(f"{output_zarr_path}.sync")
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)

        output_data = zarr.create(
            shape=output_shape,
            chunks=output_chunks,
            dtype=numpy.float32,
            store=output_zarr_path,
            overwrite=True,
            fill_value=numpy.nan,  # Value to use for empty chunks
            synchronizer=synchronizer,
            compressor=compressor,
        )
    else:
        raise ValueError(f"block_method '{block_method}' not recognised.")

    if calculate_mask:
        log.info("Calculating mask...")
        mask = tissue_detection(
            input_data[:, 0, ::image_downsample, ::image_downsample]
            .compute()
            .transpose(1, 2, 0)
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

    # Create tasks. This is a list of delayed jobs to be run on the dask
    # backend
    log.info(f"Creating delayed functions for {len(regions)} regions...")

    # Determine model identifier to pass (string if possible for efficiency)
    if isinstance(feature_extraction_method, str):
        model_identifier = feature_extraction_method
    else:
        # If a callable was passed directly, use it (less efficient but supported)
        model_identifier = feature_extract_fn

    # Create delayed tasks directly without multiprocessing
    # This allows Dask to distribute the task creation itself
    tasks = []
    for reg in regions:
        task = process_region(
            reg,
            input_zarr_path,
            model_identifier,
            feature_extract_fn.n_features,
            output_zarr_path,
            block_size,
            output_chunks,
            mask_store_path,
        )
        tasks.append(task)

    start_time = time.time()

    # Only pass model_identifier for warmup if it's a string (serializable)
    warmup_model = model_identifier if isinstance(model_identifier, str) else None

    run_dask_backend(
        tasks,
        n_workers=n_workers,
        python_path=python_path,
        memory=memory,
        model_identifier=warmup_model,
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


def process_region(
    reg,
    input_zarr_path,
    model_identifier,
    n_features,
    output_zarr_path,
    block_size,
    output_chunks,
    mask_store_path=None,
):
    """Process a single region

    Args:
        reg: Region tuple (slices or with chunk_id/mask_index)
        input_zarr_path: Path to input zarr
        model_identifier: String model name (serialization-friendly) or callable
        n_features: Number of features the model produces
        output_zarr_path: Path to output zarr
        block_size: Size of blocks
        output_chunks: Output chunk shape
        mask_store_path: Path to zarr store containing mask data (for CellProfiler)
    """

    # Block method is being used
    if len(reg) == 4:
        delayed_chunk = dask.delayed(read)(input_zarr_path, reg)
        delayed_result = dask.delayed(infer, pure=True)(
            delayed_chunk, model_identifier
        )
        # Build where the new region will be in the output zarr
        # (n_features, 1, H, W)
        new_region = [
            slice(0, n_features, None),
            slice(0, 1, None),
        ]
        # feature_blocks only supports square blocks, so we can infer that
        # H == W, hence why we only select output_chunks[2]
        chunk_size = block_size // output_chunks[2]
        new_region.extend(normalize_slices(reg[-2:], chunk_size))

        delayed_write = dask.delayed(write, pure=False)(
            output_zarr_path, delayed_result, new_region
        )
    # ROI method is being used (standard centroid)
    elif len(reg) == 2:
        chunk_id, chunk_slices = reg[0], reg[1:][0]

        delayed_chunk = dask.delayed(read)(input_zarr_path, chunk_slices)

        delayed_result = dask.delayed(infer, pure=True)(
            delayed_chunk, model_identifier
        )

        new_region = [
            chunk_id,
            slice(0, n_features),
        ]

        delayed_write = dask.delayed(write, pure=False)(
            output_zarr_path, delayed_result, new_region
        )

    # CellProfiler method with mask index (centroid + mask)
    elif len(reg) == 3:
        chunk_id, chunk_slices, mask_index = reg

        # Load mask from zarr store instead of embedding in graph
        # This is much more efficient for large numbers of segmentations
        delayed_chunk = dask.delayed(read_with_mask)(
            input_zarr_path, (chunk_id, chunk_slices, mask_index), mask_store_path
        )

        delayed_result = dask.delayed(infer, pure=True)(
            delayed_chunk, model_identifier
        )

        new_region = [
            chunk_id,
            slice(0, n_features),
        ]

        delayed_write = dask.delayed(write, pure=False)(
            output_zarr_path, delayed_result, new_region
        )

    else:
        raise ValueError(
            f"Region is of length {len(reg)} rather than the expected 2, 3, or 4. Region: {reg}"
        )

    return delayed_write
