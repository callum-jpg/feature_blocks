import dask.array
import skimage
from feature_blocks.models import available_models
from feature_blocks.image import tissue_detection
from feature_blocks.task import create_task, read, infer, write
from feature_blocks.slice import generate_slices, filter_slices_by_mask, normalize_slices
from feature_blocks.backend import run_dask_backend
import logging
import numpy
import zarr
import functools
from tqdm import tqdm
import typing
import math
import multiprocessing

log = logging.getLogger(__name__)

def extract(
    zarr_path: str,
    feature_extraction_method: str,
    block_size: int, 
    save_path: str,
    calculate_mask: bool = False,
    image_downsample: int = 1,
    masked_block_value = numpy.nan,
):

    input_data = dask.array.from_zarr(zarr_path)

    assert input_data.ndim == 4, f"Expected zarr store to have 4 dimensions (C, Z, H, W). Got {input_data.ndim }."

    # Load the zarr store to be processed
    feature_extract_fn = _get_model(feature_extraction_method)

    # For each dimension, (channels, z, y, x) construct a list of
    # slice objects that will be used to index the zarr store.
    # XY slices are have shape (SIZE, SIZE) due to the slice object
    # step size being defined as SIZE
    regions = generate_slices(input_data.shape, block_size, [2, 3])

    if calculate_mask:
            mask = tissue_detection(
                input_data[:, 0, ::image_downsample, ::image_downsample].compute().transpose(1, 2, 0) # Downsample 8x and convert to YXC
            )
            # Mask has shape (Y, X). Resize this to the input image
            mask = skimage.transform.resize(
                mask,
                order=0,
                output_shape = input_data.shape[-2:]
            )
            mask = mask[numpy.newaxis, numpy.newaxis] # Add C an Z dimensions back

            log.info(f"Before masking: {len(regions)}")
            regions, _background_slices = filter_slices_by_mask(regions, mask)
            log.info(f"After masking: {len(regions)}")

            assert (
                len(regions) > 0
            ), "No foreground regions found after masking. Adjust or disable masking."

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

    # Prepare output zarr file
    # Optional parallel writing (I don't think we need this 
    # since there are no block overlaps/conflicts): 
    # https://zarr.readthedocs.io/en/v1.1.0/tutorial.html#parallel-computing-and-synchronization
    new_zarr = zarr.create(
        shape=output_shape,
        chunks=output_chunks,
        dtype=numpy.float32,
        store=save_path,
        overwrite=True,
        fill_value=numpy.nan,  # Value to use for empty chunks
    )

    # create_task_fn = functools.partial(
    #     create_task,
    #     input_data=input_data,
    #     new_zarr=new_zarr,
    #     chunk_size=block_size // output_chunks[2],
    #     feature_extract_fn=feature_extract_fn,
    # )

    # input_data = dask.delayed(input_data)

    # Create tasks. This is a list of delayed jobs to be run on the dask
    # backend
    tasks = []

    def update_region(region, block_size, feature_extract_fn):
        # Since we are 
        chunk_size = block_size // feature_extract_fn.output_shape[2]

        # Construct the new region        
        new_region = [
            slice(0, feature_extract_fn.n_features, None),
            slice(0, 1, None),
        ]  # Set the C and Z slices

        output_chunks = feature_extract_fn.output_shape

    # args = args_list = [(input_data, reg, block_size, feature_extract_fn, new_zarr) for reg in regions]

    # with multiprocessing.Pool() as pool:
    #     tasks = pool.starmap(create_task, tqdm(args, total=len(regions)))

    import time
    for reg in tqdm(regions, total=len(regions)):
        start_time = time.time()
        delayed_chunk = read(input_data, reg)
        # print(1, time.time() - start_time)

        # delayed_result = dask.delayed(infer, pure=True)(delayed_chunk, run_phikon)
        delayed_result = dask.delayed(infer, pure=True)(delayed_chunk, feature_extract_fn)
        # print(2, time.time() - start_time)

        new_region = [
            slice(0, feature_extract_fn.n_features, None),
            slice(0, 1, None),
        ]  # Set the C and Z slices
        chunk_size = block_size // output_chunks[2]
        new_region.extend(normalize_slices(reg[-2:]
        , chunk_size))  # Reduce the YX slices
        # print(3, time.time() - start_time)

        delayed_write = dask.delayed(write)(new_zarr, delayed_result, new_region)
        # print(4, time.time() - start_time)

        tasks.append(delayed_write)

    run_dask_backend(tasks)

def _get_model(model: typing.Callable | str) -> "torch.nn.Module":
    if isinstance(model, str):
        assert model in available_models, (
            f"'{model}' is not a valid model name. Valid names are: {', '.join(list(available_models.keys()))}"
        )

        return available_models[model]()

    return model

def create_task(input_data, region, block_size, feature_extract_fn, new_zarr):
    delayed_chunk = read(input_data, region)

    delayed_result = dask.delayed(infer)(delayed_chunk, feature_extract_fn)

    new_region = [
        slice(0, feature_extract_fn.n_features, None),
        slice(0, 1, None),
    ]  # Set the C and Z slices
    chunk_size = block_size // feature_extract_fn.output_shape[2]
    new_region.extend(normalize_slices(region[-2:]
    , chunk_size))  # Reduce the YX slices

    delayed_write = dask.delayed(write)(new_zarr, delayed_result, new_region)

    return delayed_write