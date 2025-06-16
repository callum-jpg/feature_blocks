import dask.array

from feature_blocks.slice import normalize_slices


@dask.delayed
def create_task(region, input_data, new_zarr, chunk_size, feature_extract_fn):
    assert (
        region[-1].stop % chunk_size == 0
    ), "Width not divisible without remainder by chunk size"
    assert (
        region[-2].stop % chunk_size == 0
    ), "Height not divisible without remainder by chunk size"

    # Delay the feature extraction class (computed on save)
    data_new = feature_extract_fn(input_data[region])

    # Since we perform a reduction (ie. one region is
    # reduced to a single feature vector) find the index
    # of this current region
    new_region = [
        slice(0, feature_extract_fn.n_features, None),
        slice(0, 1, None),
    ]  # Set the C and Z slices

    new_region.extend(normalize_slices(region[-2:], chunk_size))  # Reduce the YX slices

    new_zarr[tuple(new_region)] = data_new
