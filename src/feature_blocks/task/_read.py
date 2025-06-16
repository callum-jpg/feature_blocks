import zarr


def read(input_zarr_path, region):
    z = zarr.open(input_zarr_path)
    return z[region]
