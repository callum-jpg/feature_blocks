import zarr


def write(zarr_path, data, region):
    z = zarr.open(zarr_path)
    z[tuple(region)] = data