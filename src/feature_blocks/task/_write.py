def write(zarr, data, region):
    zarr[tuple(region)] = data