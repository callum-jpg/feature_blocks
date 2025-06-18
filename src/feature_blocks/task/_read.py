import zarr


def read(input_zarr_path, region):
    z = zarr.open(input_zarr_path)
    if len(region) == 4:
        # Read a block
        return z[region]
    elif len(region) == 5:
        # Read a ROI
        return z[region[1:]]
