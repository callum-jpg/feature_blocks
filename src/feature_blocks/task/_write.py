import zarr


def write(zarr_path, data, region):
    z = zarr.open(zarr_path)
    if len(region) == 4:
        # Feature block
        z[tuple(region)] = data
    elif len(region) == 2:
        # ROI features
        # Squeeze to drop ZYX dimensions since we do not need them for ROI
        z[tuple(region)] = data.squeeze()
