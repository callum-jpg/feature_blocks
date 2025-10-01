import zarr
from ome_zarr.io import parse_url


def write(zarr_path, data, region):
    # Open zarr store and write only to the required region
    # For OME-Zarr (4D blocks), data is at path "0"
    # For regular zarr (2D centroid features), it's the root
    try:
        # Try OME-Zarr structure first
        store = parse_url(zarr_path, mode="r+").store
        root = zarr.open_group(store=store, mode="r+")
        z = root["0"]  # OME-Zarr data is at "0"
    except (KeyError, AttributeError, IndexError, TypeError):
        # Regular zarr (centroid features)
        z = zarr.open(zarr_path, mode="r+")

    if len(region) == 4:
        # Feature block
        z[tuple(region)] = data
    elif len(region) == 2:
        # ROI features
        # Squeeze to drop ZYX dimensions since we do not need them for ROI
        z[tuple(region)] = data.squeeze()
