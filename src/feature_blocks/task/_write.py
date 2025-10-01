import zarr
from ome_zarr.io import parse_url


def write(zarr_path, data, region):
    # Open OME-Zarr store and write only to the required region
    # All zarr stores are OME-Zarr format with data at path "0"
    store = parse_url(zarr_path, mode="r+").store
    root = zarr.open_group(store=store, mode="r+")
    z = root["0"]

    if len(region) == 4:
        # Feature block (4D data)
        z[tuple(region)] = data
    else:  # len(region) == 2
        # ROI features (2D data)
        z[tuple(region)] = data.squeeze()
