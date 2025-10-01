import zarr
from ome_zarr.io import parse_url


def read(input_zarr_path, region):
    # Open OME-Zarr store and read only the required chunk
    # OME-Zarr stores data at path "0"
    store = parse_url(input_zarr_path, mode="r").store
    root = zarr.open_group(store=store, mode="r")
    z = root["0"]  # OME-Zarr data is at "0"

    if len(region) == 4:
        # Read a block
        data = z[region]
    elif len(region) == 2:
        # Read a ROI
        data = z[region[1:]]

    return data
