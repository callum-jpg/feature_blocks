import zarr
from dask.distributed import get_worker
from ome_zarr.io import parse_url


def write(zarr_path, data, region):
    # Try to use cached zarr store from worker plugin
    try:
        worker = get_worker()
        if hasattr(worker, 'output_zarr'):
            z = worker.output_zarr
        else:
            # Open as OME-Zarr format
            store = parse_url(zarr_path, mode="r+").store
            root = zarr.group(store=store)
            z = root["0"]  # OME-Zarr stores full-resolution data at "0"
    except (ValueError, AttributeError):
        # Not in a Dask worker context, open directly as OME-Zarr
        store = parse_url(zarr_path, mode="r+").store
        root = zarr.group(store=store)
        z = root["0"]  # OME-Zarr stores full-resolution data at "0"

    if len(region) == 4:
        # Feature block
        z[tuple(region)] = data
    elif len(region) == 2:
        # ROI features
        # Squeeze to drop ZYX dimensions since we do not need them for ROI
        z[tuple(region)] = data.squeeze()
