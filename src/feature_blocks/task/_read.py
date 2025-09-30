import zarr
from dask.distributed import get_worker
from ome_zarr.io import parse_url


def read(input_zarr_path, region):
    # Try to use cached zarr store from worker plugin
    try:
        worker = get_worker()
        if hasattr(worker, 'input_zarr'):
            z = worker.input_zarr
        else:
            # Try to open as OME-Zarr first, fall back to regular zarr
            try:
                store = parse_url(input_zarr_path, mode="r").store
                root = zarr.group(store=store)
                z = root["0"]  # OME-Zarr stores full-resolution data at "0"
            except (KeyError, AttributeError):
                # Not OME-Zarr format, open as regular zarr
                z = zarr.open(input_zarr_path, mode='r')
    except (ValueError, AttributeError):
        # Not in a Dask worker context, open directly
        try:
            store = parse_url(input_zarr_path, mode="r").store
            root = zarr.group(store=store)
            z = root["0"]  # OME-Zarr stores full-resolution data at "0"
        except (KeyError, AttributeError):
            # Not OME-Zarr format, open as regular zarr
            z = zarr.open(input_zarr_path, mode='r')

    if len(region) == 4:
        # Read a block
        return z[region]
    elif len(region) == 2:
        # Read a ROI
        return z[region[1:]]
