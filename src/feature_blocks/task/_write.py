import zarr
from ome_zarr.io import parse_url
import gc

def write(zarr_path, data, region):
    """Write data to zarr store.

    If a ZarrHandlePlugin is registered on the worker, uses the cached
    zarr handle. Otherwise, opens the store directly (backward compatible).
    """
    z = None

    try:
        # Try to use cached handle from worker plugin
        from distributed import get_worker
        worker = get_worker()
        if 'zarr_output' in worker.data:
            z = worker.data['zarr_output']
    except (ValueError, ImportError):
        # Not in a Dask worker context or distributed not available
        pass

    # Fallback to opening store directly if not cached
    if z is None:
        store = parse_url(zarr_path, mode="r+").store
        root = zarr.open_group(store=store, mode="r+")
        z = root["0"]

    if len(region) == 4:
        # Feature block (4D data)
        z[tuple(region)] = data
    else:  # len(region) == 2
        # ROI features (2D data)
        z[tuple(region)] = data.squeeze()

    del data
    gc.collect()