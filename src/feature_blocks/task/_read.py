import zarr
from dask.distributed import get_worker
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def read(input_zarr_path, region):
    # Try to use cached zarr store from worker plugin
    try:
        worker = get_worker()
        if hasattr(worker, "input_zarr"):
            z = worker.input_zarr
        else:
            # Try to open as OME-Zarr first, fall back to regular zarr
            try:
                store = parse_url(input_zarr_path, mode="r")
                reader = Reader(store)
                nodes = list(reader())
                z = nodes[0].data[0]  # First node, first resolution level
            except (KeyError, AttributeError, IndexError, TypeError):
                # Not OME-Zarr format, open as regular zarr
                z = zarr.open(input_zarr_path, mode="r")
    except (ValueError, AttributeError):
        # Not in a Dask worker context, open directly
        try:
            store = parse_url(input_zarr_path, mode="r")
            reader = Reader(store)
            nodes = list(reader())
            z = nodes[0].data[0]  # First node, first resolution level
        except (KeyError, AttributeError, IndexError, TypeError):
            # Not OME-Zarr format, open as regular zarr
            z = zarr.open(input_zarr_path, mode="r")

    if len(region) == 4:
        # Read a block
        data = z[region]
    elif len(region) == 2:
        # Read a ROI
        data = z[region[1:]]

    # If data is a Dask array (from OME-Zarr Reader), compute it to numpy
    if hasattr(data, 'compute'):
        data = data.compute()

    return data
