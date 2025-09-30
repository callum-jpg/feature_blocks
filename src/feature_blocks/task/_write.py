import zarr
from dask.distributed import get_worker
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def write(zarr_path, data, region):
    # Try to use cached zarr store from worker plugin
    try:
        worker = get_worker()
        if hasattr(worker, "output_zarr"):
            z = worker.output_zarr
        else:
            # Try to open as OME-Zarr first (for 4D block output), fall back to regular zarr (for 2D centroid features)
            try:
                store = parse_url(zarr_path, mode="r+")
                reader = Reader(store)
                nodes = list(reader())
                z = nodes[0].data[0]  # First node, first resolution level
            except (KeyError, AttributeError, IndexError, TypeError):
                # Not OME-Zarr, open as regular zarr (centroid features)
                z = zarr.open(zarr_path, mode="r+")
    except (ValueError, AttributeError):
        # Not in a Dask worker context, open directly
        try:
            store = parse_url(zarr_path, mode="r+")
            reader = Reader(store)
            nodes = list(reader())
            z = nodes[0].data[0]  # First node, first resolution level
        except (KeyError, AttributeError, IndexError, TypeError):
            # Not OME-Zarr, open as regular zarr (centroid features)
            z = zarr.open(zarr_path, mode="r+")

    if len(region) == 4:
        # Feature block
        z[tuple(region)] = data
    elif len(region) == 2:
        # ROI features
        # Squeeze to drop ZYX dimensions since we do not need them for ROI
        z[tuple(region)] = data.squeeze()
