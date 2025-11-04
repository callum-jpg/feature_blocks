import numpy
import zarr
from ome_zarr.io import parse_url


def read(input_zarr_path, region):
    """Read data from zarr store.

    If a ZarrHandlePlugin is registered on the worker, uses the cached
    zarr handle. Otherwise, opens the store directly (backward compatible).
    """
    try:
        # Try to use cached handle from worker plugin
        from distributed import get_worker
        worker = get_worker()
        if 'zarr_input' in worker.data:
            z = worker.data['zarr_input']
            return z[region]
    except (ValueError, ImportError):
        # Not in a Dask worker context or distributed not available
        pass

    # Fallback to opening store directly (backward compatible)
    store = parse_url(input_zarr_path, mode="r").store
    root = zarr.open_group(store=store, mode="r")
    z = root["0"]

    return z[region]


def read_with_mask(input_zarr_path, region_with_mask, mask_store_path):
    """Read image data and combine with mask data from mask store.

    If a ZarrHandlePlugin is registered on the worker, uses the cached
    zarr handles. Otherwise, opens the stores directly (backward compatible).
    """
    centroid_id, slice_obj, mask_index = region_with_mask

    # Try to use cached handles from worker plugin
    z = None
    mask_store = None

    try:
        from distributed import get_worker
        worker = get_worker()

        if 'zarr_input' in worker.data:
            z = worker.data['zarr_input']

        if 'zarr_mask' in worker.data:
            mask_store = worker.data['zarr_mask']
    except (ValueError, ImportError):
        # Not in a Dask worker context or distributed not available
        pass

    # Fallback to opening stores directly if not cached
    if z is None:
        store = parse_url(input_zarr_path, mode="r").store
        root = zarr.open_group(store=store, mode="r")
        z = root["0"]

    if mask_store is None:
        mask_store = zarr.open(mask_store_path, mode="r")

    # Read image data
    image_data = z[slice_obj]  # Shape: (C, Z, H, W)

    # Load mask data from zarr store
    mask_data = mask_store[mask_index]  # Shape: (H, W)

    # Ensure mask has the same spatial dimensions as image
    _, _, img_h, img_w = image_data.shape
    mask_h, mask_w = mask_data.shape

    if (mask_h, mask_w) != (img_h, img_w):
        # Resize mask to match image dimensions if needed
        import skimage.transform

        mask_data = skimage.transform.resize(
            mask_data, (img_h, img_w), order=0, preserve_range=True
        ).astype(numpy.int32)

    # Add mask as additional channel
    mask_channel = mask_data[numpy.newaxis, numpy.newaxis, :, :]  # (1, 1, H, W)

    # Repeat mask for all Z slices if needed
    _, z_dim, _, _ = image_data.shape
    if z_dim > 1:
        mask_channel = numpy.repeat(mask_channel, z_dim, axis=1)  # (1, Z, H, W)

    # Combine image and mask
    combined_data = numpy.concatenate(
        [image_data, mask_channel], axis=0
    )  # (C+1, Z, H, W)

    return combined_data
