import numpy
import zarr
from ome_zarr.io import parse_url


def read_with_mask(input_zarr_path, region_with_mask, mask_store_path):
    """
    Read image data from zarr and combine with mask data loaded from mask store.

    Args:
        input_zarr_path: Path to the zarr store containing image data
        region_with_mask: Tuple of (centroid_id, slice_obj, mask_index)
        mask_store_path: Path to the zarr store containing mask data

    Returns:
        numpy.ndarray: Combined image+mask data with shape (C+1, Z, H, W)
                       where the last channel is the mask
    """
    centroid_id, slice_obj, mask_index = region_with_mask

    # Open OME-Zarr store and read only the required chunk
    store = parse_url(input_zarr_path, mode="r").store
    root = zarr.open_group(store=store, mode="r")
    z = root["0"]  # OME-Zarr data is at "0"

    # Open mask store and read only the required mask
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
    # mask_data is (H, W), we need (1, Z, H, W)
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
