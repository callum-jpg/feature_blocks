import numpy
import zarr


def read_with_mask(input_zarr_path, region_with_mask):
    """
    Read image data from zarr and combine with pre-computed mask data.

    Args:
        input_zarr_path: Path to the zarr store containing image data
        region_with_mask: Tuple of (centroid_id, slice_obj, mask_data)

    Returns:
        numpy.ndarray: Combined image+mask data with shape (C+1, Z, H, W)
                       where the last channel is the mask
    """
    centroid_id, slice_obj, mask_data = region_with_mask

    # Read image data
    z = zarr.open(input_zarr_path)
    image_data = z[slice_obj]  # Shape: (C, Z, H, W)

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
