import dask.array
import tifffile
import numpy

def load_tiff_scale(tiff_path: str, level: int) -> dask.array.Array:
    """From a OME-TIFF, load only a specific level. Prevents loading
    of an entire image into memory."""
    with tifffile.imread(tiff_path, aszarr=True) as store:
        # If len is 2, zarr store only contains one array
        if len(store) == 2:
            image = dask.array.from_zarr(store)
        else:
            # Otherwise, load the required array
            image = dask.array.from_zarr(store, level)
    return image

def normalise_rgb(image, mean, std):
    # Convert image to float32 to ensure precision during calculations
    image = image.astype(numpy.float32)
    # Normalize each channel separately
    for i in range(3):  # assuming RGB channels
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]

    return image