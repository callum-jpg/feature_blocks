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

def normalise_rgb(image, mean, std, uint8: bool = False):
    # Convert image to float32 to ensure precision during calculations
    image = image.astype(numpy.float32)
    # Normalize each channel separately
    for i in range(3):  # assuming RGB channels
        image[i, ...] = (image[i, ...] - mean[i]) / std[i]

    if uint8:
        image = numpy.clip((image - image.min()) / (image.max() - image.min()) * 255, 0, 255)
        image = image.astype(numpy.uint8)

    return image

def standardise_image(
    image,
    dimension_order: tuple[str]
    ):
    """
    Standardise a dask array to have a fixed number of dimensions
    in the dimension order CZYX.
    """

    assert image.ndim <= 4, f"Expected an image with ndim <=4, got an image with {image.ndim} dimensions."

    dims = ("c", "z", "y", "x")

    missing = [i for i in dims if i not in dimension_order]

    for i, d in enumerate(dims):
        if d in missing:
            image = dask.array.expand_dims(image, axis=i)

    return image, dims