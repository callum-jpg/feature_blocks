import numpy
from xarray import DataArray, DataTree

from spatialdata.models import SpatialElement


def get_spatial_element(
    element_dict: dict[str, SpatialElement],
    key: str | None = None,
    return_key: bool = False,
    as_spatial_image: bool = False,
    image_scale: int = None,
) -> SpatialElement | tuple[str, SpatialElement]:
    """Gets an element from a SpatialData object.

    Args:
        element_dict: Dictionnary whose values are spatial elements (e.g., `sdata.images`).
        key: Optional element key. If `None`, returns the only element (if only one).
        return_key: Whether to also return the key of the element.
        as_spatial_image: Whether to return the element as a `SpatialImage` (if it is a `DataTree`)

    Returns:
        If `return_key` is False, only the element is returned, else a tuple `(element_key, element)`
    """
    assert len(element_dict), "No spatial element was found in the dict."

    if key is not None:
        assert key in element_dict, f"Spatial element '{key}' not found."
        return _return_element(
            element_dict, key, return_key, as_spatial_image, image_scale
        )

    assert (
        len(element_dict) > 0
    ), "No spatial element found. Provide an element key to denote which element you want to use."
    assert (
        len(element_dict) == 1
    ), f"Multiple valid elements found: {', '.join(element_dict.keys())}. Provide an element key to denote which element you want to use."

    key = next(iter(element_dict.keys()))

    return _return_element(element_dict, key, return_key, as_spatial_image, image_scale)


def _return_element(
    element_dict: dict[str, SpatialElement],
    key: str,
    return_key: bool,
    as_spatial_image: bool,
    image_scale: int,
) -> SpatialElement | tuple[str, SpatialElement]:
    element = element_dict[key]

    if as_spatial_image and isinstance(element, DataTree):
        if image_scale is None:
            image_scale = 0
        element = next(iter(element[f"scale{image_scale}"].values()))

    return (key, element) if return_key else element


def normalise_rgb(image, mean, std):
    assert image.shape[-1] == 3, "Expected RGB image, with C in -1th dimension."
    # Convert image to float32 to ensure precision during calculations
    image = image.astype(numpy.float32)
    # Normalize each channel separately
    for i in range(3):  # assuming RGB channels
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]

    return image
