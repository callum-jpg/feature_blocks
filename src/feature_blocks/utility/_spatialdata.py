from spatialdata.models import SpatialElement
from xarray import DataTree


def get_spatial_element(
    element_dict: dict[str, SpatialElement],
    key: str,
    as_spatial_image: bool = True,
) -> SpatialElement | tuple[str, SpatialElement]:
    """Gets an element from a SpatialData object.

    Args:
        element_dict: Dictionnary whose values are spatial elements (e.g., `sdata.images`).
        key: Element key.
        as_spatial_image: Whether to return the element as a `SpatialImage` (if it is a `DataTree`)

    Returns:
        SpatialData element.
    """
    # assert len(element_dict), "No spatial element was found in the dict."

    assert key in element_dict, f"Spatial element '{key}' not found."
    return _return_element(element_dict, key, as_spatial_image)


def _return_element(
    element_dict: dict[str, SpatialElement], key: str, as_spatial_image: bool
) -> SpatialElement | tuple[str, SpatialElement]:

    element = element_dict[key]

    if as_spatial_image and isinstance(element, DataTree):
        element = getattr(element, "scale0").image.data

    return element
