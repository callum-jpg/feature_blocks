from spatialdata.models import SpatialElement
from xarray import DataArray, DataTree

def get_spatial_element(
    element_dict: dict[str, SpatialElement],
    key: str | None = None,
    return_key: bool = False,
    as_spatial_image: bool = False,
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
        return _return_element(element_dict, key, return_key, as_spatial_image)

    assert (
        len(element_dict) > 0
    ), "No spatial element found. Provide an element key to denote which element you want to use."
    assert (
        len(element_dict) == 1
    ), f"Multiple valid elements found: {', '.join(element_dict.keys())}. Provide an element key to denote which element you want to use."

    key = next(iter(element_dict.keys()))

    return _return_element(element_dict, key, return_key, as_spatial_image)

def _return_element(
    element_dict: dict[str, SpatialElement], key: str, return_key: bool, as_spatial_image: bool
) -> SpatialElement | tuple[str, SpatialElement]:
    element = element_dict[key]

    if as_spatial_image and isinstance(element, DataTree):
        element = next(iter(element["scale0"].values()))

    return (key, element) if return_key else element