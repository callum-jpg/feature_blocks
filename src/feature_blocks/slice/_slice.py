from itertools import product
from typing import List, Optional, Tuple

import numpy


def generate_nd_slices(
    shape: Tuple[int, int, int, int], size: int, slice_axes: Optional[List[int]] = None
) -> List[Tuple[slice, ...]]:
    """
    Generate slices over a multidimensional array.

    Parameters:
    - shape: tuple indicating the size of each dimension.
    - size: size of the chunks to slice (used only on specified axes).
    - slice_axes: list of axis indices to apply slicing to. Others use slice(None).

    Returns:
    - A list of slice tuples.
    """

    assert len(shape) == 4, "Expected shape of length 4 (C, Z, H, W)"

    ndim = len(shape)
    slice_axes = slice_axes or []

    # Build the list of ranges or single-step slices
    ranges = []
    for axis in range(ndim):
        if axis in slice_axes:
            # Create slices for this axis
            # Set stop, which is the maximum of this dimension
            stop = shape[axis]
            # Step is the size. Ie. the size of each bounding box
            step = size
            # Add the range fn to ranges. This fn will create the indices
            # we need for our slices``
            ranges.append(range(0, stop - 1, step))
        else:
            # Use slice(None) since this axis is not to be sliced
            ranges.append([None])

    slices = []
    for indices in product(*ranges):
        # ranges is a list of range functions (or None).
        # By creating the product of these range iterables,
        # we define the slice start and stop for each dimension.
        # Each range defined the **start** index of a given bounding
        # box, which is why we add "size".
        # If slice is None (ie. dimension is not to be iterated over)
        # we define the slice object as None
        slc = tuple(
            slice(i, i + size) if i is not None else slice(None) for i in indices
        )
        slices.append(slc)

    return slices


def generate_centroid_slices(
    shape: Tuple[int, int, int, int],
    size: int,
    segmentations: "geopandas.GeoDataFrame",
    id_col: str = None,
):
    """
    For segmentations provided in a GeoPandasDataframe, create slice objects around the
    centroid of each polygon.
    """

    assert len(shape) == 4, "Expected shape of length 4 (C, Z, H, W)"

    def polygon_bb(polygon):
        if id_col is None:
            # Name is the equivalent to index
            centroid_id = polygon.name
        else:
            centroid_id = polygon[id_col]

        y, x = round(polygon.geometry.centroid.y), round(polygon.geometry.centroid.x)

        # Amount to expand out from XY by
        half_size = size // 2

        y_min = max(0, y - half_size)
        y_max = min(shape[2], y + half_size)

        x_min = max(0, x - half_size)
        x_max = min(shape[3], x + half_size)

        slc = (
            slice(None),
            slice(None),
            slice(y_min, y_max),
            slice(x_min, x_max),
        )

        return centroid_id, slc

    slices = segmentations.apply(polygon_bb, axis=1).tolist()

    return slices


def generate_centroid_slices_with_single_masks(
    shape: Tuple[int, int, int, int],
    size: int,
    segmentations: "geopandas.GeoDataFrame",
    id_col: str = None,
):
    """
    For segmentations provided in a GeoPandasDataframe, create slice objects around the
    centroid of each polygon with individual mask data for CellProfiler feature extraction.

    Each returned tuple represents one distributed task that will extract features
    from a single segmentation polygon.

    Args:
        shape: Shape of the full image (C, Z, H, W)
        size: Size of the region to extract around each centroid
        segmentations: GeoDataFrame with polygon geometries
        id_col: Column to use for object IDs, defaults to index

    Returns:
        List of tuples: (centroid_id, slice_obj, mask_data)
        where mask_data contains only the single segmentation polygon
    """

    from ..utility._rasterize import rasterize_single_polygon

    assert len(shape) == 4, "Expected shape of length 4 (C, Z, H, W)"

    def polygon_bb_with_single_mask(polygon):
        if id_col is None:
            # Name is the equivalent to index
            centroid_id = polygon.name
        else:
            centroid_id = polygon[id_col]

        y, x = round(polygon.geometry.centroid.y), round(polygon.geometry.centroid.x)

        # Amount to expand out from XY by
        half_size = size // 2

        y_min = max(0, y - half_size)
        y_max = min(shape[2], y + half_size)

        x_min = max(0, x - half_size)
        x_max = min(shape[3], x + half_size)

        slc = (
            slice(None),
            slice(None),
            slice(y_min, y_max),
            slice(x_min, x_max),
        )

        # Calculate region bounds and shape for mask rasterization
        region_bounds = (x_min, y_min, x_max, y_max)
        region_shape = (y_max - y_min, x_max - x_min)

        # Rasterize only this single polygon
        mask_data = rasterize_single_polygon(
            polygon.geometry, region_bounds, region_shape, object_id=1
        )

        return centroid_id, slc, mask_data

    slices_with_masks = segmentations.apply(
        polygon_bb_with_single_mask, axis=1
    ).tolist()

    return slices_with_masks


def filter_slices_by_mask(
    slices: List[Tuple[slice, ...]], mask_array: numpy.ndarray
) -> List[Tuple[slice, ...]]:
    """
    Filters a list of slices, keeping only those for which the corresponding
    region in mask_array is fully equal to 1.

    Parameters:
    - slices: list of tuples of slices.
    - mask_array: ndarray of the same shape as the data the slices apply to.

    Returns:
    - A list of slice tuples where mask_array[slc] == 1 for all elements.
    """
    foreground = []
    background = []

    for slc in slices:
        if numpy.any(mask_array[slc] > 0):
            foreground.append(slc)
        else:
            background.append(slc)

    return foreground, background


def normalize_slices(slices, step=200):
    """Reduce slice indices based on a scaling factor (step).

    This is used when an aggregation has been applied to an array.
    For example, an array of [4, 4] aggregated to [1, 1] will require
    a slice downsampling to fit in the zarr store."""
    return [slice(s.start // step, s.stop // step) for s in slices]
