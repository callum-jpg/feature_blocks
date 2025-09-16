import logging

import numpy
from rasterio import features
from shapely.geometry import box

log = logging.getLogger(__name__)


def rasterize_single_polygon(
    polygon_geometry,
    region_bounds: tuple,
    region_shape: tuple,
    object_id: int = 1,
) -> numpy.ndarray:
    """
    Rasterize a single polygon geometry within a specific region.

    Args:
        polygon_geometry: Single shapely geometry object
        region_bounds: (x_min, y_min, x_max, y_max) in coordinate space
        region_shape: (height, width) of the output raster
        object_id: ID to assign to the rasterized polygon (default: 1)

    Returns:
        numpy.ndarray: Rasterized mask containing only this polygon
    """
    x_min, y_min, x_max, y_max = region_bounds
    region_box = box(x_min, y_min, x_max, y_max)

    # Check if polygon intersects with the region
    if not polygon_geometry.intersects(region_box):
        return numpy.zeros(region_shape, dtype=numpy.int32)

    # Crop geometry to the region bounds
    cropped_geom = polygon_geometry.intersection(region_box)

    if cropped_geom.is_empty:
        return numpy.zeros(region_shape, dtype=numpy.int32)

    # Create transform for the region
    from rasterio.transform import from_bounds

    transform = from_bounds(
        x_min, y_min, x_max, y_max, region_shape[1], region_shape[0]
    )

    # Rasterize the single geometry
    mask = features.rasterize(
        [(cropped_geom, object_id)],
        out_shape=region_shape,
        transform=transform,
        fill=0,
        dtype=numpy.int32,
    )

    return mask


def slice_to_bounds(slice_obj: tuple, shape: tuple) -> tuple:
    """
    Convert a slice object to spatial bounds.

    Args:
        slice_obj: Tuple of slice objects (C, Z, Y, X)
        shape: Original data shape

    Returns:
        tuple: (x_min, y_min, x_max, y_max) bounds
    """
    # Extract Y and X slices (assuming CZYX format)
    y_slice = slice_obj[2] if len(slice_obj) > 2 else slice(None)
    x_slice = slice_obj[3] if len(slice_obj) > 3 else slice(None)

    # Convert slice to bounds
    y_start = y_slice.start if y_slice.start is not None else 0
    y_stop = y_slice.stop if y_slice.stop is not None else shape[2]
    x_start = x_slice.start if x_slice.start is not None else 0
    x_stop = x_slice.stop if x_slice.stop is not None else shape[3]

    # Return as (x_min, y_min, x_max, y_max)
    return (x_start, y_start, x_stop, y_stop)
