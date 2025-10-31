import logging

import geopandas

from feature_blocks.utility import parse_path

log = logging.getLogger(__name__)


def load_segmentations(config: dict):
    """
    Load and process segmentations if specified in config.

    Returns segmentations GeoDataFrame or None.
    """
    segmentation_path = config.get("segmentations", None)

    if segmentation_path is None:
        return None

    segmentations = parse_path(segmentation_path, geopandas.read_file)

    # Set the index to the row number. We will use this
    # index value to slice the zarr array. That is,
    # row 0 will be in (0, N) of the zarr store.
    segmentations.index = range(len(segmentations))

    # Scale segmentations. Typically used to convert from
    # micron to pixel space
    segmentation_scale_factor = config.get("segmentation_scale_factor", None)

    if segmentation_scale_factor is not None:
        log.info(
            f"Scaling segmentation shapes with segmentation_scale_factor: {segmentation_scale_factor}"
        )
        # Only scale if needed.
        segmentations.geometry = segmentations.scale(
            segmentation_scale_factor, segmentation_scale_factor, origin=(0, 0)
        )

    # Scale segmentations according to image downsampling
    segmentations.geometry = segmentations.scale(
        xfact=1 / downsample_factor, yfact=1 / downsample_factor, origin=(0, 0)
    )

    return segmentations
