"""OME-Zarr format utilities for cloud-optimized storage."""

import logging

import numpy
import zarr
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url

from zarr.codecs import BloscCodec

log = logging.getLogger(__name__)


def create_ome_zarr_output(
    output_zarr_path: str,
    shape: tuple,
    chunks: tuple,
    dtype=numpy.float32,
    shards = None,
    axes: list = None,
    compressor=None,
    synchronizer=None,
    fill_value=numpy.nan,
):
    """
    Create an OME-Zarr store for output features.

    OME-Zarr is a cloud-optimized format that includes metadata following
    the OME (Open Microscopy Environment) specification.

    Args:
        output_zarr_path: Path to output zarr store
        shape: Shape of the output array
        chunks: Chunk size for the array
        dtype: Data type for the array
        axes: List of axis names (e.g., ["c", "z", "y", "x"])
        compressor: Zarr compressor to use
        synchronizer: Zarr synchronizer for parallel writes
        fill_value: Value to use for uninitialized chunks

    Returns:
        zarr.Array: The created array
    """
    # Default axes for feature data
    if axes is None:
        if len(shape) == 4:
            axes = ["c", "z", "y", "x"]
        elif len(shape) == 2:
            # For 2D feature data, treat as 1D spatial + channels
            # OME-Zarr requires at least one spatial axis
            axes = ["y", "c"]  # y (observations as spatial dimension), c (features)
        else:
            axes = [f"axis_{i}" for i in range(len(shape))]

    # Create the zarr store with OME-Zarr metadata
    store = parse_url(output_zarr_path, mode="w").store
    root = zarr.group(store=store, overwrite=True, synchronizer=synchronizer)

    # Set default compressor if not provided
    if compressor is None:
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")

    # Create the array directly in the root group
    # For feature data, we typically don't need multi-resolution pyramids
    array = root.create_array(  
        "0",  # OME-Zarr uses "0" for the full-resolution data
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        compressor=compressor,
        fill_value=fill_value,
        overwrite=True,
    )

    # Write OME-Zarr metadata
    # This makes the data compatible with OME-Zarr viewers and tools
    fmt = CurrentFormat()

    # Create coordinate transformations (identity for now)
    coordinate_transformations = [[{"type": "identity"}]]

    # Write the multiscales metadata
    # Map axis names to OME-Zarr types (space, time, or channel)
    axis_type_map = {
        "x": "space",
        "y": "space",
        "z": "space",
        "c": "channel",
        "t": "time",
    }

    multiscales = [
        {
            "version": fmt.version,
            "name": "features",
            "axes": [
                {"name": ax, "type": axis_type_map.get(ax, "space")} for ax in axes
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": coordinate_transformations[0],
                }
            ],
            "coordinateTransformations": coordinate_transformations,
        }
    ]

    root.attrs["multiscales"] = multiscales

    # Add omero metadata for visualization (optional but helpful)
    if len(shape) == 4:  # Standard image-like data
        root.attrs["omero"] = {
            "channels": [
                {
                    "label": f"feature_{i}",
                    "color": "FFFFFF",
                    "window": {"start": 0, "end": 1, "min": 0, "max": 1},
                }
                for i in range(shape[0])
            ]
        }

    log.info(f"Created OME-Zarr output at {output_zarr_path}")
    log.info(f"  Shape: {shape}")
    log.info(f"  Chunks: {chunks}")
    log.info(f"  Axes: {axes}")
    log.info(f"  Compressor: {compressor}")

    return array
