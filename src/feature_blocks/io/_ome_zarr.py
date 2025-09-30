"""OME-Zarr format utilities for cloud-optimized storage."""

import logging

import numpy
import zarr
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

log = logging.getLogger(__name__)


def create_ome_zarr_output(
    output_zarr_path: str,
    shape: tuple,
    chunks: tuple,
    dtype=numpy.float32,
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
            axes = ["observation", "feature"]
        else:
            axes = [f"axis_{i}" for i in range(len(shape))]

    # Create the zarr store with OME-Zarr metadata
    store = parse_url(output_zarr_path, mode="w").store
    root = zarr.group(store=store, overwrite=True, synchronizer=synchronizer)

    # Set default compressor if not provided
    if compressor is None:
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)

    # Create the array directly in the root group
    # For feature data, we typically don't need multi-resolution pyramids
    array = root.create_dataset(
        "0",  # OME-Zarr uses "0" for the full-resolution data
        shape=shape,
        chunks=chunks,
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
    multiscales = [
        {
            "version": fmt.version,
            "name": "features",
            "axes": [{"name": ax, "type": "space" if ax in ["x", "y", "z"] else "channel" if ax == "c" else "custom"}
                     for ax in axes],
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


def write_to_ome_zarr(zarr_path: str, data: numpy.ndarray, region: tuple):
    """
    Write data to an OME-Zarr store.

    This function is compatible with the existing write interface but
    works with OME-Zarr formatted stores.

    Args:
        zarr_path: Path to the OME-Zarr store
        data: Data to write
        region: Region to write to (as slice objects or indices)
    """
    # Open the OME-Zarr store
    store = parse_url(zarr_path, mode="r+").store
    root = zarr.group(store=store)

    # Access the full-resolution data (always at path "0")
    z = root["0"]

    if len(region) == 4:
        # Feature block
        z[tuple(region)] = data
    elif len(region) == 2:
        # ROI features - squeeze to drop ZYX dimensions
        z[tuple(region)] = data.squeeze()


def read_from_ome_zarr(zarr_path: str, region: tuple):
    """
    Read data from an OME-Zarr store.

    Args:
        zarr_path: Path to the OME-Zarr store
        region: Region to read (as slice objects or indices)

    Returns:
        numpy.ndarray: The requested data
    """
    # Open the OME-Zarr store
    store = parse_url(zarr_path, mode="r").store
    root = zarr.group(store=store)

    # Access the full-resolution data (always at path "0")
    z = root["0"]

    if len(region) == 4:
        # Read a block
        return z[region]
    elif len(region) == 2:
        # Read a ROI
        return z[region[1:]]