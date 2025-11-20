import numpy
import zarr
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import dask.array
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from zarr.codecs import BloscCodec


def save_ome_zarr(
    array: Union[numpy.ndarray, dask.array.Array],
    output_path: Union[str, Path],
    chunks: Optional[Tuple[int, ...]] = None,
    shards: Optional[Tuple[int, ...]] = None,
    axes: Optional[str] = None,
    compression: str = None,
) -> None:
    """
    Save a numpy or dask array as an OME-Zarr file with Zarr v3 and sharding support.
    
    Parameters
    ----------
    array : numpy.ndarray or dask.array.Array
        The input array to save. Can be 2D-5D (e.g., YX, ZYX, CYX, TZYX, TCZYX).
    output_path : str or Path
        Path where the OME-Zarr file will be saved.
    chunks : tuple of int, optional
        Chunk size for the array. If None, uses array chunks (for dask) or auto-chunks.
    shards : tuple of int, optional
        Shard size for Zarr v3. Should be multiples of chunks. If None, no sharding.
    axes : str, optional
        Axis labels (e.g., "tczyx", "zyx"). If None, inferred from array dimensions.
    """
    output_path = Path(output_path)
    
    # Infer axes if not provided
    if axes is None:
        ndim = array.ndim
        axes_map = {
            2: "yx",
            3: "zyx",
            4: "czyx",
            5: "tczyx"
        }
        axes = axes_map.get(ndim, "".join([f"dim_{i}" for i in range(ndim)]))
    
    axes = axes.lower()
    
    # Handle chunks
    if chunks is None:
        if isinstance(array, dask.array.Array):
            chunks = array.chunksize
        else:
            # Auto-chunk based on array size
            chunks = tuple(min(s, 256) for s in array.shape)
    
    # Create zarr store with v3
    store = parse_url(output_path, mode="w").store
    
    # Create root group with zarr v3
    root = zarr.group(store=store, zarr_format=3, overwrite=True)
    
    # Prepare storage options for sharding
    storage_options = {
        "chunks": chunks,
    }
    
    if shards is not None:
        # Zarr v3 sharding configuration
        storage_options["shards"] = shards

    if compression == "zstd":
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    else:
        compressor = None
    
    # Write the image data using ome_zarr
    write_image(
        image=array,
        group=root,
        axes=axes,
        storage_options=storage_options,
        compute=True,  # Compute immediately for dask arrays
        compressors=compressor,
        scaler=None, # Do not create a resolution pyramid
    )

    z = zarr.open(output_path)["0"]
    
    print(f"Successfully saved OME-Zarr to {output_path}")
    print(z.info_complete())