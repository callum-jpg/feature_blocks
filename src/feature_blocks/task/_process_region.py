import numpy
import zarr
from ome_zarr.io import parse_url
import gc
from ._read import read, read_with_mask
from ._infer import infer
from ._write import write

# Cache to avoid re-instantiating models on each call within the same worker
_model_cache = {}


def process_region(
    reg,
    input_zarr_path,
    model_identifier,
    n_features,
    output_zarr_path,
    block_size,
    output_chunks,
    mask_store_path=None,
):
    """Process a complete region: read -> infer -> write

    Args:
        reg: Region tuple (slices or with chunk_id/mask_index)
        input_zarr_path: Path to input zarr
        model_identifier: String model name (serialization-friendly) or callable
        n_features: Number of features the model produces
        output_zarr_path: Path to output zarr
        block_size: Size of blocks
        output_chunks: Output chunk shape
        mask_store_path: Path to zarr store containing mask data (for CellProfiler)

    Returns:
        None (writes directly to zarr)
    """
    if len(reg) == 3:
        # CellProfiler method with mask index (centroid + mask)
        chunk_id, chunk_slices, mask_index = reg
        chunk_data = read_with_mask(input_zarr_path, (chunk_id, chunk_slices, mask_index), mask_store_path)
        output_region = [chunk_id, slice(0, n_features)]
    elif len(reg) == 2:
        # ROI method (standard centroid)
        chunk_id, chunk_slices = reg[0], reg[1:][0]
        chunk_data = read(input_zarr_path, chunk_slices)
        output_region = [chunk_id, slice(0, n_features)]
    elif len(reg) == 4:
        # Block method
        chunk_data = read(input_zarr_path, reg)
        # Build where the new region will be in the output zarr
        from feature_blocks.slice import normalize_slices
        output_region = [
            slice(0, n_features, None),
            slice(0, 1, None),
        ]
        chunk_size = block_size // output_chunks[2]
        output_region.extend(normalize_slices(reg[-2:], chunk_size))
    else:
        raise ValueError(
            f"Region is of length {len(reg)} rather than the expected 2, 3, or 4. Region: {reg}"
        )

    result = infer(chunk_data, model_identifier)

    write(output_zarr_path, result, output_region)

    return None



