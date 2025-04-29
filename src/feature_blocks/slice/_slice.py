from itertools import product
from typing import List, Optional, Tuple

import numpy as np


def generate_slices(
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
            stop = shape[axis]
            step = size
            ranges.append(range(0, stop - 1, step))
        else:
            ranges.append([None])  # Use slice(None)

    slices = []
    for indices in product(*ranges):
        slc = tuple(
            slice(i, i + size) if i is not None else slice(None)
            for i, axis in zip(indices, range(ndim))
        )
        slices.append(slc)

    return slices


def filter_slices_by_mask(
    slices: List[Tuple[slice, ...]], mask_array: np.ndarray
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
    foreground = [slc for slc in slices if np.any(mask_array[slc] > 0)]

    background = [slc for slc in slices if slc not in foreground]

    return foreground, background


def normalize_slices(slices, step=200):
    """Reduce slice indices based on a scaling factor (step).

    This is used when an aggregation has been applied to an array.
    For example, an array of [4, 4] aggregated to [1, 1] will require
    a slice downsampling to fit in the zarr store."""
    return [slice(s.start // step, s.stop // step) for s in slices]
