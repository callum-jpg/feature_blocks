"""
Generate synthetic data for benchmarking feature_blocks.

Creates zarr arrays and segmentations of various sizes for controlled testing.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import dask.array
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from feature_blocks.io import save_ome_zarr


def create_synthetic_image_zarr(
    output_path: str,
    image_size: Tuple[int, int, int, int],
    chunk_size: Optional[Tuple[int, int, int, int]] = None,
    shard_size: Optional[Tuple[int, int, int, int]] = None,
    dtype: str = "uint8",
    pattern: str = "random",
    seed: int = 42,
) -> str:
    """
    Create a synthetic image zarr array.

    Args:
        output_path: Path to save zarr array
        image_size: (C, Z, H, W) dimensions
        chunk_size: Chunk dimensions (defaults to (C, Z, 512, 512))
        shard_size: shard dimensions (defaults to chunk_size
        dtype: Data type for array
        pattern: "random", "gradient", or "checkerboard"
        seed: Random seed for reproducibility

    Returns:
        Path to created zarr array
    """
    np.random.seed(seed)

    C, Z, H, W = image_size
    if chunk_size is None:
        chunk_size = (C, Z, min(512, H), min(512, W))

    # Generate data based on pattern
    if pattern == "random":
        # Generate random noise
        if dtype == "uint8":
            data = np.random.randint(0, 256, size=image_size, dtype=np.uint8)
        else:
            data = np.random.rand(*image_size).astype(dtype)

    elif pattern == "gradient":
        # Create gradient pattern
        h_grad = np.linspace(0, 1, H)
        w_grad = np.linspace(0, 1, W)
        gradient = np.outer(h_grad, w_grad)

        if dtype == "uint8":
            gradient = (gradient * 255).astype(np.uint8)
        else:
            gradient = gradient.astype(dtype)

        # Tile across channels and z-slices
        data = np.tile(gradient[np.newaxis, np.newaxis, :, :], (C, Z, 1, 1))

    elif pattern == "checkerboard":
        # Create checkerboard pattern
        block_size = 64
        n_h = int(np.ceil(H / block_size))
        n_w = int(np.ceil(W / block_size))

        checkerboard = np.zeros((H, W), dtype=dtype)
        for i in range(n_h):
            for j in range(n_w):
                if (i + j) % 2 == 0:
                    h_start = i * block_size
                    h_end = min((i + 1) * block_size, H)
                    w_start = j * block_size
                    w_end = min((j + 1) * block_size, W)

                    value = 255 if dtype == "uint8" else 1.0
                    checkerboard[h_start:h_end, w_start:w_end] = value

        data = np.tile(checkerboard[np.newaxis, np.newaxis, :, :], (C, Z, 1, 1))

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Convert to dask array for efficient writing
    data = dask.array.from_array(data, chunks=chunk_size)

    # Create OME-Zarr
    zarr_store = save_ome_zarr(
        array=data,
        output_path=output_path,
        chunks=chunk_size,
        shards = shard_size,
    )

    print(f"Created synthetic zarr at {output_path}")
    print(f"  Shape: {image_size}")
    print(f"  Chunks: {chunk_size}")

    return output_path


def create_synthetic_segmentations(
    image_size: Tuple[int, int],
    n_regions: int,
    region_size_range: Tuple[int, int] = (50, 200),
    output_path: Optional[str] = None,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """
    Create synthetic segmentations (polygons) for centroid-based processing.

    Args:
        image_size: (H, W) dimensions
        n_regions: Number of regions to create
        region_size_range: (min, max) size for region diameter
        output_path: Optional path to save GeoJSON
        seed: Random seed

    Returns:
        GeoDataFrame with polygon geometries
    """
    np.random.seed(seed)

    H, W = image_size
    min_size, max_size = region_size_range

    polygons = []
    ids = []

    for i in range(n_regions):
        # Random center point
        cx = np.random.randint(max_size // 2, W - max_size // 2)
        cy = np.random.randint(max_size // 2, H - max_size // 2)

        # Random size
        size = np.random.randint(min_size, max_size)
        half_size = size // 2

        # Create square polygon
        polygon = Polygon(
            [
                (cx - half_size, cy - half_size),
                (cx + half_size, cy - half_size),
                (cx + half_size, cy + half_size),
                (cx - half_size, cy + half_size),
            ]
        )

        polygons.append(polygon)
        ids.append(i)

    gdf = gpd.GeoDataFrame({"id": ids, "geometry": polygons}, crs="EPSG:4326")

    if output_path:
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"Saved {n_regions} segmentations to {output_path}")

    return gdf


def create_test_scenario(
    name: str,
    image_size: Tuple[int, int, int, int],
    block_size: int,
    shard_size: int = None,
    n_regions: Optional[int] = None,
    base_dir: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """
    Create a complete test scenario with image and segmentations.

    Args:
        name: Scenario name
        image_size: (C, Z, H, W) dimensions
        n_regions: Number of regions (if None, estimated from image size)
        base_dir: Base directory (if None, uses temp directory)
        seed: Random seed

    Returns:
        Dictionary with paths to created files
    """
    base_dir = Path(base_dir) / name
    base_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = (image_size[0], image_size[1], block_size, block_size)

    if shard_size is not None:
        shard_size_tuple = (image_size[0], image_size[1], shard_size, shard_size)

    base_dir = Path(base_dir)

    C, Z, H, W = image_size

    # Create image
    image_path = base_dir / "image.zarr"
    create_synthetic_image_zarr(
        str(image_path),
        image_size=image_size,
        pattern="random",
        chunk_size=chunk_size,
        shard_size=shard_size_tuple,
        seed=seed,
    )

    # Create segmentations if requested
    segmentations_path = None
    if n_regions is not None:
        segmentations_path = base_dir / "segmentations.geojson"
        create_synthetic_segmentations(
            image_size=(H, W),
            n_regions=n_regions,
            output_path=str(segmentations_path),
            seed=seed,
        )

    scenario = {
        "name": name,
        "image_path": str(image_path),
        "segmentations_path": str(segmentations_path) if segmentations_path else None,
        "image_size": image_size,
        "n_regions": n_regions,
        "block_size": block_size,
        "shard_size": shard_size,
        "base_dir": str(base_dir),
    }

    return scenario


def create_scenarios(
    block_size: Union[Tuple[int], int],
    image_size: Union[Tuple[int], Tuple[Tuple[int]]],
    base_dir: str,
    shard_size: Union[Tuple[int], int]  = None,
    n_regions: Union[Tuple[int], int] = None,
) -> List[dict]:
    """
    Create a suite of test scenarios for scaling benchmarks.

    Generates images of increasing size to test zarr+dask scaling behavior.

    Returns:
        List of scenario dictionaries
    """
    scenarios = []

    if isinstance(block_size, int):
        block_size = [block_size]

    if isinstance(shard_size, int) or shard_size is None:
        shard_size = [shard_size]

    if not all(isinstance(i_s, tuple) for i_s in image_size):
        image_size = image_size

    if isinstance(n_regions, int) or n_regions is None:
        n_regions = [n_regions]

    for blk_sz in block_size:
        for shd_sz in shard_size:
            for img_sz in image_size:
                for n_reg in n_regions:

                    scenario_name = f"image_size_{img_sz[-1]}_block_size_{blk_sz}"
                    
                    if shd_sz is not None:
                        scenario_name += f"_shard_size_{shd_sz}"

                    scenarios.append(
                        create_test_scenario(
                            name=scenario_name,
                            base_dir=base_dir,
                            image_size=img_sz,
                            block_size=blk_sz,
                            shard_size=shd_sz,
                            n_regions=n_reg,
                        )
                    )

    print(f"\nCreated {len(scenarios)} test scenarios")
    return scenarios


def format_bytes(n_bytes: int) -> str:
    """Format byte size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}PB"
