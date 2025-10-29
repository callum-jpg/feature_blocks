"""
Generate synthetic data for benchmarking feature_blocks.

Creates zarr arrays and segmentations of various sizes for controlled testing.
"""

import numpy as np
import zarr
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
import tempfile
from typing import Tuple, Optional, List
import shutil
import dask.array

from feature_blocks.io import create_ome_zarr_output


def create_synthetic_image_zarr(
    output_path: str,
    image_size: Tuple[int, int, int, int],
    chunk_size: Optional[Tuple[int, int, int, int]] = None,
    dtype: str = "uint8",
    pattern: str = "random",
    seed: int = 42
) -> str:
    """
    Create a synthetic image zarr array.

    Args:
        output_path: Path to save zarr array
        image_size: (C, Z, H, W) dimensions
        chunk_size: Chunk dimensions (defaults to (C, Z, 512, 512))
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

    # Create OME-Zarr output using the existing utility
    zarr_store = create_ome_zarr_output(
        output_zarr_path=output_path,
        shape=image_size,
        chunks=chunk_size,
        dtype=data.dtype,
        axes=["c", "z", "y", "x"],
        fill_value=0.0,
    )

    # Write the data
    data.to_zarr(zarr_store, compute=True)

    print(f"Created synthetic zarr at {output_path}")
    print(f"  Shape: {image_size}")
    print(f"  Chunks: {chunk_size}")

    return output_path


def create_synthetic_segmentations(
    image_size: Tuple[int, int],
    n_regions: int,
    region_size_range: Tuple[int, int] = (50, 200),
    output_path: Optional[str] = None,
    seed: int = 42
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
        polygon = Polygon([
            (cx - half_size, cy - half_size),
            (cx + half_size, cy - half_size),
            (cx + half_size, cy + half_size),
            (cx - half_size, cy + half_size),
        ])

        polygons.append(polygon)
        ids.append(i)

    gdf = gpd.GeoDataFrame({
        "id": ids,
        "geometry": polygons
    }, crs="EPSG:4326")

    if output_path:
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"Saved {n_regions} segmentations to {output_path}")

    return gdf


def create_test_scenario(
    name: str,
    image_size: Tuple[int, int, int, int],
    n_regions: Optional[int] = None,
    base_dir: Optional[str] = None,
    seed: int = 42
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
    if base_dir is None:
        base_dir = tempfile.mkdtemp(prefix=f"benchmark_{name}_")
    else:
        base_dir = Path(base_dir) / name
        base_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(base_dir)

    C, Z, H, W = image_size

    # Create image
    image_path = base_dir / "image.zarr"
    create_synthetic_image_zarr(
        str(image_path),
        image_size=image_size,
        pattern="random",
        seed=seed
    )

    # Create segmentations if requested
    segmentations_path = None
    if n_regions is not None:
        segmentations_path = base_dir / "segmentations.geojson"
        create_synthetic_segmentations(
            image_size=(H, W),
            n_regions=n_regions,
            output_path=str(segmentations_path),
            seed=seed
        )

    scenario = {
        "name": name,
        "image_path": str(image_path),
        "segmentations_path": str(segmentations_path) if segmentations_path else None,
        "image_size": image_size,
        "n_regions": n_regions,
        "base_dir": str(base_dir)
    }

    return scenario


def create_scaling_scenarios(base_dir: Optional[str] = None) -> List[dict]:
    """
    Create a suite of test scenarios for scaling benchmarks.

    Generates images of increasing size to test zarr+dask scaling behavior.

    Returns:
        List of scenario dictionaries
    """
    scenarios = []

    # Small images (should fit in memory, dask overhead may dominate)
    scenarios.append(create_test_scenario(
        name="small_512",
        image_size=(3, 1, 512, 512),
        n_regions=10,
        base_dir=base_dir,
        seed=42
    ))

    scenarios.append(create_test_scenario(
        name="small_1024",
        image_size=(3, 1, 1024, 1024),
        n_regions=25,
        base_dir=base_dir,
        seed=43
    ))

    # Medium images (starting to benefit from chunking)
    scenarios.append(create_test_scenario(
        name="medium_2048",
        image_size=(3, 1, 2048, 2048),
        n_regions=100,
        base_dir=base_dir,
        seed=44
    ))

    scenarios.append(create_test_scenario(
        name="medium_4096",
        image_size=(3, 1, 4096, 4096),
        n_regions=256,
        base_dir=base_dir,
        seed=45
    ))

    # Large images (zarr+dask should excel here)
    scenarios.append(create_test_scenario(
        name="large_8192",
        image_size=(3, 1, 8192, 8192),
        n_regions=500,
        base_dir=base_dir,
        seed=46
    ))

    scenarios.append(create_test_scenario(
        name="large_16384",
        image_size=(3, 1, 16384, 16384),
        n_regions=1000,
        base_dir=base_dir,
        seed=47
    ))

    # Very large (extreme case)
    scenarios.append(create_test_scenario(
        name="xlarge_32768",
        image_size=(3, 1, 32768, 32768),
        n_regions=2000,
        base_dir=base_dir,
        seed=48
    ))

    print(f"\nCreated {len(scenarios)} test scenarios")
    return scenarios


def cleanup_scenario(scenario: dict):
    """Remove files created for a scenario."""
    base_dir = Path(scenario["base_dir"])
    if base_dir.exists():
        shutil.rmtree(base_dir)
        print(f"Cleaned up {base_dir}")


def format_bytes(n_bytes: int) -> str:
    """Format byte size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}PB"


if __name__ == "__main__":
    # Example usage
    print("Creating scaling scenarios...")
    scenarios = create_scaling_scenarios(base_dir="/tmp/benchmarks")

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Image: {scenario['image_path']}")
        print(f"  Size: {scenario['image_size']}")
        if scenario['segmentations_path']:
            print(f"  Regions: {scenario['n_regions']}")
