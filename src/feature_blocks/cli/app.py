import logging
import tomllib
from pathlib import Path

import numpy
import typer
from dask_image.imread import imread
from dask_image.ndfilters import gaussian
from spatial_image import to_spatial_image

import geopandas

from feature_blocks.backend import run_dask_backend
from feature_blocks.features import extract as _extract
from feature_blocks.image import standardise_image
from feature_blocks.utility import get_spatial_element
from feature_blocks import FeatureBlockConstants

import typing

log = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def extract(config_file: str):
    """
    Perform feature block extraction from an image file.
    """
    config = load_config(config_file)

    # We take the input_zarr_path since it allows each dask worker
    # to load the chunk needed from the zarr store. ie. the whole
    # array is not loaded into memory
    input_zarr_path = load_and_process_image(config)

    segmentations = load_segmentations(config)

    _extract(
        input_zarr_path=input_zarr_path,
        feature_extraction_method=config["feature_extraction_method"],
        segmentations = segmentations,
        block_method = config.get("block_method", "block"),
        block_size=config["block_size"],
        output_zarr_path=config["save_path"],
        calculate_mask=config["calculate_mask"],
        image_downsample=config["image_downsample"],
    )


@app.command()
def cluster(config_file: str):
    """
    Cluster embeddings from a zarr store.
    """
    pass


def load_config(config_file: str) -> dict:
    """Load configuration from TOML file."""
    with open(config_file, "rb") as f:
        return tomllib.load(f)

def parse_path(path, reader_fn: typing.Callable | None = None):
    """
    If path refers to a file, return path.

    If the path is a SpatialData object with ::data_key
    (example below), return this data object

    path_to_sdata.zarr::image_key
    """

    if "::" in path:
        import spatialdata
        
        sdata_path, data_key = path.split("::")

        log.info(f"Loading {data_key} from {sdata_path}")
        
        assert sdata_path.endswith(".zarr"), "SpatialData files must be zarr."

        sdata = spatialdata.read_zarr(sdata_path)

        data = get_spatial_element(sdata, data_key, as_spatial_image=True)
        
        return data
    else:
        log.info(f"Loading {path}...")
        return reader_fn(path)


def load_and_process_image(config: dict) -> Path:
    """
    Load and process image according to configuration.
    
    Returns the path to the processed zarr file.
    """
    input_path = Path(config["image_path"])

    if input_path.suffix == ".zarr":
        # Image is already in zarr format
        return input_path

    image = parse_path(input_path.as_posix(), imread)
    
    # Standardise the image to CZYX
    image, dimension_order = standardise_image(
        image, config["image_dimension_order"]
    )
    
    # Create zarr save path
    zarr_path = input_path.parent / FeatureBlockConstants.FEATURE_BLOCK_CACHE_DIR / FeatureBlockConstants.ZARR_IMAGE_NAME
    
    log.info(f"Saving image as chunked zarr to: {zarr_path}")
    
    # Convert to spatial_image for intuitive dimensions
    image = (
        to_spatial_image(image, dims=dimension_order)
        .chunk("auto")
        .transpose("c", "z", "y", "x")
        .data
    )

    # Apply downsampling
    downsample_factor = config.get("image_downsample", 1)
    image = image[
        :,
        :,
        ::downsample_factor,
        ::downsample_factor,
    ].rechunk((1, 1, config["block_size"], config["block_size"]))
    
    # Save to zarr
    image.to_zarr(
        zarr_path,
        compute=False,
        overwrite=True,
    ).compute()
    
    return zarr_path

def load_segmentations(config: dict):
    """
    Load and process segmentations if specified in config.
    
    Returns segmentations GeoDataFrame or None.
    """
    segmentation_path = config.get("segmentations", None)

    if segmentation_path is None:
        return None
    
    segmentations = parse_path(segmentation_path, geopandas.read_file)

    # Scale segmentations. Typically used to convert from 
    # micron to pixel space
    segmentation_scale_factor = config.get("segmentation_scale_factor", None)

    if segmentation_scale_factor is not None:
        log.info(f"Scaling segmentation shapes with segmentation_scale_factor: {segmentation_scale_factor}")
        # Only scale if needed.
        segmentations.geometry = segmentations.scale(
            segmentation_scale_factor, 
            segmentation_scale_factor, 
            origin=(0, 0)
        )
    
    # Scale segmentations according to image downsampling
    downsample_factor = config.get("image_downsample", 1)
    segmentations.geometry = segmentations.scale(
        xfact=1/downsample_factor, 
        yfact=1/downsample_factor, 
        origin=(0, 0)
    )
    
    return segmentations