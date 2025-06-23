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

log = logging.getLogger(__name__)

app = typer.Typer()


def load_config(config_file: str) -> dict:
    """Load configuration from TOML file."""
    with open(config_file, "rb") as f:
        return tomllib.load(f)

# def parse_

def load_and_process_image(config: dict) -> Path:
    """
    Load and process image according to configuration.
    
    Returns the path to the processed zarr file.
    """
    input_path = Path(config["image_path"])
    
    if input_path.suffix == ".zarr":
        # Image is already in zarr format
        return input_path
    
    # Load and process the image
    image = imread(input_path)
    
    # Standardise the image to CZYX
    image, dimension_order = standardise_image(
        image, config["image_dimension_order"]
    )
    
    # Create zarr path
    zarr_path = input_path.parent / (input_path.stem + ".zarr")
    
    log.info(f"Saving image as chunked zarr to: {zarr_path}")
    
    # Convert to spatial_image for intuitive dimensions
    image = (
        to_spatial_image(image, dims=dimension_order)
        .chunk("auto")
        .transpose("c", "z", "y", "x")
        .data
    )
    
    # Apply optional preprocessing (currently commented out)
    # image = normalise_rgb(
    #     image,
    #     mean=(0.5, 0.5, 0.5),
    #     std=(0.5, 0.5, 0.5),
    # )
    
    # SIGMA = 0.25
    # image = gaussian(image, sigma=(0, 0, SIGMA, SIGMA), mode="reflect", cval=0.0)
    
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
    
    segmentations = geopandas.read_file(segmentation_path)
    
    # Scale segmentations according to image downsampling
    downsample_factor = config.get("image_downsample", 1)
    segmentations.geometry = segmentations.scale(
        xfact=1/downsample_factor, 
        yfact=1/downsample_factor, 
        origin=(0, 0)
    )
    
    return segmentations



@app.command()
def extract(config_file: str):
    """
    Perform feature block extraction from an image file.
    """
    config = load_config(config_file)

    input_image_zarr_path = load_and_process_image(config)

    segmentations = load_segmentations(config)

    segmentation_path = config.get("segmentations", None)

    _extract(
        input_zarr_path=input_path,
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
