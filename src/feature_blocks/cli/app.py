import logging
import tomllib
from pathlib import Path

import typer
from dask_image.imread import imread
from spatial_image import to_spatial_image

from feature_blocks import FeatureBlockConstants
from feature_blocks.features import extract as _extract
from feature_blocks.image import standardise_image, zarr_exists
from feature_blocks.segmentation import load_segmentations
from feature_blocks.utility import parse_path

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
        feature_extraction_method=config.get("feature_extraction_method"),
        segmentations=segmentations,
        block_method=config.get("block_method", "block"),
        block_size=config.get("block_size"),
        output_zarr_path=config.get("save_path"),
        n_workers=config.get("n_workers", 1),
        python_path=config.get("python_path", "python"),
        memory=config.get("memory", "16GB"),
        calculate_mask=config.get("calculate_mask"),
        image_downsample=config.get("image_downsample"),
    )


def load_config(config_file: str) -> dict:
    """Load configuration from TOML file."""
    with open(config_file, "rb") as f:
        return tomllib.load(f)


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
    image, dimension_order = standardise_image(image, config["image_dimension_order"])

    # Create zarr save path
    zarr_path = (
        input_path.parent
        / FeatureBlockConstants.FEATURE_BLOCK_CACHE_DIR
        / FeatureBlockConstants.ZARR_IMAGE_NAME
    )

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

    log.info("Checking if image has already been saved to zarr...")
    if zarr_exists(zarr_path, image):
        log.info(f"Saving image as chunked zarr to: {zarr_path}")
        # Save to zarr if it hasn't already been saved.
        image.to_zarr(
            zarr_path,
            compute=False,
            overwrite=True,
        ).compute()
    else:
        log.info(f"Loading existing zarr store: {zarr_path}")

    return zarr_path
