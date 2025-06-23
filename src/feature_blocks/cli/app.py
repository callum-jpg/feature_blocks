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


@app.command()
def extract(config_file: str):
    """
    Perform feature block extraction from an image file.
    """
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    input_path = Path(config["image_path"])

    save_path = "feature_blocks_" + input_path.stem + ".zarr"

    if input_path.suffix != ".zarr":
        # Load image
        image = imread(input_path)

        # Standarise the image to CZYX
        image, dimension_order = standardise_image(
            image, config["image_dimension_order"]
        )

        # Update the input_path to now reflect the zarr store
        input_path = input_path.parent / (input_path.stem + ".zarr")

        log.info(f"Saving image as chunked zarr to: {input_path}")

        # Convert to a spatial_image for intuitive dimensions
        image = (
            to_spatial_image(image, dims=dimension_order)
            .chunk("auto")
            .transpose("c", "z", "y", "x")
            .data
        )

        # image = normalise_rgb(
        #     image,
        #     mean=(0.5, 0.5, 0.5),
        #     std=(0.5, 0.5, 0.5),
        # )

        # SIGMA = 0.25
        # image = gaussian(image, sigma=(0, 0, SIGMA, SIGMA), mode="reflect", cval=0.0)

        # Simple downsample
        image = image[
            :,
            :,
            :: config.get("image_downsample", 1),
            :: config.get("image_downsample", 1),
        ].rechunk((1, 1, config["block_size"], config["block_size"]))

        # Create a dask graph for zarr saving
        # TODO: Would distributed write improve speed?
        image.to_zarr(
            input_path,
            compute=False,
            overwrite=True,
        ).compute()

    segmentation_path = config.get("segmentations", None)

    if segmentation_path is not None:
        segmentations = geopandas.read_file(segmentation_path)

        segmentations.geometry = segmentations.scale(
            xfact=1/config.get("image_downsample", 1), 
            yfact=1/config.get("image_downsample", 1), 
            origin=(0, 0)
        )
    else:
        segmentations = None
        


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
