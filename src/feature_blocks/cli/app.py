import typer

from feature_blocks.features import extract as _extract
import logging
from pathlib import Path
from dask_image.imread import imread
import tomllib
from spatial_image import to_spatial_image
from feature_blocks.backend import run_dask_backend
from feature_blocks.image import standardise_image

from dask_image.ndfilters import gaussian
import numpy

log = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def extract(
    config_file: str
):
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    input_path = Path(config["image_path"])

    save_path = "feature_blocks_" + input_path.stem + ".zarr"

    if input_path.suffix != ".zarr":
        # Load image
        image = imread(input_path)

        image, dimension_order = standardise_image(image, config["image_dimension_order"])
        
        # Update the input_path to now reflect the zarr store
        input_path = input_path.parent / (input_path.stem + ".zarr")

        log.info(f"Saving image as chunked zarr to: {input_path}")

        # Convert to a spatial_image
        # TODO: Utilize spatial_image features more
        image = (
            to_spatial_image(
                image, 
                dims=dimension_order
            )
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
            :, :, ::config.get("image_downsample", 1), ::config.get("image_downsample", 1)
            ].rechunk(
                (1, 1, config["block_size"], config["block_size"]
            )
        )

        # Create a dask graph for zarr saving
        image.to_zarr(
            input_path,
            compute=False,
            overwrite=True,
        ).compute()

    _extract(
        zarr_path=input_path,
        feature_extraction_method=config["feature_extraction_method"],
        block_size=config["block_size"],
        save_path=config["save_path"],
        calculate_mask=config["calculate_mask"],
        image_downsample=config["image_downsample"],
    )

    

@app.command()
def cluster(
    config_file: str
):
    pass


