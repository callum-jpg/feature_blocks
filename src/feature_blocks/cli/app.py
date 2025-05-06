import typer

from feature_blocks.features import extract as _extract
import logging
from pathlib import Path
from dask_image.imread import imread
import tomllib
from spatial_image import to_spatial_image
from feature_blocks.backend import run_dask_backend

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
        log.info("Saving image as chunked zarr")
        # Load image
        image = imread(input_path)
        
        # Update the input_path to now reflect the zarr store
        input_path = input_path.stem + ".zarr"

        # Convert to a spatial_image
        # TODO: Utilize spatial_image features more
        image = (
            to_spatial_image(
                image, 
                dims=config["image_dimension_order"]
            )
            .chunk("auto")
            .transpose("c", "z", "y", "x")
            .data
        )

        # Simple downsample
        image = image[
            :, :, ::config["image_downsample"], ::config["image_downsample"]
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


