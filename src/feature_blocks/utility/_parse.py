import logging
import typing

log = logging.getLogger(__name__)


def parse_path(path, reader_fn: typing.Callable | None = None):
    """
    If path refers to a file, return path.

    If the path is a SpatialData object with ::data_key
    (example below), return this data object

    path_to_sdata.zarr::image_key
    """
    from feature_blocks.utility import get_spatial_element

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
