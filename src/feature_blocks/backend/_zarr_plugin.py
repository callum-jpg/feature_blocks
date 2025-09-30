"""Worker plugin to cache Zarr stores and avoid repeated open/close operations."""

import zarr
from distributed.diagnostics.plugin import WorkerPlugin
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


class ZarrStorePlugin(WorkerPlugin):
    """
    Dask worker plugin that opens and caches zarr stores once per worker.

    This avoids the overhead of opening zarr stores on every task,
    which can be a significant bottleneck when processing thousands of regions.

    Supports both regular Zarr and OME-Zarr formats.
    """

    def __init__(self, input_zarr_path, output_zarr_path, mask_store_path=None):
        """
        Initialize the plugin with paths to zarr stores.

        Args:
            input_zarr_path: Path to input zarr store
            output_zarr_path: Path to output zarr store (OME-Zarr format)
            mask_store_path: Optional path to mask zarr store (for CellProfiler)
        """
        self.input_zarr_path = input_zarr_path
        self.output_zarr_path = output_zarr_path
        self.mask_store_path = mask_store_path

        # These will be set once per worker during setup
        self.input_store = None
        self.output_store = None
        self.mask_store = None

    def setup(self, worker):
        """
        Called once when the plugin is registered with a worker.
        Opens and caches the zarr stores.
        """
        # Open input store - try OME-Zarr first, fall back to regular zarr
        try:
            store = parse_url(self.input_zarr_path, mode="r")
            reader = Reader(store)
            nodes = list(reader())
            self.input_store = nodes[0].data[0]  # First node, first resolution level
        except (KeyError, AttributeError, IndexError, TypeError):
            # Not OME-Zarr format, open as regular zarr
            self.input_store = zarr.open(self.input_zarr_path, mode="r")

        # Open output store - try OME-Zarr (for 4D blocks), fall back to regular zarr (for 2D centroids)
        try:
            store = parse_url(self.output_zarr_path, mode="r+")
            reader = Reader(store)
            nodes = list(reader())
            self.output_store = nodes[0].data[0]  # First node, first resolution level
        except (KeyError, AttributeError, IndexError, TypeError):
            # Not OME-Zarr, open as regular zarr
            self.output_store = zarr.open(self.output_zarr_path, mode="r+")

        # Open mask store if provided (regular zarr format)
        if self.mask_store_path is not None:
            self.mask_store = zarr.open(self.mask_store_path, mode="r")

        # Store in worker's state for easy access
        worker.input_zarr = self.input_store
        worker.output_zarr = self.output_store
        if self.mask_store is not None:
            worker.mask_zarr = self.mask_store

    def teardown(self, worker):
        """
        Called when the worker is shut down.
        Clean up resources if needed.
        """
        # Zarr stores don't need explicit cleanup
        pass
