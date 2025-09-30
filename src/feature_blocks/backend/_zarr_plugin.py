"""Worker plugin to cache Zarr stores and avoid repeated open/close operations."""

import zarr
from distributed.diagnostics.plugin import WorkerPlugin
from ome_zarr.io import parse_url


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
            store = parse_url(self.input_zarr_path, mode="r").store
            root = zarr.group(store=store)
            self.input_store = root["0"]  # OME-Zarr full-resolution data
        except (KeyError, AttributeError):
            # Not OME-Zarr format, open as regular zarr
            self.input_store = zarr.open(self.input_zarr_path, mode='r')

        # Open output store - always OME-Zarr format
        store = parse_url(self.output_zarr_path, mode="r+").store
        root = zarr.group(store=store)
        self.output_store = root["0"]  # OME-Zarr full-resolution data

        # Open mask store if provided (regular zarr format)
        if self.mask_store_path is not None:
            self.mask_store = zarr.open(self.mask_store_path, mode='r')

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