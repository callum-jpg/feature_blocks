import logging
from distributed import WorkerPlugin
import zarr

log = logging.getLogger(__name__)


class ZarrHandlePlugin(WorkerPlugin):
    """Worker plugin that maintains persistent zarr store handles.

    This plugin opens zarr stores once when the worker starts and keeps them
    open for the duration of the worker's lifetime. This eliminates the overhead
    of opening/closing stores on every task.

    The handles are stored in the worker's data dictionary and can be accessed
    by tasks running on that worker.
    """

    def __init__(self, input_path=None, output_path=None, mask_store_path=None):
        """Initialize the plugin with paths to zarr stores.

        Args:
            input_path: Path to input zarr store (optional)
            output_path: Path to output zarr store (optional)
            mask_store_path: Path to mask zarr store (optional, for CellProfiler)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.mask_store_path = mask_store_path

    def setup(self, worker):
        """Open zarr stores once per worker on startup.

        Args:
            worker: The Dask worker instance
        """
        from ome_zarr.io import parse_url

        try:
            # Open input zarr store if path provided
            if self.input_path:
                in_store = parse_url(self.input_path, mode="r").store
                input_root = zarr.open_group(store=in_store, mode="r")
                worker.data['zarr_input'] = input_root["0"]
                log.debug(f"Worker {worker.id}: Opened input zarr at {self.input_path}")

            # Open output zarr store if path provided
            if self.output_path:
                out_store = parse_url(self.output_path, mode="r+").store
                output_root = zarr.open_group(store=out_store, mode="r+")
                worker.data['zarr_output'] = output_root["0"]
                log.debug(f"Worker {worker.id}: Opened output zarr at {self.output_path}")

            # Open mask store if path provided (for CellProfiler method)
            if self.mask_store_path:
                mask_store = zarr.open(self.mask_store_path, mode="r")
                worker.data['zarr_mask'] = mask_store
                log.debug(f"Worker {worker.id}: Opened mask zarr at {self.mask_store_path}")

        except Exception as e:
            log.error(f"Worker {worker.id}: Failed to open zarr stores: {e}")
            raise

    def teardown(self, worker):
        """Clean up zarr handles when worker shuts down.

        Args:
            worker: The Dask worker instance
        """
        # Remove references to allow garbage collection
        if 'zarr_input' in worker.data:
            del worker.data['zarr_input']
            log.debug(f"Worker {worker.id}: Closed input zarr handle")

        if 'zarr_output' in worker.data:
            del worker.data['zarr_output']
            log.debug(f"Worker {worker.id}: Closed output zarr handle")

        if 'zarr_mask' in worker.data:
            del worker.data['zarr_mask']
            log.debug(f"Worker {worker.id}: Closed mask zarr handle")
