import logging
import os
from typing import Callable
import traceback

import dask
from dask.distributed import Client, progress, wait, performance_report
from distributed.utils import silence_logging_cmgr
from ._zarr_plugin import ZarrHandlePlugin

log = logging.getLogger(__name__)


def in_slurm():
    return "SLURM_JOB_ID" in os.environ


def in_lsf():
    return "LSB_JOBID" in os.environ


def get_n_workers():
    # Check if it's a SLURM job
    slurm_cpus = os.getenv("SLURM_CPUS_ON_NODE")
    # Check if it's an LSF job
    lsf_cpus = os.getenv("LSB_SLOTS")

    if slurm_cpus is not None:
        return int(slurm_cpus)
    elif lsf_cpus is not None:
        return int(lsf_cpus)
    else:
        return os.cpu_count()


def run_dask_backend(
    function: Callable,
    regions: list,
    visualise_graph: bool = False,
    n_workers: int | None = None,
    python_path: str = "python",
    memory: str = "16GB",
    batch_size: int = 1,
    model_identifier: str | None = None,
    input_zarr_path: str | None = None,
    output_zarr_path: str | None = None,
    mask_store_path: str | None = None,
    function_kwargs: dict | None = None,
):
    if in_slurm():
        from dask_jobqueue import SLURMCluster

        if n_workers is None:
            n_workers = 1
            log.info(f"n_workers is {n_workers}. Defaulting to using only 1 worker.")

        log.info(f"Using SLURM cluster with {n_workers} n_workers")

        cluster = SLURMCluster(
            n_workers=n_workers,
            cores=2,
            memory=memory,
            walltime="08:00:00",
            log_directory="logs",
            python=python_path,
        )
    elif in_lsf():
        from dask_jobqueue import LSFCluster

        if n_workers is None:
            n_workers = 1
            log.info(f"n_workers is {n_workers}. Defaulting to using only 1 worker.")

        log.info(f"Using LSF cluster with {n_workers} n_workers")

        cluster = LSFCluster(
            n_workers=n_workers,
            cores=1,
            memory=memory,
            walltime="08:00:00",
            log_directory="logs",
            python=python_path,
        )
    else:
        try:
            from dask_cuda import LocalCUDACluster

            cluster = LocalCUDACluster()
            log.info("Using CUDA cluster")
        except:
            n_workers = get_n_workers()
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=1,
                memory_limit=memory
            )
            log.info("Using CPU cluster")

            if len(regions) < n_workers:
                n_workers = len(regions)

            log.info(f"Using {n_workers} n_workers with {memory} memory per worker")

    # Note: visualise_graph is not applicable with client.map()
    # The task graph is much simpler now (N independent tasks instead of 3N chained tasks)
    if visualise_graph:
        log.warning("visualise_graph is not supported with client.map() workflow")

    client = Client(cluster, asynchronous=False)

    # Register ZarrHandlePlugin to keep zarr stores open for worker lifetime
    # This eliminates the overhead of opening/closing stores on every task
    if input_zarr_path or output_zarr_path or mask_store_path:
        log.info("Registering ZarrHandlePlugin for persistent zarr handles...")
        plugin = ZarrHandlePlugin(
            input_path=input_zarr_path,
            output_path=output_zarr_path,
            mask_store_path=mask_store_path
        )
        client.register_plugin(plugin, name='zarr-handle-plugin')
        log.info("ZarrHandlePlugin registered successfully")

    # Pre-load model on all workers to avoid redundant loading
    # if model_identifier is not None:
    #     log.info(f"Pre-loading model '{model_identifier}' on all workers...")

    #     def warmup_model(model_name):
    #         """Load model into worker's local cache."""
    #         # This will initialize the model in the worker's _model_cache
    #         # We use a dummy input just to trigger model loading
    #         import numpy

    #         from feature_blocks.task import infer

    #         dummy_input = numpy.zeros((1, 1, 1, 1), dtype=numpy.float32)
    #         try:
    #             infer(dummy_input, model_name)
    #         except Exception:
    #             # Some models may fail with dummy input, but the model
    #             # will still be loaded into the cache
    #             pass
    #         return f"Model {model_name} loaded"

    #     # Run warmup on all workers (use actual worker count from cluster)
    #     actual_workers = len(client.scheduler_info()["workers"])
    #     warmup_futures = client.map(warmup_model, [model_identifier] * actual_workers)
    #     client.gather(warmup_futures)
    #     log.info("Model pre-loading complete")

    if function_kwargs is None:
        function_kwargs = {}

    log.info(f"Mapping {function.__name__} over {len(regions)} regions with batch_size={batch_size}...")

    with performance_report(filename="performance_report.html"):
        futures = client.map(function, regions, pure=True, batch_size=batch_size, **function_kwargs)
        progress(futures, notebook=False)

    # Just ensure all tasks are finished (and handle errors)
    wait(futures, timeout=600 * len(regions))

    failed = [f for f in futures if f.status == "error"]
    if failed:
        print(f"{len(failed)} tasks failed.")

    # Silence cluster shutdown
    with silence_logging_cmgr(logging.CRITICAL):
        client.cancel(futures)
        client.shutdown()
