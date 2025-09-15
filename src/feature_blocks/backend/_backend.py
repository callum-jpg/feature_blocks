import logging
import os
from typing import Callable

import dask
from dask.distributed import (
    Client,
    LocalCluster,
    as_completed,
    performance_report,
    progress,
    wait,
)
from distributed.utils import silence_logging_cmgr

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


def run_dask_backend(functions: list[Callable], visualise_graph: bool = False, n_workers: int | None = None, python_path: str = "python", memory: str = "16GB"):
    if in_slurm():
        from dask_jobqueue import SLURMCluster

        if n_workers is None:
            n_workers = 1
            log.info(f"n_workers is {n_workers}. Defaulting to using only 1 worker.")

        log.info(f"Using SLURM cluster with {n_workers} n_workers")

        cluster = SLURMCluster(
            n_workers=n_workers,
            cores=1,
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
            cluster = LocalCluster(n_workers=n_workers)
            log.info("Using CPU cluster")

            if len(functions) < n_workers:
                n_workers = len(functions)

            log.info(f"Using {n_workers} n_workers")

    if visualise_graph:
        dask.visualize(
            *functions,
            filename="dask-task-graph",
            format="svg",
            optimize_graph=True,
            color="order",
        )

    client = Client(cluster, asynchronous=False)

    with performance_report(filename="performance_report.html"):
        futures = client.compute(functions)
        progress(futures, notebook=False)

    # Process results with timeout
    completed_count = 0
    failed_indices = []

    for i, future in enumerate(as_completed(futures, timeout=600 * len(functions))):
        try:
            # Get result with individual timeout
            result = future.result(timeout=600)

            completed_count += 1
            if completed_count % 100 == 0:  # Progress update every 100 functions
                print(f"Completed {completed_count}/{len(functions)} tasks")

        except Exception as e:
            print(f"Task {i} failed: {e}")
            failed_indices.append(i)

    print(f"Completed: {completed_count}, Failed: {len(failed_indices)}")

    # Silence cluster shutdown
    with silence_logging_cmgr(logging.CRITICAL):
        client.cancel(futures)
        client.shutdown()
