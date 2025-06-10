import logging
import os
from typing import Callable

from dask.distributed import Client, LocalCluster, progress, performance_report, wait
from distributed.utils import silence_logging_cmgr

import dask

log = logging.getLogger(__name__)


def in_slurm():
    return "SLURM_JOB_ID" in os.environ


def get_n_workers():
    # Check if it's a SLURM job
    slurm_cpus = os.getenv("SLURM_CPUS_ON_NODE")

    if slurm_cpus is not None:
        return int(slurm_cpus)
    else:
        return os.cpu_count()


def run_dask_backend(functions: list[Callable], visualise_graph: bool = False):
    if in_slurm():  # Temp block
        from dask_jobqueue import SLURMCluster

        log.info("Using SLURM cluster")
        print("Using SLURM cluster")

        cluster = SLURMCluster(
            # n_workers=200,
            n_workers=400,
            cores=1,
            memory="16GB",
            walltime="03:00:00",
            log_directory="logs",
            python="singularity exec --env PATH=/homes/callum/.local/lib/python3.11/site-packages:$PATH /nfs/research/uhlmann/callum/dockerfiles/histology_features/histology_features.simg python"
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
        dask.visualize(*functions, filename="dask-task-graph", format="svg", optimize_graph=True, color="order")

    client = Client(cluster, asynchronous=False)

    with performance_report(filename = "performance_report.html"):
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
