import os
import dask
import dask.delayed
from dask.distributed import Client, progress
import logging
from typing import Callable
from dask.delayed import Delayed

log = logging.getLogger(__name__)

def get_n_workers():
    # Check if it's a SLURM job
    slurm_cpus = os.getenv("SLURM_CPUS_ON_NODE")

    if slurm_cpus is not None:
        return int(slurm_cpus)
    else:
        return os.cpu_count()


def run_dask_backend(functions: list[Callable]):
    try:
        from dask_cuda import LocalCUDACluster
        n_workers = LocalCUDACluster()
        log.info(f"Using CUDA cluster")
    except:
        n_workers = get_n_workers()

        if len(functions) < n_workers:
            n_workers = len(functions)

        log.info(f"Using Dask backend with {n_workers} workers.")

    with Client(n_workers) as client:
        if not all([isinstance(fn, Delayed) for fn in functions]):
            functions = [dask.delayed(fn)() for fn in functions]

        futures = client.compute(functions)
        progress(futures)
        results = client.gather(futures)
        return results