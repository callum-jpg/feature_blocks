import logging
import os
from typing import Callable

from dask.distributed import Client, LocalCluster, progress, performance_report
from distributed.utils import silence_logging_cmgr

import dask

log = logging.getLogger(__name__)


def in_slurm():
    return "slurm" in os.environ["HOSTNAME"]


def get_n_workers():
    # Check if it's a SLURM job
    slurm_cpus = os.getenv("SLURM_CPUS_ON_NODE")

    if slurm_cpus is not None:
        return int(slurm_cpus)
    else:
        return os.cpu_count()


def run_dask_backend(functions: list[Callable], visualise_graph: bool = False):
    # if not in_slurm():  # Temp block
    #     from dask_jobqueue import SLURMCluster

    #     log.info("Using SLURM cluster")
    #     print("Using SLURM cluster")

    #     # import tomllib
    #     # config = tomllib.load(open("/nfs/research/uhlmann/callum/zarr_backed_dask_compute/zarr_back/config/slurm_config.toml", "rb"))
    #     # dask.config.update(config)

    #     cluster = SLURMCluster(
    #         # queue='research',
    #         # account="callum",
    #         cores=4,
    #         processes=1,
    #         memory="12GB",
    #         walltime="01:00:00",
    #         # python="singularity run /nfs/research/uhlmann/callum/dockerfiles/histology_features/histology_features.simg python"
    #     )
    #     cluster.scale(20)
    # else:
    try:
        from dask_cuda import LocalCUDACluster

        cluster = LocalCUDACluster()
        log.info("Using CUDA cluster")
    except:
        n_workers = get_n_workers()
        # n_workers = 6
        cluster = LocalCluster(n_workers=n_workers)
        log.info("Using CPU cluster")

        if len(functions) < n_workers:
            n_workers = len(functions)

        log.info(f"Using {n_workers} n_workers")

    if visualise_graph:
        dask.visualize(*functions, filename="dask-task-graph", format="svg", optimize_graph=True, color="order")

    client = Client(cluster, asynchronous=False)

    with performance_report(filename = "performance_report.html"):
        futures = dask.compute(functions)
        progress(futures, notebook=False)

    # Silence cluster shutdown
    with silence_logging_cmgr(logging.CRITICAL):
        client.cancel(futures)
        client.shutdown()
