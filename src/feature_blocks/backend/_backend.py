import os
import dask
import dask.delayed
from dask.distributed import Client, progress
import logging
from typing import Callable
from dask.delayed import Delayed

log = logging.getLogger(__name__)

def run_dask_backend(functions: list[Callable]):
    # n_cpus: int = os.cpu_count()
    n_cpus = 6
    
    if len(functions) < n_cpus:
        n_workers = len(functions)
    else:
        n_workers = n_cpus

    log.info(f"Using Dask backend with {n_workers} n_workers.")

    with Client(n_workers=n_workers) as client:
        ram_per_worker = client.cluster.workers[0].memory_manager.memory_limit

        if ram_per_worker < 4 * 1024**3:
            log.warning(
                f"Each worker has less than 4GB of RAM ({ram_per_worker / 1024**3:.2f}GB). Consider increasing available memory. "
            )

        if not all([isinstance(fn, Delayed) for fn in functions]):
            functions = [dask.delayed(fn)() for fn in functions]

        # print("Are they all delayed?", {type(f) for f in functions})
        # futures = dask.persist(*functions)
        # print("Are futures all delayed?", {type(f) for f in futures})
        # progress(futures)
        # return client.gather(client.compute(list(futures)))

        futures = client.compute(functions)
        print("Are futures all delayed?", {type(f) for f in futures})
        progress(futures)
        results = client.gather(futures)
        print("Is results all delayed?", {type(f) for f in results})
        return results

        # futures = [client.submit(fn) for fn in functions]
        # progress(futures)
        # results = client.gather(futures)
        # return results