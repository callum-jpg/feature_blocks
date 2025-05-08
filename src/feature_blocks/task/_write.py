import gc

def write(zarr, data, region):
    zarr[tuple(region)] = data
    del data
    gc.collect()