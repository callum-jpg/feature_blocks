import numpy 
import matplotlib.pyplot as plt
import spatialdata_plot
from feature_blocks.utility import get_spatial_element

def assign_cell_cluster_block(
    sdata,
    sdata_image_key,
    cluster_blocks,
    extra_scale_factor: int | float = 1/0.2125 # Xenium micron/px scaling
):

    original_image = get_spatial_element(getattr(sdata, "images"), sdata_image_key, as_spatial_image=True)
    original_shape = (original_image.y.shape[0], original_image.x.shape[0])

    adata = get_spatial_element(getattr(sdata, "tables"), "table")
    
    centroids = adata.obsm["spatial"]

    centroids = scale_coords(
        xy = centroids,
        original_shape = original_shape,
        new_shape = cluster_blocks.squeeze().shape # TODO: make cluster_blocks have a standardised size
        )
    
    cluster_block_id = cluster_blocks[centroids[:, 1], centroids[:, 0]]

    adata.obs["cluster_block_id"] = cluster_block_id

def scale_coords(xy: numpy.ndarray, original_shape, new_shape, extra_scale_factor = 1/0.2125) -> numpy.ndarray:
    if extra_scale_factor is None:
        extra_scale_factor = 1

    H_orig, W_orig = original_shape
    H_new, W_new = new_shape

    x_scale = W_new / W_orig
    y_scale = H_new / H_orig

    xy = numpy.multiply(
        xy,
        [x_scale * extra_scale_factor, y_scale * extra_scale_factor]
    ).astype(int)

    return xy

def get_feature_block_cluster(
    centroids,
    clustered_blocks,
    original_image_shape,
):
    centroids = scale_coords(
        centroids,
        original_shape=original_image_shape,
        new_shape=clustered_blocks.squeeze().shape,   
    )

    t = np.zeros_like(clustered_blocks.squeeze())

    t[centroids[:, 1], centroids[:, 0]] = 1

    cluster_block_id = clustered_blocks[centroids[:, 1], centroids[:, 0]]

    adata.obs["cluster_block_id"] = cluster_block_id