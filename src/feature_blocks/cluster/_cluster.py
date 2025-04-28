import typing

import anndata
import matplotlib.pyplot as plt
import numpy
import scanpy
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA

from feature_blocks.utility import make_cmap

import einops



def cluster_blocks(
    feature_blocks: numpy.array,
    cluster_method: typing.Literal["kmeans", "hdbscan", "leiden"],
    return_adata: bool = False,
    **kwargs,
):
    """
    cluster the output of feature_map_blocks, which is
    an array with shape (x_dims, y_dims, feature_vector)
    """

    # Define the linearised block shapes (n_features, YXZ)
    linear_shape = (
        feature_blocks.shape[0], feature_blocks.shape[1] * feature_blocks.shape[2] * feature_blocks.shape[3]
    )

    # Reshape to array of (num_features_per_block, num_blocks)
    reshaped_features = einops.rearrange(feature_blocks, "C Z H W -> C (Z H W)")

    # pca_proj_ch = 5

    # pca = PCA(pca_proj_ch)

    # reshaped_features = nan_drop_fn(
    #     reshaped_features.T,
    #     pca.fit_transform,
    #     pca_proj_ch,
    # ).T


    reshaped_features = reshaped_features / numpy.linalg.norm(reshaped_features, axis=0, keepdims=True)

    # Find where feature_blocks is NaN, which represents background
    mask = numpy.any(numpy.isnan(reshaped_features), axis=0)

    # Invert the mask to get foreground only indices
    idx_to_cluster = numpy.where(~mask)[0]

    if cluster_method.casefold() == "kmeans":
        cluster_id = KMeans(**kwargs).fit_predict(reshaped_features[:, idx_to_cluster])
    elif cluster_method.casefold() == "hdbscan":
        cluster_id = HDBSCAN(**kwargs).fit_predict(reshaped_features[:, idx_to_cluster])
    elif cluster_method.casefold() == "leiden":
        # Tranpose the reshaped_features into (block, feature) shape
        adata = anndata.AnnData(reshaped_features[:, idx_to_cluster].T)
        try:
            # Use GPU, if available
            import rapids_singlecell

            rapids_singlecell.pp.neighbors(adata, use_rep="X")
            print("Using GPU-accelerated clustering")
            rapids_singlecell.tl.leiden(adata, **kwargs)
            cluster_id = adata.obs.leiden.to_numpy().astype(int)
            rapids_singlecell.tl.umap(adata)
        except:
            print("No GPU detected. Using CPU clustering")
            # We use pynndescent to find KNN since Scanpy's
            # defaults (which also uses automatic brute forcing for finding KNN
            # in small datasets for speed) to prevent indexing errors.
            scanpy.pp.neighbors(adata, use_rep="X", transformer="pynndescent")
            scanpy.tl.leiden(adata, **kwargs)
            cluster_id = adata.obs.leiden.to_numpy().astype(int)
            # Create and plot UMAP. This generates the unstructured "leiden_colors"
            # in adata, which can be used to generate a cmap. This allows for UMAP
            # and feature block images to have spatially mapped clusters for
            # visualisation Downside: this returns an inline UMAP plot, though this
            # is sort of handy to always have.
            scanpy.tl.umap(adata)
        if not return_adata:
            num_clusters = adata.obs.leiden.unique().categories.shape[0]
            if num_clusters > 100:
                raise Warning(
                    f"Number of leiden clusters ({num_clusters}) too high to plot. Consider reducing leiden resolution."
                )
            with plt.rc_context({"figure.figsize": (5, 5), "figure.dpi": (300)}):
                scanpy.pl.umap(adata, color=["leiden"])

            cmap = make_cmap(adata.uns["leiden_colors"])

    else:
        raise NotImplementedError

    output_features = numpy.zeros((1, linear_shape[1]))
    output_features[:, idx_to_cluster] = cluster_id

    inverted_idx = numpy.setdiff1d(range(output_features.shape[1]), idx_to_cluster)
    output_features[:, inverted_idx] = numpy.nan

    output_features = einops.rearrange(output_features, "C (Z H W) -> C Z H W", Z=feature_blocks.shape[1], H=feature_blocks.shape[2], W=feature_blocks.shape[3])

    if return_adata:
        return output_features, adata
    elif cluster_method.casefold() == "leiden" and return_adata == False:
        return output_features, cmap
    else:
        return output_features

def nan_drop_fn(array, fn, output_feature_size):
    """Apply a function to an array that has the form 
    (n_samples, n_features). If n_samples contains NaN values, these
    will not be passed to the function.
    """
    # Find where a row si composed of only nan
    mask = numpy.any(numpy.isnan(array), axis=1)

    # Get the indices for where there is not a nan
    valid_mask = numpy.where(~mask)[0]

    # Subset
    no_nan_array = array[valid_mask]

    # Apply function to non-nan subset
    output = fn(no_nan_array)

    # Create an array of NaN values based on n_samples and output_feature_size
    output_features = numpy.full((array.shape[0], output_feature_size), numpy.nan)
    # Fill in the features for the non NaN values. 
    output_features[valid_mask] = output

    return output_features


def cluster_batch_blocks(
    feature_blocks_dict: dict,
    cluster_method: typing.Literal["kmeans", "hdbscan", "leiden"],
    masked_value=None,
    **kwargs,
):
    # Reshape dict values into a single array
    feature_blocks = numpy.array([i for i in feature_blocks_dict.values()])

    if masked_value is not None:
        if numpy.isnan(masked_value):
            # Get the mask from where blocks are nan'd
            mask = numpy.where(numpy.isnan(feature_blocks), False, True)
            mask = mask[..., 0]
        elif masked_value == 0:
            mask = numpy.where(feature_blocks == masked_value, False, True)
            mask = mask[..., 0]
        else:
            raise ValueError
    else:
        raise ValueError

    return cluster_blocks(
        feature_blocks, cluster_method, mask=mask, volumetric=True, **kwargs
    )


def get_cluster_mask_consensus(
    clusters1: numpy.ndarray, clusters2: numpy.ndarray
) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    """In histology_features, we set background pixels to
    NaN to prevent these pixels from influencing clustering.

    However, the mask that is used to define these background
    pixels is often resized to match the resolution of the
    feature block image.

    As a result, the resized mask may have resulting NaN values
    in one image where they are not in the other. This happens at
    the **border** of the foreground object.

    This function compares the NaN values of two cluster images
    and finds a better mask between the two: where there is a consensus
    of NaN values. If the NaN occurs in only one cluster
    image, the other image also sets the same element as NaN.

    Alternative solution to this problem: a binary mask dilation and
    erosion earlier in the pipeline.
    """

    assert clusters1.shape == clusters2.shape, "Cluster images should be equal"

    # If NaN is in both images, that element is True. Otherwise, False.
    shared_mask = numpy.logical_or(numpy.isnan(clusters1), numpy.isnan(clusters2))

    # Set all individual and join NaNs to NaN in both images
    clusters1_fixed = numpy.where(~shared_mask, clusters1, numpy.nan)
    clusters2_fixed = numpy.where(~shared_mask, clusters2, numpy.nan)

    assert (
        numpy.isnan(clusters1_fixed).shape == numpy.isnan(clusters2_fixed).shape
    ), "NaN's not equal"

    return clusters1_fixed, clusters2_fixed
