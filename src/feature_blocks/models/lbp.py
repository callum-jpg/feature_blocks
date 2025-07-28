import typing

import numpy
import skimage
from torch import nn


def lbp_features(
    image: numpy.ndarray,
    radius: typing.Union[int, list[int]] = [1, 2, 3],
    method: str = "uniform",
    normalize: str = "l2",
) -> numpy.ndarray:
    """
    Extract Local Binary Pattern (LBP) features from an image for clustering.

    Parameters:
        image (numpy.ndarray): Grayscale image to process (2D array).
        radius (int or list of int): Radius or radii for LBP computation.
        method (str): LBP method ('uniform', 'default', 'ror', 'var').
        normalize (str): Normalization method ('l1', 'l2', 'none').

    Returns:
        numpy.ndarray: Concatenated LBP feature vector.

    Raises:
        ValueError: If image is not 2D or parameters are invalid.
    """
    # Input validation
    if not isinstance(image, numpy.ndarray) or image.ndim != 2:
        raise ValueError("Image must be a 2D numpy array")

    if isinstance(radius, int):
        radius = [radius]

    if any(r <= 0 for r in radius):
        raise ValueError("All radius values must be positive")

    n_points = [8 * r for r in radius]

    features = []

    for rad, pts in zip(radius, n_points):
        # Compute LBP
        lbp = skimage.feature.local_binary_pattern(image, P=pts, R=rad, method=method)

        # Determine number of bins based on method
        if method == "uniform":
            n_bins = pts + 2  # uniform patterns + non-uniform
        else:
            n_bins = 2**pts

        # Compute histogram
        hist, _ = numpy.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        # Convert to float and normalize
        hist = hist.astype(float)
        if normalize == "l1":
            hist = hist / (hist.sum() + 1e-7)
        elif normalize == "l2":
            hist = hist / (numpy.linalg.norm(hist) + 1e-7)

        features.append(hist)

    return numpy.concatenate(features)


class LBP(nn.Module):
    """
    An example feature extraction class
    """

    def __init__(
        self,
        radius: typing.Union[int, list[int]] = [1, 2, 3],
    ):
        super().__init__()

        self.radius = radius

        # Define how many features the model returns (ie. the hidden dim)
        # This allows up to build the zarr store before any inference has been
        # run, which therefore allows for on the fly writing results to disk.
        self.n_features = self._get_n_features(self.radius)
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

    def forward(self, x):
        features = lbp_features(x, radius=self.radius)

        return features

    def _get_n_features(self, radius, method: str = "uniform"):
        """
        Calculate the size of the feature vector that lbp_features will return.

        Parameters:
            radius (int or list of int): Radius or radii for LBP computation.s.
            method (str): LBP method ('uniform', 'default', 'ror', 'var').

        Returns:
            int: Total size of the feature vector.
        """
        if isinstance(radius, int):
            radius = [radius]

        n_points = [8 * r for r in radius]

        total_size = 0

        for rad, pts in zip(radius, n_points):
            # Calculate histogram size based on method
            if method == "uniform":
                hist_size = pts + 2  # uniform patterns + non-uniform
            elif method in ["default", "ror"]:
                hist_size = 2**pts
            elif method == "var":
                hist_size = 1  # variance method returns single value per pixel, but we histogram it
                # For variance method, we'd need to know the image to determine bins
                # This is a limitation - variance method is less predictable
                raise ValueError(
                    "Cannot predict feature size for 'var' method without image data"
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            total_size += hist_size

        return total_size
