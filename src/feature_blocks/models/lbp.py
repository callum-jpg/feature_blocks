import dask
import dask.array
import numpy
import skimage

def lbp_features(
    image: numpy.ndarray, radius: typing.Union[int, list[int]]
) -> numpy.ndarray:
    """
    Extract Local Binary Pattern (LBP) features from an image.

    Parameters:
        image (numpy.ndarray): Grayscale image to process.
        radius (int or list of int): Radius or list of radii to define the LBP footprint.
                                      Multiple radii will compute features at multiple scales.

    Returns:
        numpy.ndarray: Concatenated normalized LBP histograms for each radius.
    """
    if isinstance(radius, int):
        radius = [radius]

    features = []

    for rad in radius:
        n_points = int(numpy.pi * rad**2)
        lbp = skimage.feature.local_binary_pattern(image, P=n_points, R=rad, method="uniform")

        hist, _ = numpy.histogram(
            lbp.ravel(),
            bins=numpy.arange(0, n_points + 3),
            range=(0, n_points + 2)
        )
        hist = hist.astype(float)
        hist /= hist.sum() + 1e-7  # Normalize histogram, + eps
        features.append(hist)

    return numpy.concatenate(features)


class LBP(nn.Module):
    """
    An example feature extraction class
    """

    def __init__(
        self,
        radius = [5, 10],
    ):
        super().__init__()

        # Define how many features the model returns (ie. the hidden dim)
        # This allows up to build the zarr store before any inference has been
        # run, which therefore allows for on the fly writing results to disk.
        self.n_features = 42
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

    

    def forward(self, x):
        features = random_torch_tensor(self.output_shape)

        return features.cpu().numpy()







class LBP(nn.Module):
    """
    An example feature extraction class
    """

    def __init__(self):
        super().__init__()

        # Define how many features the model returns (ie. the hidden dim)
        # This allows up to build the zarr store before any inference has been
        # run, which therefore allows for on the fly writing results to disk.
        self.n_features = 42
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

    def forward(self, x):
        features = random_torch_tensor(self.output_shape)

        return features.cpu().numpy()
