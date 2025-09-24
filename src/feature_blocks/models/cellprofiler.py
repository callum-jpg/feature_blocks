import logging

import numpy
import pandas
import skimage.color
import torch
from torch import nn

log = logging.getLogger(__name__)

try:
    from cp_measure.bulk import get_core_measurements
except ImportError:
    log.warning("cp_measure not available. Install with: pip install cp-measure")

try:
    import pycytominer
except ImportError:
    raise ImportError(
        "pycytominer is required for CellProfiler.postprocess_features(). "
        "Install with: pip install pycytominer"
    )


class CellProfiler(nn.Module):
    """
    CellProfiler feature extraction for segmented regions.

    This class extracts morphological and intensity features from image regions
    defined by segmentation masks using the cp_measure library. It's designed
    to work with the centroid-based block method where segmentations define
    the regions of interest.

    Features extracted include:
    - Radial distribution (12 features)
    - Radial Zernike moments (60 features)
    - Intensity statistics (21 features)
    - Size and shape (78 features)
    - Zernike moments (30 features)
    - Ferret diameter (2 features)
    - Texture features (52 features)
    - Granularity (16 features)

    Total: 271 features per object
    """

    def __init__(self):
        super().__init__()

        # Initialize cp_measure functions
        self.measurements = get_core_measurements()

        # Model attributes required by feature_blocks
        self.n_features = 271  # Total features across all measurement types
        self.output_shape = (self.n_features, 1, 1, 1)  # (C, Z, H, W)

        # Store feature names for reference
        self._feature_names = None
        self._initialize_feature_names()

    def _initialize_feature_names(self):
        """Initialize feature names by running a dummy extraction."""
        try:
            # Create minimal test data to get feature names
            dummy_mask = numpy.array([[0, 1], [0, 1]], dtype=numpy.int32)
            dummy_image = numpy.array([[0.1, 0.5], [0.1, 0.5]], dtype=numpy.float32)

            feature_names = []
            for measurement_name, measurement_func in self.measurements.items():
                try:
                    result = measurement_func(dummy_mask, dummy_image)
                    feature_names.extend(result.keys())
                except Exception as e:
                    log.warning(f"Failed to initialize {measurement_name}: {e}")

            self._feature_names = feature_names
            log.info(f"Initialized {len(feature_names)} CellProfiler features")

        except Exception as e:
            log.warning(f"Failed to initialize feature names: {e}")
            self._feature_names = [f"cp_feature_{i}" for i in range(self.n_features)]

    def _convert_to_grayscale(self, image_channels):
        """
        Convert multi-channel image to grayscale.

        Args:
            image_channels: numpy array of shape (C, Z, H, W)

        Returns:
            numpy array of shape (H, W) - grayscale image
        """
        # Take first Z slice
        image = image_channels[:, 0, :, :]  # (C, H, W)

        if image.shape[0] == 3:
            # Use ColorToGray (rgb2gray) for 3-channel images
            image = skimage.color.rgb2gray(image, channel_axis=0)
        else:
            # Use mean for other multi-channel images
            image = numpy.mean(image, axis=0)

        return image

    def forward(self, x):
        """
        Extract CellProfiler features from a segmented image block.

        Args:
            x: Input tensor of shape (C, Z, H, W) where:
               - C channels should include both image data (0 to -1th) and segmentation mask (-1th)
               - The last channel is expected to be the segmentation mask
               - Other channels are the image data

        Returns:
            numpy.ndarray: Feature vector of shape (n_features, 1, 1, 1)
        """
        if not isinstance(x, (torch.Tensor, numpy.ndarray)):
            raise ValueError(
                f"Input must be torch.Tensor or numpy.ndarray, got {type(x)}"
            )

        # Convert to numpy if tensor
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        # Validate input shape
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (C, Z, H, W), got {x.ndim}D")

        if x.shape[0] < 2:
            raise ValueError("Need at least 2 channels: image data + segmentation mask")

        # Extract image and mask
        # Assume last channel is segmentation mask, others are image channels
        image_channels = x[:-1]  # All but last channel
        mask = x[-1, 0]  # Last channel, first Z slice (segmentation mask)

        # Convert to grayscale
        if image_channels.shape[0] > 1:
            image = self._convert_to_grayscale(image_channels)
        else:
            image = image_channels[0, 0]  # First channel, first Z slice

        # Ensure image is float in range [0, 1]
        if image.dtype != numpy.float32:
            image = image.astype(numpy.float32)

        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image / image.max()

        # Ensure mask has integer labels
        mask = mask.astype(numpy.int32)

        # Check if there are any objects in the mask
        unique_labels = numpy.unique(mask)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background (0)

        if len(unique_labels) == 0:
            # No objects found, return NaN features
            log.warning("No objects found in mask, returning NaN features")
            features = numpy.full(
                (self.n_features, 1, 1, 1), numpy.nan, dtype=numpy.float32
            )
            return features

        # Extract features for all objects
        all_features = []

        for measurement_name, measurement_func in self.measurements.items():
            try:
                result = measurement_func(mask, image)

                # Convert result dict to feature arrays
                for feature_name, feature_values in result.items():
                    assert len(feature_values) == 1, "Multiple masks processed, expected one."
                    if isinstance(feature_values, numpy.ndarray):
                        # Take mean across all objects for this feature
                        mean_value = numpy.nanmean(feature_values)
                        all_features.append(mean_value)
                    else:
                        # Scalar value
                        all_features.append(float(feature_values))

            except Exception as e:
                log.warning(f"Failed to compute {measurement_name}: {e}")
                # Add NaN features for this measurement type
                if measurement_name == "radial_distribution":
                    all_features.extend([numpy.nan] * 12)
                elif measurement_name == "radial_zernikes":
                    all_features.extend([numpy.nan] * 60)
                elif measurement_name == "intensity":
                    all_features.extend([numpy.nan] * 21)
                elif measurement_name == "sizeshape":
                    all_features.extend([numpy.nan] * 78)
                elif measurement_name == "zernike":
                    all_features.extend([numpy.nan] * 30)
                elif measurement_name == "ferret":
                    all_features.extend([numpy.nan] * 2)
                elif measurement_name == "texture":
                    all_features.extend([numpy.nan] * 52)
                elif measurement_name == "granularity":
                    all_features.extend([numpy.nan] * 16)

        # Ensure we have the expected number of features
        if len(all_features) != self.n_features:
            log.warning(
                f"Expected {self.n_features} features, got {len(all_features)}. "
                "Padding or truncating to match expected size."
            )
            if len(all_features) < self.n_features:
                # Pad with NaN
                all_features.extend([numpy.nan] * (self.n_features - len(all_features)))
            else:
                # Truncate
                all_features = all_features[: self.n_features]

        # Convert to numpy array
        features = numpy.array(all_features, dtype=numpy.float32)

        features = features.reshape(self.n_features, 1, 1, 1)

        return features

    def postprocess(self, features):
        """Perform feature normalisation and feature selection
        of CellProfiler features"""

        cellprofiler_features = pandas.DataFrame(
            features, columns=self.feature_names
        )

        # Pycytominer requires a metadata column
        cellprofiler_features["meta"] = None

        cellprofiler_features = pycytominer.normalize(
            profiles=cellprofiler_features,
            method="standardize",
            features=self.feature_names,
            meta_features=["meta"],
        )

        cellprofiler_features = pycytominer.feature_select(
            profiles=cellprofiler_features,
            features=self.feature_names,
        )

        # Drop the meta column
        cellprofiler_features = cellprofiler_features.drop(["meta"], axis=1)

        return cellprofiler_features.to_numpy()

    def add_to_sdata(self, sdata, features, obsm_key: str = "cellprofiler_features", table_key: str = "table"):
        features = self.process(features)

        sdata[table_key].obsm[obsm_key] = features

        return sdata

    @property
    def feature_names(self):
        """Get list of feature names."""
        return self._feature_names.copy() if self._feature_names else None
