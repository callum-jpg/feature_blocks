import tempfile
import os
import shutil
import numpy 
import zarr
import geopandas as gpd
from shapely.geometry import Polygon
import pytest
import toml

import sys
import os

from feature_blocks.features import extract
from feature_blocks.models.cellprofiler import CellProfiler
from feature_blocks.io import create_ome_zarr_output


class TestCellProfilerExtraction:
    """Comprehensive test for CellProfiler feature extraction using synthetic data"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def synthetic_image(self, temp_dir):
        """Create synthetic 3-channel RGB image with distinct objects"""
        # Create a 3-channel RGB image (C, Z, H, W)
        image_shape = (3, 1, 512, 512)
        image = numpy.zeros(image_shape, dtype=numpy.float32)

        # Add some background texture
        numpy.random.seed(42)
        background = numpy.random.normal(0.2, 0.05, image_shape[2:])
        background = numpy.clip(background, 0, 1)

        # Set background for all channels
        for c in range(3):
            image[c, 0, :, :] = background

        # Add distinct objects with different intensities
        # Object 1: Bright circular object (center: 150, 150, radius: 30)
        y1, x1 = numpy.ogrid[:512, :512]
        mask1 = ((x1 - 150) ** 2 + (y1 - 150) ** 2) <= 30 ** 2
        image[0, 0, mask1] = 0.9  # Red channel
        image[1, 0, mask1] = 0.3  # Green channel
        image[2, 0, mask1] = 0.1  # Blue channel

        # Object 2: Medium intensity elongated object (center: 350, 200)
        mask2 = ((x1 - 350) ** 2 / 40 ** 2 + (y1 - 200) ** 2 / 20 ** 2) <= 1
        image[0, 0, mask2] = 0.6  # Red channel
        image[1, 0, mask2] = 0.8  # Green channel
        image[2, 0, mask2] = 0.4  # Blue channel

        # Object 3: Small dark object (center: 100, 400, radius: 15)
        mask3 = ((x1 - 100) ** 2 + (y1 - 400) ** 2) <= 15 ** 2
        image[0, 0, mask3] = 0.1  # Red channel
        image[1, 0, mask3] = 0.2  # Green channel
        image[2, 0, mask3] = 0.6  # Blue channel

        # Save as zarr
        image_path = os.path.join(temp_dir, "synthetic_image2.zarr")
        zarr_store = create_ome_zarr_output(
                output_zarr_path=image_path,
                shape=image_shape,
                chunks=(1, 1, 256, 256),
                dtype=numpy.float32,
                axes=["c", "z", "y", "x"],
                fill_value=0.0,
            )
        zarr_store[:] = image

        return image_path, image

    @pytest.fixture
    def synthetic_segmentations(self, temp_dir):
        """Create synthetic polygon segmentations corresponding to the image objects"""
        # Create polygons that roughly match the objects in the synthetic image
        polygons = [
            # Object 1: Circle around (150, 150) with radius ~35
            Polygon([
                (120, 120), (180, 120), (180, 180), (120, 180), (120, 120)
            ]),
            # Object 2: Rectangle around (350, 200)
            Polygon([
                (310, 180), (390, 180), (390, 220), (310, 220), (310, 180)
            ]),
            # Object 3: Small square around (100, 400)
            Polygon([
                (85, 385), (115, 385), (115, 415), (85, 415), (85, 385)
            ])
        ]

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'geometry': polygons,
            'object_id': ['obj_1', 'obj_2', 'obj_3'],
            'area': [poly.area for poly in polygons]
        })

        # Save as GeoJSON
        seg_path = os.path.join(temp_dir, "segmentations.geojson")
        gdf.to_file(seg_path, driver='GeoJSON')

        return seg_path, gdf

    @pytest.fixture
    def config_file(self, temp_dir, synthetic_image, synthetic_segmentations):
        """Create configuration file for CellProfiler extraction"""
        image_path, _ = synthetic_image
        seg_path, _ = synthetic_segmentations

        config = {
            'image_path': image_path,
            'segmentations': seg_path,
            'image_dimension_order': ['c', 'y', 'x'],
            'image_downsample': 1,
            'block_size': 128,  # Large enough to capture each object
            'block_method': 'centroid',
            'feature_extraction_method': 'cellprofiler',
            'save_path': os.path.join(temp_dir, 'cellprofiler_features.zarr'),
            'calculate_mask': False,
            'n_workers': 1,  # Single worker for testing
            'python_path': 'python',
            'memory': '8GB'
        }

        config_path = os.path.join(temp_dir, 'cellprofiler_config.toml')
        with open(config_path, 'w') as f:
            toml.dump(config, f)

        return config_path, config

    def test_cellprofiler_model_initialization(self):
        """Test that CellProfiler model initializes correctly"""
        model = CellProfiler()

        assert model.n_features == 271
        assert model.output_shape == (271, 1, 1, 1)
        assert model.feature_names is not None
        assert len(model.feature_names) == 271

        # Test that feature names follow expected patterns
        feature_names = model.feature_names
        assert any('Intensity_' in name for name in feature_names)
        assert any('RadialDistribution_' in name for name in feature_names)
        # assert any('AreaShape_' in name for name in feature_names)

    def test_cellprofiler_single_object_inference(self):
        """Test CellProfiler model on single synthetic object"""
        model = CellProfiler()

        # Create synthetic image + mask data (4 channels: RGB + mask)
        image_data = numpy.random.rand(3, 1, 64, 64).astype(numpy.float32)
        mask_data = numpy.zeros((1, 1, 64, 64), dtype=numpy.float32)

        # Add a single object in the center
        mask_data[0, 0, 20:44, 20:44] = 1.0

        print(numpy.sum(mask_data))

        # Combine image and mask
        combined_data = numpy.concatenate([image_data, mask_data], axis=0)

        print(combined_data.shape)

        # Run inference
        features = model(combined_data)

        print("Area", features[model.feature_names.index("Area")])

        # Validate output
        assert features.shape == (271, 1, 1, 1)
        assert not numpy.all(numpy.isnan(features))

        # Should have some meaningful features (not all zeros)
        non_zero_features = numpy.sum(features != 0)
        assert non_zero_features > 50  # Expect meaningful feature extraction

    def test_cellprofiler_grayscale_conversion(self):
        """Test that 3-channel images are converted to grayscale correctly"""
        model = CellProfiler()

        # Create 3-channel RGB image with distinct values per channel
        rgb_data = numpy.zeros((3, 1, 32, 32), dtype=numpy.float32)
        rgb_data[0, 0, :, :] = 0.8  # Red
        rgb_data[1, 0, :, :] = 0.4  # Green
        rgb_data[2, 0, :, :] = 0.2  # Blue

        # Add mask
        mask_data = numpy.ones((1, 1, 32, 32), dtype=numpy.float32)

        combined_data = numpy.concatenate([rgb_data, mask_data], axis=0)

        # This should use rgb2gray conversion
        features = model(combined_data)

        assert features.shape == (271, 1, 1, 1)
        assert not numpy.all(numpy.isnan(features))

    def test_cellprofiler_multichannel_conversion(self):
        """Test that >3-channel images use mean conversion"""
        model = CellProfiler()

        # Create 5-channel image
        multi_data = numpy.random.rand(5, 1, 32, 32).astype(numpy.float32)
        mask_data = numpy.ones((1, 1, 32, 32), dtype=numpy.float32)

        combined_data = numpy.concatenate([multi_data, mask_data], axis=0)

        # This should use mean conversion
        features = model(combined_data)

        assert features.shape == (271, 1, 1, 1)
        assert not numpy.all(numpy.isnan(features))

    def test_full_cellprofiler_extraction_pipeline(self, config_file):
        """Test the complete CellProfiler extraction pipeline with config"""
        config_path, config = config_file

        # Run extraction using the configuration
        extract(
            input_zarr_path=config['image_path'],
            feature_extraction_method=config['feature_extraction_method'],
            block_size=config['block_size'],
            output_zarr_path=config['save_path'],
            n_workers=config['n_workers'],
            python_path=config['python_path'],
            memory=config['memory'],
            block_method=config['block_method'],
            segmentations=gpd.read_file(config['segmentations']),
            calculate_mask=config['calculate_mask'],
            image_downsample=config['image_downsample']
        )

        # Verify output file was created
        assert os.path.exists(config['save_path'])

        # Load and validate results
        output_zarr = zarr.open(config['save_path'], mode='r')

        # Should have features for 3 objects
        assert output_zarr.shape[0] == 3  # Number of segmentations
        assert output_zarr.shape[1] == 271  # Number of features

        # Check that features were extracted (not all NaN)
        features = output_zarr[:]
        assert not numpy.all(numpy.isnan(features))

        # Each object should have some non-zero features
        for i in range(3):
            object_features = features[i, :]
            non_nan_features = numpy.sum(~numpy.isnan(object_features))
            assert non_nan_features > 200  # Most features should be valid

            non_zero_features = numpy.sum(object_features != 0)
            assert non_zero_features > 50  # Should have meaningful values

    def test_cellprofiler_empty_mask_handling(self):
        """Test handling of regions with no segmentation objects"""
        model = CellProfiler()

        # Create image with empty mask
        image_data = numpy.random.rand(3, 1, 64, 64).astype(numpy.float32)
        empty_mask = numpy.zeros((1, 1, 64, 64), dtype=numpy.float32)

        combined_data = numpy.concatenate([image_data, empty_mask], axis=0)

        # Should return NaN features for empty regions
        features = model(combined_data)

        assert features.shape == (271, 1, 1, 1)
        # With no objects, should return NaN features
        assert numpy.all(numpy.isnan(features))

    def test_cellprofiler_feature_consistency(self):
        """Test that CellProfiler produces consistent features for identical inputs"""
        model = CellProfiler()

        # Create identical inputs
        image_data = numpy.random.rand(3, 1, 64, 64).astype(numpy.float32)
        mask_data = numpy.zeros((1, 1, 64, 64), dtype=numpy.float32)
        mask_data[0, 0, 20:44, 20:44] = 1.0

        combined_data = numpy.concatenate([image_data, mask_data], axis=0)

        # Run inference twice
        features1 = model(combined_data.copy())
        features2 = model(combined_data.copy())

        # Results should be identical
        numpy.testing.assert_array_equal(features1, features2)

    def test_integration_with_slice_generation(self, synthetic_image, synthetic_segmentations):
        """Test that the new slice generation works correctly with CellProfiler"""
        from feature_blocks.slice import generate_centroid_slices_with_single_masks

        image_path, image_data = synthetic_image
        seg_path, gdf = synthetic_segmentations

        # Test slice generation with masks
        slices_with_masks = generate_centroid_slices_with_single_masks(
            shape=image_data.shape,
            size=128,
            segmentations=gdf
        )

        # Should have 3 slices (one per segmentation)
        assert len(slices_with_masks) == 3

        # Each slice should have 3 elements: (id, slice_obj, mask_data)
        for slice_info in slices_with_masks:
            assert len(slice_info) == 3
            centroid_id, slice_obj, mask_data = slice_info

            # Validate slice object structure
            assert len(slice_obj) == 4  # (C, Z, H, W)
            assert slice_obj[0] == slice(None)  # C axis
            assert slice_obj[1] == slice(None)  # Z axis

            # Validate mask data
            assert mask_data.dtype == numpy.int32
            assert mask_data.shape[0] > 0 and mask_data.shape[1] > 0

            # Should contain exactly one object (value 1) or be empty
            unique_values = numpy.unique(mask_data)
            assert len(unique_values) <= 2  # 0 (background) and possibly 1 (object)
            if len(unique_values) == 2:
                assert 0 in unique_values and 1 in unique_values

    def test_config_based_workflow(self, config_file):
        """Test that the configuration-based workflow produces expected results"""
        config_path, config = config_file

        # Parse config and run extraction
        with open(config_path, 'r') as f:
            parsed_config = toml.load(f)

        # Verify config parsing
        assert parsed_config['feature_extraction_method'] == 'cellprofiler'
        assert parsed_config['block_method'] == 'centroid'
        assert os.path.exists(parsed_config['image_path'])
        assert os.path.exists(parsed_config['segmentations'])

        # Run the full pipeline
        segmentations = gpd.read_file(parsed_config['segmentations'])

        extract(
            input_zarr_path=parsed_config['image_path'],
            feature_extraction_method=parsed_config['feature_extraction_method'],
            block_size=parsed_config['block_size'],
            output_zarr_path=parsed_config['save_path'],
            n_workers=parsed_config['n_workers'],
            python_path=parsed_config['python_path'],
            memory=parsed_config['memory'],
            block_method=parsed_config['block_method'],
            segmentations=segmentations,
            calculate_mask=parsed_config['calculate_mask'],
            image_downsample=parsed_config['image_downsample']
        )

        # Validate results
        assert os.path.exists(parsed_config['save_path'])

        results = zarr.open(parsed_config['save_path'], mode='r')

        # Should match number of segmentations and features
        assert results.shape == (len(segmentations), 271)

    def test_cellprofiler_bounding_box_mode_initialization(self):
        """Test that CellProfiler initializes correctly in bounding box mode"""
        model = CellProfiler(use_bounding_box=True)

        assert model.n_features == 271
        assert model.output_shape == (271, 1, 1, 1)
        assert model.use_bounding_box is True
        assert model.feature_names is not None

    def test_cellprofiler_bounding_box_inference(self):
        """Test CellProfiler inference in bounding box mode (no mask channel needed)"""
        model = CellProfiler(use_bounding_box=True)

        # Create image data WITHOUT mask channel (just RGB)
        image_data = numpy.random.rand(3, 1, 64, 64).astype(numpy.float32)

        # Add a bright region in center to ensure features are meaningful
        image_data[:, 0, 20:44, 20:44] = 0.8

        # Run inference - should work without mask channel
        features = model(image_data)

        # Validate output
        assert features.shape == (271, 1, 1, 1)
        assert not numpy.all(numpy.isnan(features))

        # Should have meaningful features
        non_zero_features = numpy.sum(features != 0)
        assert non_zero_features > 50

    def test_cellprofiler_bounding_box_vs_mask_mode(self):
        """Compare bounding box mode vs mask mode output structure"""
        bbox_model = CellProfiler(use_bounding_box=True)
        mask_model = CellProfiler(use_bounding_box=False)

        # Create image data
        image_data = numpy.random.rand(3, 1, 64, 64).astype(numpy.float32)
        image_data[:, 0, 20:44, 20:44] = 0.8

        # Bounding box mode: just image
        bbox_features = bbox_model(image_data)

        # Mask mode: image + mask
        mask_data = numpy.zeros((1, 1, 64, 64), dtype=numpy.float32)
        mask_data[0, 0, 20:44, 20:44] = 1.0
        combined_data = numpy.concatenate([image_data, mask_data], axis=0)
        mask_features = mask_model(combined_data)

        # Both should produce valid feature vectors
        assert bbox_features.shape == (271, 1, 1, 1)
        assert mask_features.shape == (271, 1, 1, 1)
        assert not numpy.all(numpy.isnan(bbox_features))
        assert not numpy.all(numpy.isnan(mask_features))

        # Features should be different (mask mode is more precise)
        # But both should have reasonable values
        assert not numpy.allclose(bbox_features, mask_features)

    def test_cellprofiler_bounding_box_extraction_pipeline(self, temp_dir, synthetic_image, synthetic_segmentations):
        """Test full extraction pipeline with CellProfiler in bounding box mode"""
        image_path, _ = synthetic_image
        seg_path, gdf = synthetic_segmentations

        # Create model in bounding box mode
        model = CellProfiler(use_bounding_box=True)
        output_path = os.path.join(temp_dir, 'cellprofiler_bbox_features.zarr')

        # Run extraction
        extract(
            input_zarr_path=image_path,
            feature_extraction_method=model,  # Pass model instance
            block_size=128,
            output_zarr_path=output_path,
            n_workers=1,
            python_path='python',
            memory='4GB',
            block_method='centroid',
            segmentations=gdf,
            calculate_mask=False,
            image_downsample=1
        )

        # Verify output
        assert os.path.exists(output_path)

        # Load and validate results
        output_zarr = zarr.open(output_path, mode='r')

        # Should have features for 3 objects
        assert output_zarr.shape[0] == 3
        assert output_zarr.shape[1] == 271

        # Check that features were extracted
        features = output_zarr[:]
        assert not numpy.all(numpy.isnan(features))

        # Each object should have meaningful features
        for i in range(3):
            object_features = features[i, :]
            non_nan_features = numpy.sum(~numpy.isnan(object_features))
            assert non_nan_features > 200
            non_zero_features = numpy.sum(object_features != 0)
            assert non_zero_features > 50

    def test_cellprofiler_bounding_box_no_mask_zarr_created(self, temp_dir, synthetic_image, synthetic_segmentations):
        """Test that bounding box mode does NOT create mask zarr store"""
        image_path, _ = synthetic_image
        seg_path, gdf = synthetic_segmentations

        model = CellProfiler(use_bounding_box=True)
        output_path = os.path.join(temp_dir, 'cellprofiler_bbox_test.zarr')

        # Run extraction
        extract(
            input_zarr_path=image_path,
            feature_extraction_method=model,
            block_size=128,
            output_zarr_path=output_path,
            n_workers=1,
            block_method='centroid',
            segmentations=gdf,
        )

        # Verify main output exists
        assert os.path.exists(output_path)

        # Verify mask zarr was NOT created (this is the key difference)
        mask_path = f"{output_path}_masks.zarr"
        assert not os.path.exists(mask_path), "Bounding box mode should not create mask zarr"