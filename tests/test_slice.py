from feature_blocks.slice import generate_nd_slices, filter_slices_by_mask, normalize_slices, generate_centroid_slices
from feature_blocks.features import extract
import geopandas
import numpy
import shapely

from unittest.mock import Mock, MagicMock
import pytest
import numpy 
from itertools import product
from typing import List, Tuple

class TestGenerateNdSlices:
    """Test suite for generate_nd_slices function"""
    
    def test_basic_functionality(self):
        """Test basic slice generation with simple parameters"""
        shape = (3, 10, 100, 100)
        size = 50
        slice_axes = [2, 3]  # Slice H and W dimensions
        
        slices = generate_nd_slices(shape, size, slice_axes)
        
        # Should generate slices for each combination
        assert len(slices) == 4
        
        # Check that non-slice axes use slice(None)
        for slc in slices:
            assert slc[0] == slice(None)  # C axis
            assert slc[1] == slice(None)  # Z axis
            assert isinstance(slc[2], slice)  # H axis
            assert isinstance(slc[3], slice)  # W axis
    
    def test_single_axis_slicing(self):
        """Test slicing on a single axis"""
        shape = (1, 1, 100, 100)
        size = 25
        slice_axes = [2]  # Only slice H dimension
        
        slices = generate_nd_slices(shape, size, slice_axes)
        
        # Should generate slices for the H dimension
        expected_starts = list(range(0, 100 - 1, 25))  # [0, 25, 50, 75]
        
        assert len(slices) == len(expected_starts)
        
        for i, slc in enumerate(slices):
            assert slc[0] == slice(None)  # C axis
            assert slc[1] == slice(None)  # Z axis
            assert slc[2].start == expected_starts[i]
            assert slc[2].stop == expected_starts[i] + size
            assert slc[3] == slice(None)  # W axis
    
    def test_no_slice_axes(self):
        """Test when no slice_axes are provided (should default to empty list)"""
        shape = (3, 10, 100, 100)
        size = 50
        
        slices = generate_nd_slices(shape, size)
        
        # Should return one slice with all slice(None)
        assert len(slices) == 1
        assert all(slc == slice(None) for slc in slices[0])
    
    def test_empty_slice_axes(self):
        """Test with explicitly empty slice_axes list"""
        shape = (3, 10, 100, 100)
        size = 50
        slice_axes = []
        
        slices = generate_nd_slices(shape, size, slice_axes)
        
        # Should return one slice with all slice(None)
        assert len(slices) == 1
        assert all(slc == slice(None) for slc in slices[0])
    
    def test_all_axes_slicing(self):
        """Test slicing on all axes"""
        shape = (4, 4, 100, 100)
        size = 2
        slice_axes = [0, 1, 2, 3]
        
        slices = generate_nd_slices(shape, size, slice_axes)
        
        # (2 * 2 * 50 * 50)
        assert len(slices) == 10000
        
        # Check that all slices have proper slice objects
        for slc in slices:
            assert all(isinstance(s, slice) for s in slc)
    
    def test_invalid_shape_assertion(self):
        """Test that function raises assertion error for non-4D shapes"""
        with pytest.raises(AssertionError, match="Expected shape of length 4"):
            generate_nd_slices((100, 100), 50, [0, 1])
        
        with pytest.raises(AssertionError, match="Expected shape of length 4"):
            generate_nd_slices((3, 10, 100, 100, 5), 50, [2, 3])
    
    def test_large_size_parameter(self):
        """Test with size parameter larger than dimension"""
        shape = (1, 1, 50, 50)
        size = 100  # Larger than dimensions
        slice_axes = [2, 3]
        
        slices = generate_nd_slices(shape, size, slice_axes)
        
        # Should still work but slices may extend beyond array bounds
        assert len(slices) == 1
        
        # Check that slice bounds are as expected
        for slc in slices:
            assert slc[2].stop > shape[2] or slc[3].stop > shape[3]


class TestGenerateCentroidSlices:
    """Test suite for generate_centroid_slices function"""
    
    def create_mock_polygon(self, centroid_x, centroid_y, name=None, id_val=None):
        """Helper to create mock polygon with specified centroid"""
        polygon = Mock()
        polygon.geometry.centroid.x = centroid_x
        polygon.geometry.centroid.y = centroid_y
        polygon.name = name
        if id_val is not None:
            polygon.__getitem__ = Mock(return_value=id_val)
        return polygon
    
    def test_basic_centroid_slicing(self):
        """Test basic centroid slice generation"""
        shape = (3, 10, 1000, 1000)
        size = 100
        
        # Create mock polygons
        polygon1 = self.create_mock_polygon(500, 500, name=0)
        polygon2 = self.create_mock_polygon(200, 300, name=1)
        
        # Mock the GeoDataFrame
        segmentations = Mock()
        
        def mock_apply(func, axis):
            results = []
            for i, poly in enumerate([polygon1, polygon2]):
                poly.name = i
                result = func(poly)
                results.append(result)
            return Mock(tolist=lambda: results)
        
        segmentations.apply = mock_apply
        
        slices = generate_centroid_slices(shape, size, segmentations)
        
        # Should return list of tuples (id, slice)
        assert len(slices) == 2
        
        # Check first slice (centroid at 500, 500)
        centroid_id, slc = slices[0]
        assert centroid_id == 0
        assert slc[0] == slice(None)  # C axis
        assert slc[1] == slice(None)  # Z axis
        assert slc[2] == slice(450, 550)  # Y axis (500 ± 50)
        assert slc[3] == slice(450, 550)  # X axis (500 ± 50)
    
    def test_boundary_clamping(self):
        """Test that slices are clamped to array boundaries"""
        shape = (1, 1, 100, 100)
        size = 60
        
        # Polygon near edge (centroid at 20, 20)
        polygon = self.create_mock_polygon(20, 20, name=0)
        
        segmentations = Mock()
        def mock_apply(func, axis):
            result = func(polygon)
            return Mock(tolist=lambda: [result])
        
        # Add an apply method to the segmentations mock object
        segmentations.apply = mock_apply
        
        slices = generate_centroid_slices(shape, size, segmentations)
        
        centroid_id, slc = slices[0]

        assert len(slices) == 1
        
        # Y slice should be clamped to start at 0 (not -10)
        assert slc[2].start == 0
        assert slc[2].stop == 50  # 20 + 30
        
        # X slice should be clamped to start at 0 (not -10)
        assert slc[3].start == 0
        assert slc[3].stop == 50  # 20 + 30
    
    def test_with_id_column(self):
        """Test using custom ID column instead of index"""
        shape = (1, 1, 100, 100)
        size = 20
        id_col = "custom_id"
        
        polygon = self.create_mock_polygon(50, 50, name=0)
        polygon.__getitem__ = Mock(return_value="poly_001")
        
        segmentations = Mock()
        def mock_apply(func, axis):
            result = func(polygon)
            return Mock(tolist=lambda: [result])
        
        segmentations.apply = mock_apply
        
        slices = generate_centroid_slices(shape, size, segmentations, id_col)
        
        centroid_id, slc = slices[0]
        assert centroid_id == "poly_001"
    
    def test_invalid_shape_assertion(self):
        """Test assertion error for non-4D shapes"""
        with pytest.raises(AssertionError, match="Expected shape of length 4"):
            generate_centroid_slices((100, 100), 50, Mock())


class TestFilterSlicesByMask:
    """Test suite for filter_slices_by_mask function"""
    
    def test_basic_filtering(self):
        """Test basic mask filtering functionality"""
        # Create a simple 2D mask for testing
        mask_array = numpy.zeros((10, 10))
        mask_array[2:4, 2:4] = 1  # Small foreground region
        
        # Create some slices
        slices = [
            (slice(0, 2), slice(0, 2)),  # Background region
            (slice(2, 4), slice(2, 4)),  # Foreground region
            (slice(6, 8), slice(6, 8)),  # Background region
            (slice(1, 3), slice(1, 3)),  # Overlaps foreground
        ]
        
        foreground, background = filter_slices_by_mask(slices, mask_array)
        
        # Should have 2 foreground slices (indices 1 and 3)
        assert len(foreground) == 2
        assert len(background) == 2
        
        # Check that foreground slices actually contain positive values
        for slc in foreground:
            assert numpy.any(mask_array[slc] > 0)
        
        # Check that background slices contain only zeros
        for slc in background:
            assert numpy.all(mask_array[slc] == 0)
    
    def test_all_background(self):
        """Test when all slices are in background"""
        mask_array = numpy.zeros((10, 10))
        
        slices = [
            (slice(0, 2), slice(0, 2)),
            (slice(4, 6), slice(4, 6)),
            (slice(8, 10), slice(8, 10)),
        ]
        
        foreground, background = filter_slices_by_mask(slices, mask_array)
        
        assert len(foreground) == 0
        assert len(background) == 3
    
    def test_all_foreground(self):
        """Test when all slices are in foreground"""
        mask_array = numpy.ones((10, 10))
        
        slices = [
            (slice(0, 2), slice(0, 2)),
            (slice(4, 6), slice(4, 6)),
            (slice(8, 10), slice(8, 10)),
        ]
        
        foreground, background = filter_slices_by_mask(slices, mask_array)
        
        assert len(foreground) == 3
        assert len(background) == 0
    
    def test_multidimensional_mask(self):
        """Test with higher dimensional mask"""
        mask_array = numpy.zeros((2, 3, 10, 10))
        mask_array[0, 1, 2:4, 2:4] = 1
        
        slices = [
            (slice(None), slice(None), slice(0, 2), slice(0, 2)),  # Background
            (slice(None), slice(None), slice(2, 4), slice(2, 4)),  # Foreground
        ]
        
        foreground, background = filter_slices_by_mask(slices, mask_array)
        
        assert len(foreground) == 1
        assert len(background) == 1
    
    def test_empty_slices_list(self):
        """Test with empty slices list"""
        mask_array = numpy.ones((10, 10))
        slices = []
        
        foreground, background = filter_slices_by_mask(slices, mask_array)
        
        assert len(foreground) == 0
        assert len(background) == 0


class TestNormalizeSlices:
    """Test suite for normalize_slices function"""
    
    def test_basic_normalization(self):
        """Test basic slice normalization"""
        slices = [
            slice(0, 200),
            slice(400, 600),
            slice(800, 1000),
        ]
        step = 200
        
        normalized = normalize_slices(slices, step)
        
        expected = [
            slice(0, 1),
            slice(2, 3),
            slice(4, 5),
        ]
        
        assert normalized == expected
    
    def test_custom_step(self):
        """Test normalization with custom step size"""
        slices = [
            slice(0, 100),
            slice(500, 600),
            slice(1000, 1100),
        ]
        step = 100
        
        normalized = normalize_slices(slices, step)
        
        expected = [
            slice(0, 1),
            slice(5, 6),
            slice(10, 11),
        ]
        
        assert normalized == expected
    
    def test_non_exact_division(self):
        """Test normalization when indices don't divide evenly"""
        slices = [
            slice(150, 350),
            slice(450, 650),
        ]
        step = 200
        
        normalized = normalize_slices(slices, step)
        
        expected = [
            slice(0, 1),  # 150//200 = 0, 350//200 = 1
            slice(2, 3),  # 450//200 = 2, 650//200 = 3
        ]
        
        assert normalized == expected
    
    def test_default_step(self):
        """Test normalization with default step size"""
        slices = [
            slice(0, 400),
            slice(600, 800),
        ]
        
        normalized = normalize_slices(slices)  # Default step=200
        
        expected = [
            slice(0, 2),  # 0//200 = 0, 400//200 = 2
            slice(3, 4),  # 600//200 = 3, 800//200 = 4
        ]
        
        assert normalized == expected
    
    def test_zero_start_indices(self):
        """Test normalization with zero start indices"""
        slices = [
            slice(0, 200),
            slice(0, 400),
            slice(0, 600),
        ]
        step = 200
        
        normalized = normalize_slices(slices, step)
        
        expected = [
            slice(0, 1),
            slice(0, 2),
            slice(0, 3),
        ]
        
        assert normalized == expected
    
    def test_empty_slices_list(self):
        """Test normalization with empty slices list"""
        slices = []
        normalized = normalize_slices(slices)
        assert normalized == []
