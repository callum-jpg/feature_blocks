import tempfile
from pathlib import Path

import numpy
import pytest
import zarr

from feature_blocks.cli.app import load_config


def test_load_config():
    """Test that config files are loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.toml"

        config_content = """
image_path = "/path/to/image.zarr"
image_dimension_order = ["c", "z", "y", "x"]
block_size = 112
block_method = "block"
feature_extraction_method = "dummy"
save_path = "/path/to/output.zarr"
n_workers = 2
memory = "4GB"
calculate_mask = false
"""
        config_path.write_text(config_content)

        config = load_config(str(config_path))

        assert config["block_size"] == 112
        assert config["feature_extraction_method"] == "dummy"
        assert config["n_workers"] == 2
        assert config["memory"] == "4GB"
        assert config["block_method"] == "block"


def test_load_config_with_invalid_toml():
    """Test that invalid TOML raises an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "invalid_config.toml"
        config_path.write_text("this is not valid toml [[[")

        with pytest.raises(Exception):
            load_config(str(config_path))


def test_zarr_configuration():
    """Test that zarr stores are created with proper compression and synchronization."""
    import zarr as zarr_lib

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test.zarr"

        # Create zarr with the same settings as in extract function
        synchronizer = zarr_lib.ProcessSynchronizer(f"{test_path}.sync")
        compressor = zarr_lib.Blosc(cname="zstd", clevel=3, shuffle=zarr_lib.Blosc.SHUFFLE)

        z = zarr_lib.create(
            shape=(10, 100),
            chunks=(1, 100),
            dtype=numpy.float32,
            store=test_path,
            synchronizer=synchronizer,
            compressor=compressor,
        )

        # Verify compression is enabled
        assert z.compressor is not None, "Compressor should be set"
        assert z.compressor.cname == "zstd", "Should use zstd compression"
        assert z.compressor.clevel == 3, "Should use compression level 3"

        # Verify synchronizer is enabled
        assert z.synchronizer is not None, "Synchronizer should be set"


def test_model_caching():
    """Test that model caching works correctly."""
    from feature_blocks.task import infer
    from feature_blocks.task._infer import _model_cache

    # Clear cache first
    _model_cache.clear()

    # Test dummy model caching
    dummy_input = numpy.zeros((1, 1, 1, 1), dtype=numpy.float32)

    result1 = infer(dummy_input, "dummy")
    assert result1.shape[0] > 0, "Model should return features"

    # Verify model is cached
    assert "dummy" in _model_cache, "Model should be cached after first use"

    # Second call should use cached model
    result2 = infer(dummy_input, "dummy")
    assert result2.shape == result1.shape, "Cached model should return same shape"