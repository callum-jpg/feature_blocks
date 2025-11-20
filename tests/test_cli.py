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
