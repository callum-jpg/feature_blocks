#!/usr/bin/env python3
"""
Configuration-based workflow example.

This example shows how to use Feature Blocks with a TOML configuration file,
similar to how you would use the CLI tool.
"""

import tempfile
from pathlib import Path
import numpy as np
import tomllib

from feature_blocks.cli.app import extract

def create_sample_config():
    """Create a sample configuration file."""
    config_content = """
# Example configuration for feature extraction
image_path = "sample_image.ome.tiff"
image_dimension_order = ["c", "y", "x"]
image_downsample = 1
block_size = 112
block_method = "block"
feature_extraction_method = "dummy"
save_path = "output_features.zarr"
calculate_mask = false
n_workers = 1
python_path = "python"
"""
    
    # Write config to temporary file
    config_file = Path("example_config.toml")
    config_file.write_text(config_content.strip())
    
    return str(config_file)

def create_sample_image():
    """Create a sample image file."""
    # Create a dummy RGB image
    image = np.random.randint(0, 255, size=(3, 500, 500), dtype=np.uint8)
    
    # In practice, you would save as OME-TIFF
    # For this example, we'll just save as numpy array
    np.save("sample_image.ome.tiff", image)
    
    return "sample_image.ome.tiff"

def main():
    print("Creating sample data...")
    
    # Create sample image and config
    image_path = create_sample_image()
    config_path = create_sample_config()
    
    print(f"Created sample image: {image_path}")
    print(f"Created config file: {config_path}")
    
    # Read and display config
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nRunning feature extraction with config: {config_path}")
    
    # This would normally run the full extraction pipeline
    # extract(config_path)
    
    print("Example completed! In a real workflow, you would:")
    print("1. Prepare your actual image file")
    print("2. Modify the config file with your parameters")
    print("3. Run: feature_blocks extract your_config.toml")
    
    # Clean up
    Path(image_path).unlink(missing_ok=True)
    Path(config_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()