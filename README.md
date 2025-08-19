# Feature Blocks

Extract and cluster features from image blocks using state-of-the-art machine learning models.

## Overview

Feature Blocks is a Python package designed for extracting features from image blocks using various pre-trained models including UNI, PhiKon, DINOv2, and others. It's particularly useful for analyzing large microscopy images, histopathology slides, and other high-resolution scientific imagery.

## Features

- **Multiple Model Support**: UNI, PhiKon, DINOv2, GigaPath, and custom models
- **Flexible Block Processing**: Extract features from image blocks or segmentation centroids
- **Distributed Computing**: Built-in support for SLURM and distributed processing
- **Multiple File Formats**: Support for OME-TIFF, Zarr, and SpatialData formats
- **Configurable Pipeline**: TOML-based configuration for reproducible workflows

## Installation

### Prerequisites

- Python ≥ 3.11
- PyTorch (CPU or GPU version)


### Installation

```bash
git clone https://github.com/callum-jpg/feature_blocks.git
cd feature_blocks
pip install -e ".[dev,test]"
```

## Quick Start

### 1. Create a Configuration File

```toml
# config.toml
image_path = "path/to/your/image.ome.tiff"
segmentations = "path/to/segmentations.geojson"
image_dimension_order = ["c", "y", "x"]
block_size = 112
block_method = "centroid"
feature_extraction_method = "uni"
save_path = "path/to/output.zarr"
n_workers = 4
```

### 2. Run Feature Extraction

```bash
feature_blocks extract config.toml
```

### 3. Use in Python

```python
from feature_blocks.features import extract
from feature_blocks.models import UNI

# Load a model
model = UNI()

# Extract features from image blocks
features = extract(
    image_path="image.ome.tiff",
    model=model,
    block_size=112
)
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image_path` | Path to input image | Required |
| `segmentations` | Path to segmentation polygons | Optional |
| `image_dimension_order` | Dimension order (e.g., ["c", "y", "x"]) | ["c", "y", "x"] |
| `image_downsample` | Image downsampling factor | 1 |
| `block_size` | Size of extraction window | 112 |
| `block_method` | "centroid" or "block" | "centroid" |
| `feature_extraction_method` | Model to use | "uni" |
| `save_path` | Output path | Required |
| `calculate_mask` | Whether to calculate background mask | false |
| `mask_downsample` | Mask downsampling factor | 8 |
| `n_workers` | Number of parallel workers | 1 |
| `python_path` | Python executable path | "python" |

## Supported Models

- **UNI**: Universal pathology foundation model
- **PhiKon**: Pathology foundation model with patch-level features  
- **DINOv2**: Self-supervised vision transformer
- **GigaPath**: Tile-level and patch-level feature extraction
- **Conv**: Simple convolutional baseline
- **LBP**: Local Binary Pattern features
- **Dummy**: Random features for testing

## Advanced Usage

### SpatialData Integration

```python
import spatialdata as sd
from feature_blocks.utility import get_spatial_element

# Load SpatialData object
sdata = sd.read_zarr("path/to/spatial_data.zarr")

# Extract features with SpatialData
features = extract(
    image_path="sdata.zarr:::image_key",
    segmentations="sdata.zarr:::segmentation_shapes",
    # ... other parameters
)
```

### Distributed Computing

```toml
# For SLURM environments
n_workers = 10
python_path = "singularity exec container.simg python"
```

### Custom Models

```python
from feature_blocks.models import BaseModel
import torch.nn as nn

class CustomModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.n_features = 512
        self.output_shape = (512, 1, 1, 1)
        
    def forward(self, x):
        # Your custom feature extraction logic
        return features
```

## API Reference

### CLI Commands

- `feature_blocks extract <config_file>` - Run feature extraction pipeline

### Python API

#### Models
- `UNI()` - Load UNI model
- `PhiKon()` - Load PhiKon model  
- `DINOv2()` - Load DINOv2 model
- `GigaPathTile()` - Load GigaPath tile model
- `GigaPathTilePatch()` - Load GigaPath patch model

#### Functions
- `extract()` - Main feature extraction function
- `load_segmentations()` - Load segmentation data
- `standardise_image()` - Standardize image format

## Examples

See the `examples/` directory for complete workflows:

- `basic_extraction.py` - Simple feature extraction
- `segmentation_workflow.py` - Using segmentation masks
- `batch_processing.py` - Processing multiple images
- `custom_model.py` - Using custom models

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE.txt](LICENSE.txt) for details.

## Citation

If you use Feature Blocks in your research, please cite:

```bibtex
@software{feature_blocks,
  title={Feature Blocks: Extract and cluster features from image blocks},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/feature_blocks}
}
```

## Support

- **Documentation**: [Full documentation](https://yourusername.github.io/feature_blocks)
- **Issues**: [GitHub Issues](https://github.com/yourusername/feature_blocks/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/feature_blocks/discussions)

## Acknowledgments

- UNI model from [MahmoodLab](https://huggingface.co/MahmoodLab/UNI2-h)
- Built with PyTorch, scikit-image, and SpatialData