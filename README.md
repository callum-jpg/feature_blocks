# Feature Blocks

Extract and cluster features from image blocks using state-of-the-art machine learning models.

## Overview

Feature Blocks is a Python package designed for extracting features from image blocks using various pre-trained models including UNI, PhiKon, DINOv2, and others. It's particularly useful for analyzing large microscopy images, histopathology slides, and other high-resolution scientific imagery.

## Features

- **Multiple Model Support**: UNI, PhiKon, DINOv2, GigaPath, and custom models
- **Flexible Block Processing**: Extract features from image blocks or segmentation centroids
- **Distributed Computing**: Built-in support for SLURM, LSF and distributed processing
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

### 2. Run Feature Extraction (CLI)

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
    input_zarr_path="image.ome.tiff",
    model=model,
    block_size=112,
    output_zarr_path = "output.zarr",
)
```

## Supported Models

- **UNI**: Universal pathology foundation model
- **PhiKon**: Pathology foundation model with patch-level features  
- **DINOv2**: Self-supervised vision transformer
- **GigaPath**: Tile-level and patch-level feature extraction
- **Conv**: Simple convolutional baseline
- **LBP**: Local Binary Pattern features
- **Dummy**: Random features for testing

### Distributed Computing

```toml
# For SLURM/LSF environments
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
- `UNI()`
- `PhiKon()`
- `DINOv2()` 
- `GigaPathTile()`
- `GigaPathTilePatch()`

## Examples

See the `examples/` directory for complete workflows

## License

This project is licensed under the MIT License - see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgments

- UNI model from [MahmoodLab](https://huggingface.co/MahmoodLab/UNI2-h)
- Built with PyTorch, scikit-image, and SpatialData