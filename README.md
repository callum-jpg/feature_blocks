# Feature Blocks

Extract features efficiently from OME-Zarr images using deep learning models, CellProfiler, or any other feature extraction method. If it returns a 1D feature vector, it can be easily implemented.

`feature_blocks` was designed with Dask in mind for distribution, but it also contains methods for GPU-inference if the feature extractor can be GPU accelerated. 

## Features

- **Adaptable feature extractor**: UNI, PhiKon, DINOv2, GigaPath, and custom models
- **Flexible Block Processing**: Extract features from image blocks or from crops centered on segmentation centroids
- **Distributed Computing**: Built-in support for SLURM, LSF and distributed processing
- **Multiple File Formats**: Support for OME-TIFF, Zarr, and SpatialData formats
- **Configurable Pipeline**: TOML-based configuration for reproducible workflows

### Installation

```bash
git clone https://github.com/callum-jpg/feature_blocks.git
cd feature_blocks
uv pip install -e ".[dev,test]"
```

## Quick Start

### Create a Configuration File

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
memory = "16GB"
```

If you have a SpatialData object, you can also load images and segmentations directly by using the `::` accessor, like so:
```toml
# config.toml
image_path = "path/to/your/spatialdata_object.zarr::image_name"
segmentations = "path/to/your/spatialdata_object.zarr::segmentation_shapes_name"
...
```

### Run Feature Extraction (CLI)

```bash
feature_blocks extract config.toml
```

### Use in Python

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
- **CellProfiler**: Extract CellProfiler features using `cp_measure`. Requires `block_method='centroid'`.

### Distributed Computing

```toml
# For SLURM/LSF environments
n_workers = 10
python_path = "singularity exec container.simg python" # Define the python path (eg. from a singularity container)
memory = "32GB"  # Configurable memory per worker
```

## Examples

See the `examples/` directory for complete workflows

## Benchmarking

Run benchmarking with:
```bash
uv run python benchmarks/benchmark.py
```