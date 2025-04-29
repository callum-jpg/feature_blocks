from .conv import ConvFeatures as ConvFeatures
from .dinov2 import DINOv2 as DINOv2
from .dummy_features import DummyModel as DummyModel
from .phikon import PhikonV2 as PhikonV2
from .phikon_patch import PhikonV2Patch as PhikonV2Patch
from .gigapath_tile import GigaPathTile as GigaPathTile

__all__ = [
    "ConvFeatures",
    "DINOv2",
    "DummyModel",
    "PhikonV2",
    "PhikonV2Patch",
    "GigaPathTile",
]

available_models = {
    "conv": ConvFeatures,
    "dinov2": DINOv2,
    "dummy": DummyModel,
    "phikon": PhikonV2,
    "phikon_patch": PhikonV2Patch,
    "gigapath": GigaPathTile,
}