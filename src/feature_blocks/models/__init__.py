from .cellprofiler import CellProfiler as CellProfiler
from .conv import ConvFeatures as ConvFeatures
from .dinov2 import DINOv2 as DINOv2
from .dummy_features import DummyModel as DummyModel
from .gigapath_tile import GigaPathTile as GigaPathTile
from .gigapath_tile_patch import GigaPathTilePatch as GigaPathTilePatch
from .lbp import LBP as LBP
from .phikon import PhikonV2 as PhikonV2
from .phikon_patch import PhikonV2Patch as PhikonV2Patch
from .h_optimus_0 import H_optimus_0 as H_optimus_0
from .uni import UNI as UNI

__all__ = [
    "CellProfiler",
    "ConvFeatures",
    "DINOv2",
    "DummyModel",
    "PhikonV2",
    "PhikonV2Patch",
    "GigaPathTile",
    "GigaPathTilePatch",
    "UNI",
    "LBP",
    "lbp",
    "h_optimus_0",
]

available_models = {
    "cellprofiler": CellProfiler,
    "conv": ConvFeatures,
    "dinov2": DINOv2,
    "dummy": DummyModel,
    "phikon": PhikonV2,
    "phikon_patch": PhikonV2Patch,
    "gigapath": GigaPathTile,
    "gigapath_patch": GigaPathTilePatch,
    "uni": UNI,
    "LBP": LBP,
    "lbp": LBP,
    "h_optimus_0": H_optimus_0,
}
