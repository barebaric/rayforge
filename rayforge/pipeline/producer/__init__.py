# flake8: noqa:F401
import inspect
from .base import OpsProducer, CutSide
from .contour import ContourProducer
from .depth import DepthEngraver
from .dither_rasterize import DitherRasterizer
from .frame import FrameProducer
from .material_test_grid import MaterialTestGridProducer, MaterialTestGridType
from .rasterize import Rasterizer
from .shrinkwrap import ShrinkWrapProducer

producer_by_name = dict(
    [
        (name, obj)
        for name, obj in locals().items()
        if inspect.isclass(obj)
        and issubclass(obj, OpsProducer)
        and not inspect.isabstract(obj)
    ]
)

__all__ = [
    "OpsProducer",
    "CutSide",
    "ContourProducer",
    "DepthEngraver",
    "DitherRasterizer",
    "FrameProducer",
    "MaterialTestGridProducer",
    "MaterialTestGridType",
    "ShrinkWrapProducer",
    "Rasterizer",
    "producer_by_name",
]
