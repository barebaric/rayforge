# flake8: noqa:F401
from .base import OpsProducer, CutSide
from .contour import ContourProducer
from .frame import FrameProducer
from .material_test_grid import MaterialTestGridProducer, MaterialTestGridType
from .raster import (
    DepthEngraver,
    DepthMode,
    DitherRasterizer,
    Rasterizer,
)
from .registry import producer_registry, ProducerRegistry
from .shrinkwrap import ShrinkWrapProducer

producer_registry.register(ContourProducer)
producer_registry.register(Rasterizer)
producer_registry.register(Rasterizer, name="DepthEngraver")
producer_registry.register(Rasterizer, name="DitherRasterizer")
producer_registry.register(FrameProducer)
producer_registry.register(MaterialTestGridProducer)
producer_registry.register(ShrinkWrapProducer)

producer_by_name = producer_registry.all_producers()

__all__ = [
    "OpsProducer",
    "CutSide",
    "ContourProducer",
    "DepthMode",
    "DepthEngraver",
    "DitherRasterizer",
    "FrameProducer",
    "MaterialTestGridProducer",
    "MaterialTestGridType",
    "ShrinkWrapProducer",
    "Rasterizer",
    "producer_by_name",
    "producer_registry",
    "ProducerRegistry",
]
