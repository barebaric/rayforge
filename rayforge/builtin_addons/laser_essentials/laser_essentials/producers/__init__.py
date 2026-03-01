"""
Laser Essentials Producers.

Provides producer implementations for laser cutting operations.
"""

from .contour import ContourProducer, CutOrder
from .raster import Rasterizer, DepthMode
from .frame import FrameProducer
from .material_test_grid import (
    MaterialTestGridProducer,
    MaterialTestGridType,
    get_material_test_proportional_size,
    draw_material_test_preview,
)
from .shrinkwrap import ShrinkWrapProducer

__all__ = [
    "ContourProducer",
    "CutOrder",
    "Rasterizer",
    "DepthMode",
    "FrameProducer",
    "MaterialTestGridProducer",
    "MaterialTestGridType",
    "get_material_test_proportional_size",
    "draw_material_test_preview",
    "ShrinkWrapProducer",
]
