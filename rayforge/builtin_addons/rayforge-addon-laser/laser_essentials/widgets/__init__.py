"""
Laser Essentials UI Widgets.
"""

from .contour_page import ContourStepSettingsPage
from .frame_page import FrameStepSettingsPage
from .material_test_grid_page import MaterialTestGridSettingsPage
from .raster_page import RasterSettingsPage
from .shrinkwrap_page import ShrinkWrapStepSettingsPage
from .wavefront_page import WavefrontStepSettingsPage

ASSEMBLER_WIDGETS = {
    "contour": ContourStepSettingsPage,
    "frame": FrameStepSettingsPage,
    "raster": RasterSettingsPage,
    "shrinkwrap": ShrinkWrapStepSettingsPage,
    "wavefront": WavefrontStepSettingsPage,
    "material_test_grid": MaterialTestGridSettingsPage,
}

__all__ = [
    "ASSEMBLER_WIDGETS",
    "ContourStepSettingsPage",
    "FrameStepSettingsPage",
    "MaterialTestGridSettingsPage",
    "RasterSettingsPage",
    "ShrinkWrapStepSettingsPage",
    "WavefrontStepSettingsPage",
]
