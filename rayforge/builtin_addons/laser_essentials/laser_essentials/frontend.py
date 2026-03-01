"""
Frontend entry point for laser-essentials addon.

Registers UI widgets with the main application.
"""

import gettext
from pathlib import Path

from rayforge.core.hooks import hookimpl
from .producers import (
    ContourProducer,
    FrameProducer,
    MaterialTestGridProducer,
    Rasterizer,
    ShrinkWrapProducer,
)
from .widgets import (
    ContourProducerSettingsWidget,
    RasterSettingsWidget,
    FrameProducerSettingsWidget,
    MaterialTestGridSettingsWidget,
    ShrinkWrapProducerSettingsWidget,
)

_localedir = Path(__file__).parent.parent / "locales"
_t = gettext.translation(
    "laser_essentials", localedir=_localedir, fallback=True
)
_ = _t.gettext

ADDON_NAME = "laser_essentials"


@hookimpl
def register_step_widgets(widget_registry):
    """Register step widgets with the widget registry."""
    widget_registry.register(
        ContourProducer,
        ContourProducerSettingsWidget,
        addon_name=ADDON_NAME,
    )
    widget_registry.register(
        Rasterizer,
        RasterSettingsWidget,
        addon_name=ADDON_NAME,
    )
    # DepthEngraver and DitherRasterizer are aliases for Rasterizer
    widget_registry.register(
        Rasterizer,
        RasterSettingsWidget,
        name="DepthEngraver",
        addon_name=ADDON_NAME,
    )
    widget_registry.register(
        Rasterizer,
        RasterSettingsWidget,
        name="DitherRasterizer",
        addon_name=ADDON_NAME,
    )
    widget_registry.register(
        FrameProducer,
        FrameProducerSettingsWidget,
        addon_name=ADDON_NAME,
    )
    widget_registry.register(
        MaterialTestGridProducer,
        MaterialTestGridSettingsWidget,
        addon_name=ADDON_NAME,
    )
    widget_registry.register(
        ShrinkWrapProducer,
        ShrinkWrapProducerSettingsWidget,
        addon_name=ADDON_NAME,
    )
