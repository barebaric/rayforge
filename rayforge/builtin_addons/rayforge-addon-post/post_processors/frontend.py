"""
Frontend entry point for post_processors addon.

Registers UI widgets for transformer settings with the main application.
"""

from rayforge.core.hooks import hookimpl

from .widgets import TRANSFORMER_WIDGETS

ADDON_NAME = "post_processors"


@hookimpl
def register_transformer_widgets(transformer_widget_registry):
    """Register transformer settings widget classes."""
    for transformer_cls, widget_cls in TRANSFORMER_WIDGETS.items():
        transformer_widget_registry.register(
            transformer_cls, widget_cls, ADDON_NAME
        )
