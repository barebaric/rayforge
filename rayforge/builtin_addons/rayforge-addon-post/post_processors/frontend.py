"""
Frontend entry point for post_processors addon.

Registers UI widgets for transformer settings with the main application.
"""

from rayforge.core.hooks import hookimpl

from .widgets import TRANSFORMER_WIDGETS

ADDON_NAME = "post_processors"


@hookimpl
def transformer_settings_loaded(dialog, step, transformer):
    """Add transformer settings widgets based on transformer type."""
    widget_cls = TRANSFORMER_WIDGETS.get(type(transformer))
    if widget_cls:
        dialog.add(
            widget_cls(
                dialog.editor,
                transformer.label,
                transformer,
                dialog,
                step,
            )
        )
