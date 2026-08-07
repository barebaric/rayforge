"""
Frontend entry point for cnc-essentials addon.

Registers UI widgets with the main application.
"""

from rayforge.core.hooks import hookimpl

from .widgets import ASSEMBLER_WIDGETS

ADDON_NAME = "cnc_essentials"


@hookimpl
def step_settings_loaded(dialog, step, producer):
    """Provide the step settings page based on assembler name."""
    widget_cls = ASSEMBLER_WIDGETS.get(step.ASSEMBLER_NAME)
    if widget_cls:
        dialog.set_step_settings_page(widget_cls(dialog.editor, step))
