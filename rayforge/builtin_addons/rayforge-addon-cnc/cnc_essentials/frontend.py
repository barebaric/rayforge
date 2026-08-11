"""
Frontend entry point for cnc-essentials addon.

Registers UI widgets with the main application.
"""

from rayforge.core.hooks import hookimpl

from .widgets import ASSEMBLER_WIDGETS

ADDON_NAME = "cnc_essentials"


@hookimpl
def register_step_settings_pages(step_settings_page_registry):
    """Register step settings page classes based on assembler name."""
    for assembler_name, page_cls in ASSEMBLER_WIDGETS.items():
        step_settings_page_registry.register(
            assembler_name, page_cls, ADDON_NAME
        )
