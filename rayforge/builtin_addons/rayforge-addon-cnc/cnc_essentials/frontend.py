"""
Frontend entry point for cnc-essentials addon.

Registers UI widgets with the main application.
"""

from pathlib import Path

from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.icons import register_icon_path

from .widgets import ASSEMBLER_WIDGETS

ADDON_NAME = "cnc_essentials"
_ICONS_DIR = Path(__file__).parent / "resources" / "icons"

register_icon_path(_ICONS_DIR)


@hookimpl
def register_step_settings_pages(step_settings_page_registry):
    """Register step settings page classes based on assembler name."""
    for assembler_name, page_cls in ASSEMBLER_WIDGETS.items():
        step_settings_page_registry.register(
            assembler_name, page_cls, ADDON_NAME
        )
