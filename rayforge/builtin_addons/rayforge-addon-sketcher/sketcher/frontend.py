"""
Frontend entry point for sketcher addon.

Registers UI widgets with the main application.
"""

import gettext
from pathlib import Path

from rayforge.core.hooks import hookimpl

_localedir = Path(__file__).parent.parent / "locale"
_t = gettext.translation("sketcher", localedir=_localedir, fallback=True)
_ = _t.gettext

ADDON_NAME = "sketcher"


@hookimpl
def step_settings_loaded(dialog, step, producer):
    """Add step settings widgets based on producer type."""
    pass


@hookimpl
def main_window_ready(main_window):
    """Set up sketch studio and mode command when main window is ready."""
    from .ui_gtk import setup_sketch_page

    setup_sketch_page(main_window)
