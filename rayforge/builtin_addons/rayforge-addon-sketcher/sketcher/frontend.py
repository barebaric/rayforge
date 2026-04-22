"""
Frontend entry point for sketcher addon.

Registers UI widgets with the main application.
"""

import gettext
import logging
from pathlib import Path

from rayforge.core.hooks import hookimpl

logger = logging.getLogger(__name__)

_localedir = Path(__file__).parent.parent / "locale"
_t = gettext.translation("sketcher", localedir=_localedir, fallback=True)
_ = _t.gettext

ADDON_NAME = "sketcher"


@hookimpl
def step_settings_loaded(dialog, step, producer):
    """Add step settings widgets based on producer type."""
    pass


@hookimpl
def register_commands(command_registry):
    """Register SketchCmd with the command registry."""
    from .ui_gtk.sketch_cmd import SketchCmd

    command_registry.register("sketch", SketchCmd, ADDON_NAME)


@hookimpl
def main_window_ready(main_window):
    """Set up sketch studio and mode command when main window is ready."""
    from .ui_gtk import setup_sketch_page

    setup_sketch_page(main_window)


@hookimpl
def on_unload():
    """Clean up sketch studio when addon is disabled."""
    logger.info("on_unload hook called, tearing down sketch page")
    from .ui_gtk import teardown_sketch_page

    teardown_sketch_page()
