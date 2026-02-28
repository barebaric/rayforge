"""
Backend entry point for laser-essentials addon.

Registers producers and menu items with the main application.
"""

import gettext
from pathlib import Path

from rayforge.core.hooks import hookimpl

_localedir = Path(__file__).parent / "locales"
_t = gettext.translation(
    "laser_essentials", localedir=_localedir, fallback=True
)
_ = _t.gettext


@hookimpl
def register_producers(producer_registry):
    """Register producers with the producer registry."""
    # Producers will be registered here after migration
    pass


@hookimpl
def register_steps(step_registry):
    """Register steps with the step registry."""
    # Steps will be registered here after migration
    pass


@hookimpl
def register_menu_items(menu_registry):
    """Register menu items with the menu registry."""
    # Menu items will be registered here after migration
    pass
