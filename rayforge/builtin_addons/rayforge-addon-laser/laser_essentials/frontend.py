"""
Frontend entry point for laser-essentials addon.

Registers UI widgets with the main application.
"""

import gettext
from pathlib import Path

from rayforge.core.hooks import hookimpl
from .widgets import PRODUCER_WIDGETS

_localedir = Path(__file__).parent.parent / "locale"
_t = gettext.translation(
    "laser_essentials", localedir=_localedir, fallback=True
)
_ = _t.gettext

ADDON_NAME = "laser_essentials"


@hookimpl
def step_settings_loaded(dialog, step, producer):
    """Add step settings widgets based on producer type."""
    if producer is None:
        return

    widget_cls = PRODUCER_WIDGETS.get(type(producer))
    if widget_cls:
        dialog.add(
            widget_cls(
                dialog.editor,
                step.typelabel,
                producer,
                dialog,
                step,
            )
        )
