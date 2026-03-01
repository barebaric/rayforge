"""
Backend entry point for laser-essentials addon.

Registers producers and menu items with the main application.
"""

import gettext
from pathlib import Path

from gi.repository import Gio

from rayforge.core.hooks import hookimpl
from .producers import (
    ContourProducer,
    FrameProducer,
    MaterialTestGridProducer,
    Rasterizer,
    ShrinkWrapProducer,
)
from .steps import (
    ContourStep,
    EngraveStep,
    FrameStep,
    MaterialTestStep,
    ShrinkWrapStep,
)
from .commands import MaterialTestCmd

_localedir = Path(__file__).parent.parent / "locales"
_t = gettext.translation(
    "laser_essentials", localedir=_localedir, fallback=True
)
_ = _t.gettext

ADDON_NAME = "laser_essentials"


@hookimpl
def register_producers(producer_registry):
    """Register producers with the producer registry."""
    producer_registry.register(ContourProducer, addon_name=ADDON_NAME)
    producer_registry.register(Rasterizer, addon_name=ADDON_NAME)
    # DepthEngraver and DitherRasterizer are aliases for Rasterizer
    producer_registry.register(
        Rasterizer, name="DepthEngraver", addon_name=ADDON_NAME
    )
    producer_registry.register(
        Rasterizer, name="DitherRasterizer", addon_name=ADDON_NAME
    )
    producer_registry.register(FrameProducer, addon_name=ADDON_NAME)
    producer_registry.register(MaterialTestGridProducer, addon_name=ADDON_NAME)
    producer_registry.register(ShrinkWrapProducer, addon_name=ADDON_NAME)


@hookimpl
def register_steps(step_registry):
    """Register steps with the step registry."""
    step_registry.register(ContourStep, addon_name=ADDON_NAME)
    step_registry.register(EngraveStep, addon_name=ADDON_NAME)
    step_registry.register(FrameStep, addon_name=ADDON_NAME)
    step_registry.register(MaterialTestStep, addon_name=ADDON_NAME)
    step_registry.register(ShrinkWrapStep, addon_name=ADDON_NAME)


@hookimpl
def register_menu_items(menu_registry):
    """Register menu items with the menu registry."""
    menu_registry.register(
        item_id="laser_essentials.material_test",
        label=_("Create Material Test Grid"),
        action="win.material_test",
        menu="Tools",
        priority=100,
        addon_name=ADDON_NAME,
    )


@hookimpl
def register_actions(window):
    """Register window actions."""
    action = Gio.SimpleAction.new("material_test", None)

    def on_activate(action, param):
        editor = window.doc_editor
        cmd = MaterialTestCmd(editor)
        cmd.create_test_grid()

    action.connect("activate", on_activate)
    window.add_action(action)
