"""
Frontend entry point for the barcodes addon: the Add Barcode action in
the Object menu and the Barcode group in the properties panel.
"""

from gettext import gettext as _

from gi.repository import Gio

from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.action_registry import MenuPlacement, action_registry
from rayforge.ui_gtk.actions import action_extension_registry
from rayforge.ui_gtk.doceditor.property_providers import (
    property_provider_registry,
)

from .commands import add_barcode, default_position
from .property_provider import BarcodePropertyProvider

ADDON_NAME = "barcodes"
ADD_ACTION = "add-barcode"


def _register_actions(action_manager):
    action = Gio.SimpleAction.new(ADD_ACTION, None)

    def on_activate(action, param):
        window = action_manager.win
        editor = window.doc_editor
        workpiece = add_barcode(editor, default_position(editor))
        window.surface.select_items([workpiece])

    action.connect("activate", on_activate)
    action_manager.win.add_action(action)
    action_manager.actions[ADD_ACTION] = action
    action_registry.register(
        action_name=ADD_ACTION,
        action=action,
        addon_name=ADDON_NAME,
        label=_("Add Barcode"),
        menu=MenuPlacement(menu_id="object", priority=60),
    )


@hookimpl
def register_actions(action_registry):
    action_extension_registry.register_setup(_register_actions, ADDON_NAME)


@hookimpl
def main_window_ready(main_window):
    property_provider_registry.register(BarcodePropertyProvider, ADDON_NAME)
