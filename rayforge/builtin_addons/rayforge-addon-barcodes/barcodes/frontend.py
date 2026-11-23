"""
Frontend entry point for the barcodes addon.

Registers the barcode property group, the Object > Add Barcode action,
the asset-browser add action and the asset double-click action.
"""

from gettext import gettext as _
from pathlib import Path

from gi.repository import Gio, GLib

from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.action_registry import MenuPlacement, action_registry
from rayforge.ui_gtk.actions import action_extension_registry
from rayforge.ui_gtk.doceditor.property_providers import (
    property_provider_registry,
)
from rayforge.ui_gtk.icons import register_icon_path

from .commands import add_barcode, default_position, new_barcode_asset
from .property_provider import BarcodePropertyProvider

ADDON_NAME = "barcodes"
ADD_ACTION = "add-barcode"
NEW_ASSET_ACTION = "new-barcode-asset"
ACTIVATE_ACTION = "activate-barcode"
_ICONS_DIR = Path(__file__).parent / "resources" / "icons"

register_icon_path(_ICONS_DIR)
property_provider_registry.register(BarcodePropertyProvider, ADDON_NAME)


def _register_actions(action_manager) -> None:
    window = action_manager.win

    action = Gio.SimpleAction.new(ADD_ACTION, None)

    def on_add(action, param):
        editor = window.doc_editor
        workpiece = add_barcode(editor, default_position(editor))
        window.surface.select_items([workpiece])

    action.connect("activate", on_add)
    action_registry.register(
        action_name=ADD_ACTION,
        action=action,
        addon_name=ADDON_NAME,
        label=_("Add Barcode"),
        menu=MenuPlacement(menu_id="object", priority=60),
    )
    action_manager.actions[ADD_ACTION] = action

    action = Gio.SimpleAction.new(NEW_ASSET_ACTION, None)

    def on_new_asset(action, param):
        new_barcode_asset(window.doc_editor)

    action.connect("activate", on_new_asset)
    action_registry.register(
        action_name=NEW_ASSET_ACTION,
        action=action,
        addon_name=ADDON_NAME,
        label=_("Add Barcode Asset"),
    )
    action_manager.actions[NEW_ASSET_ACTION] = action

    action = Gio.SimpleAction.new(ACTIVATE_ACTION, GLib.VariantType.new("s"))

    def on_activate(action, param):
        editor = window.doc_editor
        asset_uid = param.get_string()
        if editor.doc.get_asset_by_uid(asset_uid) is None:
            return
        workpiece = editor.edit.add_geometry_provider_instance(
            asset_uid, default_position(editor)
        )
        window.surface.select_items([workpiece])

    action.connect("activate", on_activate)
    action_registry.register(
        action_name=ACTIVATE_ACTION,
        action=action,
        addon_name=ADDON_NAME,
        label=_("Add Barcode Instance"),
    )
    action_manager.actions[ACTIVATE_ACTION] = action


action_extension_registry.register_setup(_register_actions, ADDON_NAME)


@hookimpl
def register_actions(action_registry):
    """Registers actions when the addon is enabled at runtime."""
    window = action_registry.window
    action_manager = getattr(window, "action_manager", None)
    if action_manager is not None:
        _register_actions(action_manager)
