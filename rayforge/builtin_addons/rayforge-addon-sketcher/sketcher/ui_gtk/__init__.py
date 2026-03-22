from typing import TYPE_CHECKING, Optional

from gi.repository import Gio, GLib
from gettext import gettext as _

from rayforge.context import get_context
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.action_registry import (
    MenuPlacement,
    action_registry,
)
from rayforge.ui_gtk.actions import action_extension_registry
from rayforge.ui_gtk.canvas2d.context_menu import (
    context_menu_extension_registry,
)
from rayforge.ui_gtk.doceditor.asset_row_factory import (
    asset_row_widget_registry,
)
from rayforge.ui_gtk.doceditor.property_providers import (
    property_provider_registry,
)
from ..core.sketch import Sketch
from .asset_row_widget import SketchAssetRowWidget
from .property_provider import SketchPropertyProvider
from .sketch_mode_cmd import SketchModeCmd
from .sketchelement import SketchElement

if TYPE_CHECKING:
    from gi.repository import Gtk

    from rayforge.core.item import DocItem
    from rayforge.ui_gtk.actions import ActionManager
    from rayforge.ui_gtk.canvas2d.surface import WorkSurface
    from rayforge.ui_gtk.mainwindow import MainWindow

    from .studio import SketchStudio


_sketch_studio: Optional["SketchStudio"] = None
_sketch_mode_cmd: Optional["SketchModeCmd"] = None


def _build_sketch_context_menu_items(
    surface: "WorkSurface",
    item: Optional["DocItem"],
    gesture: "Gtk.Gesture",
    menu: Gio.Menu,
):
    """
    Context menu extension handler for sketch workpieces.

    Adds "Edit Sketch" and "Export Object" items when the context menu
    is shown for a sketch-based workpiece.
    """
    if not isinstance(item, WorkPiece):
        return

    if not item.geometry_provider_uid:
        return

    # Create a section for sketch-specific items at the top of the menu
    sketch_section = Gio.Menu.new()
    sketch_section.append_item(
        Gio.MenuItem.new(_("Edit Sketch"), "win.edit_sketch")
    )
    sketch_section.append_item(
        Gio.MenuItem.new(_("Export Object..."), "win.export_object")
    )

    # Prepend the sketch section to the menu
    menu.prepend_section(None, sketch_section)


def _on_config_changed(sender, **kwargs):
    """Handle config changes to update sketch studio dimensions."""
    if _sketch_studio is None:
        return

    config = get_context().config
    if config.machine:
        width_mm, height_mm = config.machine.axis_extents
        _sketch_studio.set_world_size(width_mm, height_mm)


def setup_sketch_page(main_window: "MainWindow") -> "SketchStudio":
    """
    Create and register the sketch studio page with the main window.

    This function is called by the sketcher addon's main_window_ready hook
    to set up the sketch editing page. All sketch-specific logic lives here.

    Args:
        main_window: The MainWindow instance

    Returns:
        The created SketchStudio instance
    """
    global _sketch_studio, _sketch_mode_cmd

    if _sketch_studio is not None:
        return _sketch_studio

    from .studio import SketchStudio

    config = get_context().config
    if config.machine:
        width_mm, height_mm = config.machine.axis_extents
    else:
        width_mm, height_mm = 100.0, 100.0

    sketch_studio = SketchStudio(
        main_window, width_mm=width_mm, height_mm=height_mm
    )

    _sketch_mode_cmd = SketchModeCmd(main_window, main_window.doc_editor)

    sketch_studio.finished.connect(_sketch_mode_cmd.on_sketch_finished)
    sketch_studio.cancelled.connect(_sketch_mode_cmd.on_sketch_cancelled)

    main_window.add_stack_page("sketch", sketch_studio)

    config.changed.connect(_on_config_changed)

    _sketch_studio = sketch_studio

    return sketch_studio


def get_sketch_studio() -> Optional["SketchStudio"]:
    """Get the global SketchStudio instance."""
    return _sketch_studio


def get_sketch_mode_cmd() -> Optional["SketchModeCmd"]:
    """Get the global SketchModeCmd instance."""
    return _sketch_mode_cmd


def _register_actions(action_manager: "ActionManager"):
    """Register sketch actions with the ActionManager."""
    if _sketch_mode_cmd is None:
        setup_sketch_page(action_manager.win)
    cmd = _sketch_mode_cmd
    assert cmd is not None

    action = Gio.SimpleAction.new("new_sketch", None)
    action.connect("activate", cmd.on_new_sketch)
    action_manager.win.add_action(action)
    action_manager.actions["new_sketch"] = action
    action_registry.register(
        action_name="new_sketch",
        action=action,
        addon_name="sketcher",
        label=_("New Sketch"),
        menu=MenuPlacement(menu_id="object", priority=50),
    )

    action = Gio.SimpleAction.new("edit_sketch", None)
    action.connect("activate", cmd.on_edit_sketch)
    action_manager.win.add_action(action)
    action_manager.actions["edit_sketch"] = action

    action = Gio.SimpleAction.new("export_object", None)
    action.connect("activate", cmd.on_export_object)
    action_manager.win.add_action(action)
    action_manager.actions["export_object"] = action

    action = Gio.SimpleAction.new("add-sketch", None)
    action.connect("activate", cmd.on_new_sketch)
    action_manager.win.add_action(action)
    action_manager.actions["add-sketch"] = action

    action = Gio.SimpleAction.new("activate-sketch", GLib.VariantType.new("s"))
    action.connect("activate", cmd.on_activate_sketch)
    action_manager.win.add_action(action)
    action_manager.actions["activate-sketch"] = action

    action = Gio.SimpleAction.new(
        "edit-sketch-item", GLib.VariantType.new("s")
    )
    action.connect("activate", cmd.on_edit_sketch_item)
    action_manager.win.add_action(action)
    action_manager.actions["edit-sketch-item"] = action


def _update_action_states(action_manager: "ActionManager"):
    """Update sketch action states based on selection."""
    action_manager.actions["add-sketch"].set_enabled(True)

    selected_wps = action_manager.win.surface.get_selected_workpieces()

    can_edit_sketch = False
    can_export_object = False
    if len(selected_wps) == 1:
        wp = selected_wps[0]
        if wp.geometry_provider_uid:
            can_edit_sketch = True
        if wp.boundaries is not None and not wp.boundaries.is_empty():
            can_export_object = True

    if "export_object" in action_manager.actions:
        action_manager.actions["export_object"].set_enabled(can_export_object)
    if "edit_sketch" in action_manager.actions:
        action_manager.actions["edit_sketch"].set_enabled(can_edit_sketch)


def register():
    """Register sketch module components with the application.

    This function is called during application initialization to
    register sketch-specific components with their respective registries.
    """
    property_provider_registry.register(SketchPropertyProvider)
    asset_row_widget_registry.register(Sketch, SketchAssetRowWidget)
    context_menu_extension_registry.register(_build_sketch_context_menu_items)
    action_extension_registry.register_setup(_register_actions)
    action_extension_registry.register_state_update(_update_action_states)


# Auto-register when module is imported
register()

__all__ = [
    "SketchAssetRowWidget",
    "SketchElement",
    "SketchModeCmd",
    "SketchPropertyProvider",
    "get_sketch_mode_cmd",
    "get_sketch_studio",
    "setup_sketch_page",
]
