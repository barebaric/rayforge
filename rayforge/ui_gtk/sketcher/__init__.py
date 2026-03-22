from typing import TYPE_CHECKING, Optional

from gi.repository import Gio
from gettext import gettext as _

from ...core.sketcher import Sketch
from ...core.workpiece import WorkPiece
from ..canvas2d.context_menu import context_menu_extension_registry
from ..doceditor.asset_row_factory import asset_row_widget_registry
from ..doceditor.property_providers import property_provider_registry
from .asset_row_widget import SketchAssetRowWidget
from .property_provider import SketchPropertyProvider
from .sketchelement import SketchElement

if TYPE_CHECKING:
    from ...core.item import DocItem
    from ..canvas2d.surface import WorkSurface
    from gi.repository import Gtk


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


def register():
    """Register sketch module components with the application.

    This function is called during application initialization to
    register sketch-specific components with their respective registries.
    """
    property_provider_registry.register(SketchPropertyProvider)
    asset_row_widget_registry.register(Sketch, SketchAssetRowWidget)
    context_menu_extension_registry.register(_build_sketch_context_menu_items)


# Auto-register when module is imported
register()

__all__ = ["SketchAssetRowWidget", "SketchElement", "SketchPropertyProvider"]
