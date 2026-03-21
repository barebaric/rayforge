import logging
from typing import Optional, Union, TYPE_CHECKING
from blinker import Signal
from gi.repository import Gtk

from rayforge.ui_gtk.shared.piemenu import PieMenu, PieMenuItem
from .tools import TOOL_REGISTRY

if TYPE_CHECKING:
    from rayforge.ui_gtk.sketcher.sketchelement import SketchElement
    from rayforge.core.sketcher.entities import Point, Entity
    from rayforge.core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class SketchPieMenu(PieMenu):
    """
    Subclass of PieMenu specifically for the SketcherCanvas.
    Builds menu dynamically from registered tools.
    """

    def __init__(self, parent_widget: Gtk.Widget):
        super().__init__(parent_widget)

        self.sketch_element: Optional["SketchElement"] = None
        self.target: Optional[Union["Point", "Entity", "Constraint"]] = None
        self.target_type: Optional[str] = None

        self.tool_selected = Signal()

        self._tool_to_display_key: dict[str, str] = {}
        for tool_name, tool_cls in TOOL_REGISTRY.items():
            if tool_cls.SHORTCUTS:
                key = tool_cls.SHORTCUTS[0]
                display_key = "Space" if key == " " else key.upper()
                self._tool_to_display_key[tool_name] = display_key

    def set_context(
        self,
        sketch_element: "SketchElement",
        target: Optional[Union["Point", "Entity", "Constraint"]],
        target_type: Optional[str],
    ):
        self.sketch_element = sketch_element
        self.target = target
        self.target_type = target_type

        sel_count = 0
        if self.sketch_element and self.sketch_element.selection:
            sel = self.sketch_element.selection
            sel_count = len(sel.point_ids) + len(sel.entity_ids)

        logger.debug(
            f"PieMenu Context: Type={target_type}, "
            f"Target={target}, SelectionCount={sel_count}"
        )

        self._rebuild_menu()

    def _rebuild_menu(self):
        """Rebuild menu items based on context and tool availability."""
        self.items.clear()
        self._active_index = -1

        if not self.sketch_element:
            return

        for tool_name, tool in self.sketch_element.tools.items():
            if tool.ICON is None or tool.LABEL is None:
                continue

            if not tool.is_available(self.target, self.target_type):
                continue

            label = self._get_tool_label(tool_name, tool.LABEL)
            item = PieMenuItem(tool.ICON, label, data=tool_name)
            item.on_click.connect(self._on_tool_clicked, weak=False)
            self.add_item(item)

    def has_items(self) -> bool:
        """Returns True if the menu has any items."""
        return len(self.items) > 0

    def _get_tool_label(self, tool_name: str, base_label: str) -> str:
        """Get label with shortcut hint if available."""
        key = self._tool_to_display_key.get(tool_name)
        if not key:
            return base_label

        if len(key) > 1 and key != "Space":
            formatted_key = "-".join(key)
        else:
            formatted_key = key

        return f"{base_label} ({formatted_key})"

    def _on_tool_clicked(self, sender):
        """Handle tool selection signals."""
        if sender.data:
            logger.info(f"Tool activated: {sender.data}")
            self.tool_selected.send(self, tool=sender.data)
