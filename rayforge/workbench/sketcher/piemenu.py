import logging
from typing import Optional, Union, TYPE_CHECKING
from blinker import Signal
from gi.repository import Gtk

from rayforge.shared.ui.piemenu import PieMenu, PieMenuItem

if TYPE_CHECKING:
    from rayforge.workbench.sketcher.element import SketchElement
    from rayforge.core.sketcher.entities import Point, Entity
    from rayforge.core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class SketchPieMenu(PieMenu):
    """
    Subclass of PieMenu specifically for the SketcherCanvas.
    Maps menu item clicks to high-level signals.
    """

    def __init__(self, parent_widget: Gtk.Widget):
        super().__init__(parent_widget)

        # Context Data
        self.sketch_element: Optional["SketchElement"] = None
        self.target: Optional[Union["Point", "Entity", "Constraint"]] = None
        self.target_type: Optional[str] = None

        # High-level Signals
        self.tool_selected = Signal()
        self.constraint_selected = Signal()
        self.action_triggered = Signal()

        # Tools
        item = PieMenuItem("drag-handle-symbolic", "Select", data="select")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("laser-path-symbolic", "Line", data="line")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("laps-symbolic", "Arc", data="arc")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        # Actions
        item = PieMenuItem(
            "layer-symbolic", "Construction", data="construction"
        )
        item.on_click.connect(self._on_action_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("delete-symbolic", "Delete", data="delete")
        item.on_click.connect(self._on_action_clicked, weak=False)
        self.add_item(item)

        # Constraints
        item = PieMenuItem(
            "tabs-equidistant-symbolic", "Distance", data="dist"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem(
            "align-vertical-center-symbolic", "Horizontal", data="horiz"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem(
            "align-horizontal-center-symbolic", "Vertical", data="vert"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

    def set_context(
        self,
        sketch_element: "SketchElement",
        target: Optional[Union["Point", "Entity", "Constraint"]],
        target_type: Optional[str],
    ):
        """
        Updates the context for the menu before it is shown.

        :param sketch_element: The parent SketchElement (provides access to
                               Sketch, Selection, etc.).
        :param target: The specific object under the cursor (Point, Entity,
                       Constraint), or None.
        :param target_type: String identifier for the target type
                            (e.g., 'point', 'line', 'constraint', 'junction').
        """
        self.sketch_element = sketch_element
        self.target = target
        self.target_type = target_type

        sel_count = 0
        if self.sketch_element and self.sketch_element.selection:
            # Just for debug logging context
            sel = self.sketch_element.selection
            sel_count = len(sel.point_ids) + len(sel.entity_ids)

        logger.debug(
            f"PieMenu Context: Type={target_type}, Target={target}, "
            f"SelectionCount={sel_count}"
        )

    def _on_tool_clicked(self, sender):
        """Handle tool selection signals."""
        if sender.data:
            logger.info(f"Emitting tool selection: {sender.data}")
            self.tool_selected.send(self, tool=sender.data)

    def _on_constraint_clicked(self, sender):
        """Handle constraint selection signals."""
        if sender.data:
            logger.info(f"Emitting constraint: {sender.data}")
            self.constraint_selected.send(self, constraint_type=sender.data)

    def _on_action_clicked(self, sender):
        """Handle generic action signals."""
        if sender.data:
            logger.info(f"Emitting action: {sender.data}")
            self.action_triggered.send(self, action=sender.data)
