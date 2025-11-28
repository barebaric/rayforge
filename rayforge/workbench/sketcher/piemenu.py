import logging
from typing import Optional, Union, TYPE_CHECKING
from blinker import Signal
from gi.repository import Gtk

from rayforge.shared.ui.piemenu import PieMenu, PieMenuItem

if TYPE_CHECKING:
    from rayforge.workbench.sketcher.sketchelement import SketchElement
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
        item = PieMenuItem("sketch-select-symbolic", "Select", data="select")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("sketch-line-symbolic", "Line", data="line")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("sketch-arc-symbolic", "Arc", data="arc")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("sketch-circle-symbolic", "Circle", data="circle")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        # Actions
        item = PieMenuItem(
            "sketch-construction-symbolic", "Construction", data="construction"
        )
        item.on_click.connect(self._on_action_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("delete-symbolic", "Delete", data="delete")
        item.on_click.connect(self._on_action_clicked, weak=False)
        self.add_item(item)

        # Constraints
        item = PieMenuItem("sketch-distance-symbolic", "Distance", data="dist")
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem(
            "sketch-constrain-horizontal-symbolic", "Horizontal", data="horiz"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem(
            "sketch-constrain-vertical-symbolic", "Vertical", data="vert"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("sketch-radius-symbolic", "Radius", data="radius")
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem(
            "sketch-diameter-symbolic", "Diameter", data="diameter"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem(
            "sketch-constrain-perpendicular-symbolic",
            "Perpendicular",
            data="perp",
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem(
            "sketch-constrain-tangential-symbolic", "Tangent", data="tangent"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem(
            "sketch-constrain-point-symbolic", "Align", data="align"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem(
            "sketch-constrain-equal-symbolic", "Equal", data="equal"
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
            sel = self.sketch_element.selection
            sel_count = len(sel.point_ids) + len(sel.entity_ids)

        logger.debug(
            f"PieMenu Context: Type={target_type}, Target={target}, "
            f"SelectionCount={sel_count}"
        )

        has_target = target is not None

        # Update item visibility based on supported actions/constraints
        if self.sketch_element:
            for item in self.items:
                key = item.data

                # Tools (creation/select) are only visible if empty space was
                # clicked
                if key in ("select", "line", "arc", "circle"):
                    item.visible = not has_target

                # Actions (delete, construction)
                elif key in ("delete", "construction"):
                    item.visible = self.sketch_element.is_action_supported(key)

                # Constraints (dist, horiz, vert, etc.)
                else:
                    item.visible = self.sketch_element.is_constraint_supported(
                        key
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
