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

    def __init__(self, parent_widget: Gtk.Widget, shortcuts: dict):
        super().__init__(parent_widget)

        # Context Data
        self.sketch_element: Optional["SketchElement"] = None
        self.target: Optional[Union["Point", "Entity", "Constraint"]] = None
        self.target_type: Optional[str] = None

        # High-level Signals
        self.tool_selected = Signal()
        self.constraint_selected = Signal()
        self.action_triggered = Signal()

        # --- Shortcut Label Generation ---
        # 1. Map pie menu item 'data' to the action string from editor.py
        data_to_action = {
            # Tools
            "select": "set_tool:select",
            "line": "set_tool:line",
            "arc": "set_tool:arc",
            "circle": "set_tool:circle",
            # Actions
            "construction": "toggle_construction_on_selection",
            # Constraints (Single Key)
            "horiz": "add_horizontal_constraint",
            "vert": "add_vertical_constraint",
            "perp": "add_perpendicular",
            "tangent": "add_tangent",
            "equal": "add_equal_constraint",
            "align": "add_alignment_constraint",
            "symmetry": "add_symmetry_constraint",
            # Constraints (K prefix)
            "dist": "add_distance_constraint",
            "radius": "add_radius_constraint",
            "diameter": "add_diameter_constraint",
        }

        # 2. Invert the editor's shortcut map to group keys by action
        action_to_keys = {}
        for key, action in shortcuts.items():
            if action not in action_to_keys:
                action_to_keys[action] = []
            # Use 'Space' for display purposes
            display_key = "Space" if key == " " else key.upper()
            action_to_keys[action].append(display_key)

        # 3. Helper function to create the label string
        def get_shortcut_label(data_key: str) -> str:
            action = data_to_action.get(data_key)
            if not action:
                return ""
            keys = action_to_keys.get(action)
            if not keys:
                return ""

            # Format each key, e.g., "GC" -> "G-C", " " -> "Space"
            formatted_keys = []
            for key in sorted(keys):
                if len(key) > 1 and key != "Space":
                    # Insert hyphens between characters for sequences
                    formatted_keys.append("-".join(key))
                else:
                    formatted_keys.append(key)

            # Format as (KEY) or (KEY-1, KEY-2)
            return f" ({', '.join(formatted_keys)})"

        # Tools
        label = f"{_('Select')}{get_shortcut_label('select')}"
        item = PieMenuItem("sketch-select-symbolic", label, data="select")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Line')}{get_shortcut_label('line')}"
        item = PieMenuItem("sketch-line-symbolic", label, data="line")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Arc')}{get_shortcut_label('arc')}"
        item = PieMenuItem("sketch-arc-symbolic", label, data="arc")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Circle')}{get_shortcut_label('circle')}"
        item = PieMenuItem("sketch-circle-symbolic", label, data="circle")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        # Actions
        label = f"{_('Construction')}{get_shortcut_label('construction')}"
        item = PieMenuItem(
            "sketch-construction-symbolic", label, data="construction"
        )
        item.on_click.connect(self._on_action_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("delete-symbolic", _("Delete"), data="delete")
        item.on_click.connect(self._on_action_clicked, weak=False)
        self.add_item(item)

        # Constraints
        label = f"{_('Distance')}{get_shortcut_label('dist')}"
        item = PieMenuItem("sketch-distance-symbolic", label, data="dist")
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Horizontal')}{get_shortcut_label('horiz')}"
        item = PieMenuItem(
            "sketch-constrain-horizontal-symbolic", label, data="horiz"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Vertical')}{get_shortcut_label('vert')}"
        item = PieMenuItem(
            "sketch-constrain-vertical-symbolic", label, data="vert"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Radius')}{get_shortcut_label('radius')}"
        item = PieMenuItem("sketch-radius-symbolic", label, data="radius")
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Diameter')}{get_shortcut_label('diameter')}"
        item = PieMenuItem("sketch-diameter-symbolic", label, data="diameter")
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Perpendicular')}{get_shortcut_label('perp')}"
        item = PieMenuItem(
            "sketch-constrain-perpendicular-symbolic",
            label,
            data="perp",
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Tangent')}{get_shortcut_label('tangent')}"
        item = PieMenuItem(
            "sketch-constrain-tangential-symbolic", label, data="tangent"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Align')}{get_shortcut_label('align')}"
        item = PieMenuItem(
            "sketch-constrain-point-symbolic", label, data="align"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Equal')}{get_shortcut_label('equal')}"
        item = PieMenuItem(
            "sketch-constrain-equal-symbolic", label, data="equal"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        label = f"{_('Symmetry')}{get_shortcut_label('symmetry')}"
        item = PieMenuItem(
            "sketch-constrain-symmetric-symbolic", label, data="symmetry"
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
