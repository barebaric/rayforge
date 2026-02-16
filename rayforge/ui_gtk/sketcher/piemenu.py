import logging
from typing import Optional, Union, TYPE_CHECKING
from blinker import Signal
from gi.repository import Gtk

from rayforge.ui_gtk.shared.piemenu import PieMenu, PieMenuItem
from rayforge.core.sketcher.constraints import (
    AngleConstraint,
    AspectRatioConstraint,
    CoincidentConstraint,
    DiameterConstraint,
    DistanceConstraint,
    EqualDistanceConstraint,
    HorizontalConstraint,
    PerpendicularConstraint,
    RadiusConstraint,
    SymmetryConstraint,
    TangentConstraint,
    VerticalConstraint,
)

if TYPE_CHECKING:
    from rayforge.ui_gtk.sketcher.sketchelement import SketchElement
    from rayforge.core.sketcher.entities import Point, Entity
    from rayforge.core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class SketchPieMenu(PieMenu):
    """
    Subclass of PieMenu specifically for the SketcherCanvas.
    Maps menu item clicks to high-level signals.
    """

    _MENU_ITEMS = {
        "select": {
            "icon": "sketch-select-symbolic",
            "label": _("Select"),
            "action": "set_tool:select",
            "handler": "_on_tool_clicked",
        },
        "line": {
            "icon": "sketch-line-symbolic",
            "label": _("Line"),
            "action": "set_tool:line",
            "handler": "_on_tool_clicked",
        },
        "arc": {
            "icon": "sketch-arc-symbolic",
            "label": _("Arc"),
            "action": "set_tool:arc",
            "handler": "_on_tool_clicked",
        },
        "text_box": {
            "icon": "sketch-text-symbolic",
            "label": _("Text Box"),
            "action": "set_tool:text_box",
            "handler": "_on_tool_clicked",
        },
        "circle": {
            "icon": "sketch-circle-symbolic",
            "label": _("Circle"),
            "action": "set_tool:circle",
            "handler": "_on_tool_clicked",
        },
        "rectangle": {
            "icon": "sketch-rect-symbolic",
            "label": _("Rectangle"),
            "action": "set_tool:rectangle",
            "handler": "_on_tool_clicked",
        },
        "rounded_rect": {
            "icon": "sketch-rounded-rect-symbolic",
            "label": _("Rounded Rectangle"),
            "action": "set_tool:rounded_rect",
            "handler": "_on_tool_clicked",
        },
        "fill": {
            "icon": "sketch-fill-symbolic",
            "label": _("Fill"),
            "action": "set_tool:fill",
            "handler": "_on_tool_clicked",
        },
        "construction": {
            "icon": "sketch-construction-symbolic",
            "label": _("Construction"),
            "action": "toggle_construction_on_selection",
            "handler": "_on_action_clicked",
        },
        "chamfer": {
            "icon": "sketch-chamfer-symbolic",
            "label": _("Chamfer"),
            "action": "add_chamfer_action",
            "handler": "_on_action_clicked",
        },
        "fillet": {
            "icon": "sketch-fillet-symbolic",
            "label": _("Fillet"),
            "action": "add_fillet_action",
            "handler": "_on_action_clicked",
        },
        "delete": {
            "icon": "delete-symbolic",
            "label": _("Delete"),
            "action": None,
            "handler": "_on_action_clicked",
        },
        "dist": {
            "icon": "sketch-distance-symbolic",
            "constraint_class": DistanceConstraint,
            "action": "add_distance_constraint",
            "handler": "_on_constraint_clicked",
        },
        "horiz": {
            "icon": "sketch-constrain-horizontal-symbolic",
            "constraint_class": HorizontalConstraint,
            "action": "add_horizontal_constraint",
            "handler": "_on_constraint_clicked",
        },
        "vert": {
            "icon": "sketch-constrain-vertical-symbolic",
            "constraint_class": VerticalConstraint,
            "action": "add_vertical_constraint",
            "handler": "_on_constraint_clicked",
        },
        "radius": {
            "icon": "sketch-radius-symbolic",
            "constraint_class": RadiusConstraint,
            "action": "add_radius_constraint",
            "handler": "_on_constraint_clicked",
        },
        "diameter": {
            "icon": "sketch-diameter-symbolic",
            "constraint_class": DiameterConstraint,
            "action": "add_diameter_constraint",
            "handler": "_on_constraint_clicked",
        },
        "perp": {
            "icon": "sketch-constrain-perpendicular-symbolic",
            "constraint_class": PerpendicularConstraint,
            "action": "add_perpendicular",
            "handler": "_on_constraint_clicked",
        },
        "angle": {
            "icon": "sketch-constrain-angle-symbolic",
            "constraint_class": AngleConstraint,
            "action": "add_angle_constraint",
            "handler": "_on_constraint_clicked",
        },
        "tangent": {
            "icon": "sketch-constrain-tangential-symbolic",
            "constraint_class": TangentConstraint,
            "action": "add_tangent",
            "handler": "_on_constraint_clicked",
        },
        "align": {
            "icon": "sketch-constrain-point-symbolic",
            "constraint_class": CoincidentConstraint,
            "action": "add_alignment_constraint",
            "handler": "_on_constraint_clicked",
        },
        "equal": {
            "icon": "sketch-constrain-equal-symbolic",
            "constraint_class": EqualDistanceConstraint,
            "action": "add_equal_constraint",
            "handler": "_on_constraint_clicked",
        },
        "symmetry": {
            "icon": "sketch-constrain-symmetric-symbolic",
            "constraint_class": SymmetryConstraint,
            "action": "add_symmetry_constraint",
            "handler": "_on_constraint_clicked",
        },
        "aspect_ratio": {
            "icon": "sketch-constrain-aspect-symbolic",
            "constraint_class": AspectRatioConstraint,
            "action": "add_aspect_ratio_constraint",
            "handler": "_on_constraint_clicked",
        },
    }

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

        # Invert the editor's shortcut map to group keys by action
        action_to_keys = {}
        for key, action in shortcuts.items():
            if action not in action_to_keys:
                action_to_keys[action] = []
            display_key = "Space" if key == " " else key.upper()
            action_to_keys[action].append(display_key)

        def get_shortcut_label(data_key: str) -> str:
            config = self._MENU_ITEMS.get(data_key)
            if not config or not config["action"]:
                return ""
            keys = action_to_keys.get(config["action"])
            if not keys:
                return ""

            formatted_keys = []
            for key in sorted(keys):
                if len(key) > 1 and key != "Space":
                    formatted_keys.append("-".join(key))
                else:
                    formatted_keys.append(key)

            return f" ({', '.join(formatted_keys)})"

        for data_key, config in self._MENU_ITEMS.items():
            if "constraint_class" in config:
                label = (
                    f"{config['constraint_class'].get_type_name()}"
                    f"{get_shortcut_label(data_key)}"
                )
            else:
                label = f"{config['label']}{get_shortcut_label(data_key)}"
            item = PieMenuItem(config["icon"], label, data=data_key)
            handler = getattr(self, config["handler"])
            item.on_click.connect(handler, weak=False)
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
                if key in (
                    "select",
                    "line",
                    "arc",
                    "circle",
                    "rounded_rect",
                    "rectangle",
                    "fill",
                    "text_box",
                ):
                    item.visible = not has_target

                # Actions (delete, construction)
                elif key in ("delete", "construction", "chamfer", "fillet"):
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
            config = self._MENU_ITEMS.get(sender.data)
            if config and "action" in config:
                logger.info(f"Emitting constraint action: {config['action']}")
                self.constraint_selected.send(self, action=config["action"])

    def _on_action_clicked(self, sender):
        """Handle generic action signals."""
        if sender.data:
            logger.info(f"Emitting action: {sender.data}")
            self.action_triggered.send(self, action=sender.data)
