from gettext import gettext as _
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

from ...core.sketcher.constraints import (
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
from .tools import (
    ArcTool,
    BezierTool,
    CircleTool,
    FillTool,
    LineTool,
    RectangleTool,
    RoundedRectTool,
    SelectTool,
    TextBoxTool,
)

if TYPE_CHECKING:
    from .tools.base import SketchTool
    from ...core.sketcher.selection import SketchSelection
    from ...core.sketcher.sketch import Sketch


ALL_TOOLS = [
    ArcTool,
    BezierTool,
    CircleTool,
    FillTool,
    LineTool,
    RectangleTool,
    RoundedRectTool,
    SelectTool,
    TextBoxTool,
]


CONSTRAINT_SHORTCUTS = [
    (HorizontalConstraint, "h", _("Horizontal")),
    (VerticalConstraint, "v", _("Vertical")),
    (PerpendicularConstraint, "n", _("Perpendicular")),
    (TangentConstraint, "t", _("Tangent")),
    (EqualDistanceConstraint, "e", _("Equal")),
    (SymmetryConstraint, "s", _("Symmetry")),
    (DistanceConstraint, "kd", _("Distance")),
    (RadiusConstraint, "kr", _("Radius")),
    (DiameterConstraint, "ko", _("Diameter")),
    (AngleConstraint, "ka", _("Angle")),
    (AspectRatioConstraint, "kx", _("Aspect Ratio")),
    (CoincidentConstraint, "o", _("Coincident")),
    (CoincidentConstraint, "c", _("Coincident")),
]


def get_active_shortcuts(
    selection: "SketchSelection",
    sketch: Optional["Sketch"] = None,
    active_tool: Optional["SketchTool"] = None,
    is_editing: bool = False,
    is_dragging_fn: Optional[Callable[[], bool]] = None,
) -> List[Tuple[str, str, Optional[Callable[[], bool]]]]:
    """
    Returns list of (key, label, condition) tuples for current context.
    Tool shortcuts are only shown when no constraint shortcuts are applicable.

    Args:
        is_dragging_fn: Optional callable that returns True if currently
                        dragging. Used to hide global shortcuts during drag.
    """
    result = []

    for cls, key, label in CONSTRAINT_SHORTCUTS:
        if cls.can_apply_to(selection, sketch):
            result.append((key, label, is_dragging_fn))

    if not result:
        for tool in ALL_TOOLS:
            if tool.SHORTCUT:
                result.append((*tool.SHORTCUT, is_dragging_fn))

    return result


def get_shortcut_key_for_constraint(constraint_class) -> Optional[str]:
    """
    Returns the shortcut key for a constraint class, or None if not found.
    """
    for cls, key, label in CONSTRAINT_SHORTCUTS:
        if cls == constraint_class:
            return key
    return None


def get_shortcuts_dict() -> dict:
    """
    Returns a dict mapping key sequences to action strings.
    Used by SketchEditor for key handling.
    """
    from .tools import (
        ArcTool,
        BezierTool,
        CircleTool,
        FillTool,
        LineTool,
        RectangleTool,
        RoundedRectTool,
        SelectTool,
        TextBoxTool,
    )

    shortcuts = {}

    tool_map = {
        ArcTool: "arc",
        BezierTool: "bezier",
        CircleTool: "circle",
        FillTool: "fill",
        LineTool: "line",
        RectangleTool: "rectangle",
        RoundedRectTool: "rounded_rect",
        SelectTool: "select",
        TextBoxTool: "text_box",
    }

    for tool_cls, tool_name in tool_map.items():
        if tool_cls.SHORTCUT:
            key, label = tool_cls.SHORTCUT
            shortcuts[key] = f"set_tool:{tool_name}"

    constraint_action_map = {
        HorizontalConstraint: "add_horizontal_constraint",
        VerticalConstraint: "add_vertical_constraint",
        PerpendicularConstraint: "add_perpendicular",
        TangentConstraint: "add_tangent",
        EqualDistanceConstraint: "add_equal_constraint",
        SymmetryConstraint: "add_symmetry_constraint",
        DistanceConstraint: "add_distance_constraint",
        RadiusConstraint: "add_radius_constraint",
        DiameterConstraint: "add_diameter_constraint",
        AngleConstraint: "add_angle_constraint",
        AspectRatioConstraint: "add_aspect_ratio_constraint",
        CoincidentConstraint: "add_alignment_constraint",
    }

    for cls, key, label in CONSTRAINT_SHORTCUTS:
        if cls in constraint_action_map:
            shortcuts[key] = constraint_action_map[cls]

    shortcuts["gn"] = "toggle_construction_on_selection"
    shortcuts["ch"] = "add_chamfer_action"
    shortcuts["cf"] = "add_fillet_action"

    return shortcuts
