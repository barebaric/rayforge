from .angle_constraint_tool import AngleConstraintTool
from .arc_tool import ArcTool
from .array_base import ArrayToolBase
from .aspect_ratio_constraint_tool import AspectRatioConstraintTool
from .base import SketcherKey, SketchTool
from .chamfer_tool import ChamferTool
from .circle_tool import CircleTool
from .circular_array_tool import CircularArrayTool
from .coincident_constraint_tool import CoincidentConstraintTool
from .construction_tool import ConstructionTool
from .curve_along_array_tool import CurveAlongArrayTool
from .delete_tool import DeleteTool
from .diameter_constraint_tool import DiameterConstraintTool
from .distance_constraint_tool import DistanceConstraintTool
from .equal_constraint_tool import EqualConstraintTool
from .fill_tool import FillTool
from .fillet_tool import FilletTool
from .grid_tool import GridTool
from .horizontal_constraint_tool import HorizontalConstraintTool
from .mirror_tool import MirrorHorizontalTool, MirrorVerticalTool
from .offset_tool import OffsetTool
from .path_tool import PathTool
from .perpendicular_constraint_tool import PerpendicularConstraintTool
from .radius_constraint_tool import RadiusConstraintTool
from .rectangle_tool import RectangleTool
from .rounded_rect_tool import RoundedRectTool
from .select_tool import SelectTool
from .snap_mixin import SnapMixin
from .straighten_tool import StraightenTool
from .symmetry_constraint_tool import SymmetryConstraintTool
from .tangent_constraint_tool import TangentConstraintTool
from .text_box_tool import TextBoxTool
from .vertical_constraint_tool import VerticalConstraintTool
from .waypoint_sharp_tool import WaypointSharpTool
from .waypoint_smooth_tool import WaypointSmoothTool
from .waypoint_symmetric_tool import WaypointSymmetricTool

TOOL_REGISTRY = {
    "angle": AngleConstraintTool,
    "arc": ArcTool,
    "aspect_ratio": AspectRatioConstraintTool,
    "chamfer": ChamferTool,
    "circle": CircleTool,
    "circular_array": CircularArrayTool,
    "coincident": CoincidentConstraintTool,
    "construction": ConstructionTool,
    "curve_along_array": CurveAlongArrayTool,
    "delete": DeleteTool,
    "diameter": DiameterConstraintTool,
    "distance": DistanceConstraintTool,
    "equal": EqualConstraintTool,
    "fill": FillTool,
    "fillet": FilletTool,
    "grid": GridTool,
    "horizontal": HorizontalConstraintTool,
    "mirror_vertical": MirrorVerticalTool,
    "mirror_horizontal": MirrorHorizontalTool,
    "offset": OffsetTool,
    "path": PathTool,
    "perpendicular": PerpendicularConstraintTool,
    "radius": RadiusConstraintTool,
    "rectangle": RectangleTool,
    "rounded_rect": RoundedRectTool,
    "select": SelectTool,
    "straighten": StraightenTool,
    "symmetry": SymmetryConstraintTool,
    "tangent": TangentConstraintTool,
    "text_box": TextBoxTool,
    "vertical": VerticalConstraintTool,
    "waypoint_sharp": WaypointSharpTool,
    "waypoint_smooth": WaypointSmoothTool,
    "waypoint_symmetric": WaypointSymmetricTool,
}


def build_key_to_tool_map() -> dict[str, str]:
    """Build reverse lookup: key sequence -> tool name."""
    key_map = {}
    for tool_name, tool_cls in TOOL_REGISTRY.items():
        for key in tool_cls.SHORTCUTS:
            key_map[key] = tool_name
    return key_map


def build_action_tool_map() -> dict[str, str]:
    """Build mapping: action name -> tool name for all tools."""
    action_map = {}
    for tool_name in TOOL_REGISTRY:
        action_name = f"tool_{tool_name}"
        action_map[action_name] = tool_name
    return action_map


KEY_TO_TOOL = build_key_to_tool_map()
ACTION_TOOL_MAP = build_action_tool_map()

__all__ = [
    "ACTION_TOOL_MAP",
    "KEY_TO_TOOL",
    "TOOL_REGISTRY",
    "AngleConstraintTool",
    "ArcTool",
    "ArrayToolBase",
    "AspectRatioConstraintTool",
    "ChamferTool",
    "CircleTool",
    "CircularArrayTool",
    "CoincidentConstraintTool",
    "ConstructionTool",
    "CurveAlongArrayTool",
    "DeleteTool",
    "DiameterConstraintTool",
    "DistanceConstraintTool",
    "EqualConstraintTool",
    "FillTool",
    "FilletTool",
    "GridTool",
    "HorizontalConstraintTool",
    "MirrorHorizontalTool",
    "MirrorVerticalTool",
    "PathTool",
    "PerpendicularConstraintTool",
    "RadiusConstraintTool",
    "RectangleTool",
    "RoundedRectTool",
    "SelectTool",
    "SketchTool",
    "SketcherKey",
    "SnapMixin",
    "StraightenTool",
    "SymmetryConstraintTool",
    "TangentConstraintTool",
    "TextBoxTool",
    "VerticalConstraintTool",
    "WaypointSharpTool",
    "WaypointSmoothTool",
    "WaypointSymmetricTool",
]
