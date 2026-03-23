from .angle_constraint_tool import AngleConstraintTool
from .arc_tool import ArcTool
from .aspect_ratio_constraint_tool import AspectRatioConstraintTool
from .base import SketchTool, SketcherKey
from .chamfer_tool import ChamferTool
from .circle_tool import CircleTool
from .coincident_constraint_tool import CoincidentConstraintTool
from .construction_tool import ConstructionTool
from .delete_tool import DeleteTool
from .diameter_constraint_tool import DiameterConstraintTool
from .distance_constraint_tool import DistanceConstraintTool
from .equal_constraint_tool import EqualConstraintTool
from .fillet_tool import FilletTool
from .fill_tool import FillTool
from .grid_tool import GridTool
from .horizontal_constraint_tool import HorizontalConstraintTool
from .path_tool import PathTool
from .perpendicular_constraint_tool import PerpendicularConstraintTool
from .radius_constraint_tool import RadiusConstraintTool
from .rectangle_tool import RectangleTool
from .rounded_rect_tool import RoundedRectTool
from .select_tool import SelectTool
from .symmetry_constraint_tool import SymmetryConstraintTool
from .straighten_tool import StraightenTool
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
    "coincident": CoincidentConstraintTool,
    "construction": ConstructionTool,
    "delete": DeleteTool,
    "diameter": DiameterConstraintTool,
    "distance": DistanceConstraintTool,
    "equal": EqualConstraintTool,
    "fill": FillTool,
    "fillet": FilletTool,
    "grid": GridTool,
    "horizontal": HorizontalConstraintTool,
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
    for tool_name, tool_cls in TOOL_REGISTRY.items():
        action_name = f"tool_{tool_name}"
        action_map[action_name] = tool_name
    return action_map


def build_studio_shortcuts() -> dict[str, list[str]]:
    """Build mapping: action name -> shortcut for ACTION_SHORTCUT tools."""
    shortcuts = {}
    for tool_name, tool_cls in TOOL_REGISTRY.items():
        if tool_cls.ACTION_SHORTCUT is not None:
            action_name = f"sketch.tool_{tool_name}"
            shortcuts[action_name] = [tool_cls.ACTION_SHORTCUT]
    return shortcuts


KEY_TO_TOOL = build_key_to_tool_map()
ACTION_TOOL_MAP = build_action_tool_map()
STUDIO_SHORTCUTS = build_studio_shortcuts()

__all__ = [
    "ACTION_TOOL_MAP",
    "AngleConstraintTool",
    "ArcTool",
    "AspectRatioConstraintTool",
    "PathTool",
    "ChamferTool",
    "CircleTool",
    "CoincidentConstraintTool",
    "ConstructionTool",
    "DeleteTool",
    "DiameterConstraintTool",
    "DistanceConstraintTool",
    "EqualConstraintTool",
    "FilletTool",
    "GridTool",
    "FillTool",
    "HorizontalConstraintTool",
    "KEY_TO_TOOL",
    "PerpendicularConstraintTool",
    "RadiusConstraintTool",
    "RectangleTool",
    "RoundedRectTool",
    "SelectTool",
    "SketchTool",
    "SketcherKey",
    "STUDIO_SHORTCUTS",
    "StraightenTool",
    "SymmetryConstraintTool",
    "TangentConstraintTool",
    "TextBoxTool",
    "TOOL_REGISTRY",
    "VerticalConstraintTool",
    "WaypointSharpTool",
    "WaypointSmoothTool",
    "WaypointSymmetricTool",
]
