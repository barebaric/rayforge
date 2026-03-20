from .angle_constraint_tool import AngleConstraintTool
from .arc_tool import ArcTool
from .aspect_ratio_constraint_tool import AspectRatioConstraintTool
from .base import SketchTool, SketcherKey
from .bezier_tool import BezierTool
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
from .horizontal_constraint_tool import HorizontalConstraintTool
from .line_tool import LineTool
from .perpendicular_constraint_tool import PerpendicularConstraintTool
from .radius_constraint_tool import RadiusConstraintTool
from .rectangle_tool import RectangleTool
from .rounded_rect_tool import RoundedRectTool
from .select_tool import SelectTool
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
    "bezier": BezierTool,
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
    "horizontal": HorizontalConstraintTool,
    "line": LineTool,
    "perpendicular": PerpendicularConstraintTool,
    "radius": RadiusConstraintTool,
    "rectangle": RectangleTool,
    "rounded_rect": RoundedRectTool,
    "select": SelectTool,
    "symmetry": SymmetryConstraintTool,
    "tangent": TangentConstraintTool,
    "text_box": TextBoxTool,
    "vertical": VerticalConstraintTool,
    "waypoint_sharp": WaypointSharpTool,
    "waypoint_smooth": WaypointSmoothTool,
    "waypoint_symmetric": WaypointSymmetricTool,
}

__all__ = [
    "AngleConstraintTool",
    "ArcTool",
    "AspectRatioConstraintTool",
    "BezierTool",
    "ChamferTool",
    "CircleTool",
    "CoincidentConstraintTool",
    "ConstructionTool",
    "DeleteTool",
    "DiameterConstraintTool",
    "DistanceConstraintTool",
    "EqualConstraintTool",
    "FilletTool",
    "FillTool",
    "HorizontalConstraintTool",
    "LineTool",
    "PerpendicularConstraintTool",
    "RadiusConstraintTool",
    "RectangleTool",
    "RoundedRectTool",
    "SelectTool",
    "SketchTool",
    "SketcherKey",
    "SymmetryConstraintTool",
    "TangentConstraintTool",
    "TextBoxTool",
    "TOOL_REGISTRY",
    "VerticalConstraintTool",
    "WaypointSharpTool",
    "WaypointSmoothTool",
    "WaypointSymmetricTool",
]
