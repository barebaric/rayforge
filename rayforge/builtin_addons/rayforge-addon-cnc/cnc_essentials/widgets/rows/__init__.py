"""CNC-domain row widgets."""

from .area_tolerance_row import AreaToleranceRow
from .depth_per_pass_row import DepthPerPassRow
from .max_deflection_row import MaxDeflectionRow
from .plunge_speed_row import PlungeSpeedRow
from .safe_z_row import SafeZRow
from .spindle_rpm_row import SpindleRpmRow
from .step_length_row import StepLengthRow
from .step_over_row import StepOverRow
from .target_depth_row import TargetDepthRow
from .tool_diameter_row import ToolDiameterRow
from .wall_margin_row import WallMarginRow

__all__ = [
    "AreaToleranceRow",
    "DepthPerPassRow",
    "MaxDeflectionRow",
    "PlungeSpeedRow",
    "SafeZRow",
    "SpindleRpmRow",
    "StepLengthRow",
    "StepOverRow",
    "TargetDepthRow",
    "ToolDiameterRow",
    "WallMarginRow",
]
