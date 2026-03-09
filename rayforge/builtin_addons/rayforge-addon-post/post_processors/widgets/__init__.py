"""
Widget classes for transformer type for use in post-processing settings.
"""

from .crop import CropTransformerSettingsWidget
from .smooth import SmoothSettingsWidget
from .multipass import MultiPassSettingsWidget
from .optimize import OptimizeSettingsWidget
from .overscan import OverscanSettingsWidget
from ..transformers import (
    CropTransformer,
    MultiPassTransformer,
    Optimize,
    OverscanTransformer,
    Smooth,
)

TRANSFORMER_WIDGETS = {
    CropTransformer: CropTransformerSettingsWidget,
    MultiPassTransformer: MultiPassSettingsWidget,
    Optimize: OptimizeSettingsWidget,
    OverscanTransformer: OverscanSettingsWidget,
    Smooth: SmoothSettingsWidget,
}
