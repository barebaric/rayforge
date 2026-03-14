"""
Widget classes for transformer type for use in post-processing settings.
"""

from .crop_widget import CropTransformerSettingsWidget
from .smooth_widget import SmoothSettingsWidget
from .multipass_widget import MultiPassSettingsWidget
from .optimize_widget import OptimizeSettingsWidget
from .overscan_widget import OverscanSettingsWidget
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
