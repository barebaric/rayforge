"""
Widget classes for transformer type for use in post-processing settings.
"""

from .crop_widget import CropTransformerSettingsWidget
from .lead_in_out_widget import LeadInOutSettingsWidget
from .merge_lines_widget import MergeLinesSettingsWidget
from .multipass_widget import MultiPassSettingsWidget
from .optimize_widget import OptimizeSettingsWidget
from .overscan_widget import OverscanSettingsWidget
from .smooth_widget import SmoothSettingsWidget
from ..transformers import (
    CropTransformer,
    LeadInOutTransformer,
    MergeLinesTransformer,
    MultiPassTransformer,
    Optimize,
    OverscanTransformer,
    Smooth,
)

TRANSFORMER_WIDGETS = {
    CropTransformer: CropTransformerSettingsWidget,
    LeadInOutTransformer: LeadInOutSettingsWidget,
    MergeLinesTransformer: MergeLinesSettingsWidget,
    MultiPassTransformer: MultiPassSettingsWidget,
    Optimize: OptimizeSettingsWidget,
    OverscanTransformer: OverscanSettingsWidget,
    Smooth: SmoothSettingsWidget,
}
