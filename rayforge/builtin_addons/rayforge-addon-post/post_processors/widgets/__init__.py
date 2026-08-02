"""
Settings group classes for transformers, used in post-processing settings.
"""

from ..transformers import (
    CropTransformer,
    LeadInOutTransformer,
    MergeLinesTransformer,
    MultiPassTransformer,
    Optimize,
    OverscanTransformer,
    Smooth,
)
from .crop_group import CropSettingsGroup
from .lead_in_out_group import LeadInOutSettingsGroup
from .merge_lines_group import MergeLinesSettingsGroup
from .multipass_group import MultiPassSettingsGroup
from .optimize_group import OptimizeSettingsGroup
from .overscan_group import OverscanSettingsGroup
from .smooth_group import SmoothSettingsGroup

TRANSFORMER_WIDGETS = {
    CropTransformer: CropSettingsGroup,
    LeadInOutTransformer: LeadInOutSettingsGroup,
    MergeLinesTransformer: MergeLinesSettingsGroup,
    MultiPassTransformer: MultiPassSettingsGroup,
    Optimize: OptimizeSettingsGroup,
    OverscanTransformer: OverscanSettingsGroup,
    Smooth: SmoothSettingsGroup,
}
