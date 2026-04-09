"""
Backend entry point for post_processors addon.

Registers post-processing transformers with the main application.
"""

import gettext
from pathlib import Path

from rayforge.core.hooks import hookimpl
from .transformers import (
    CropTransformer,
    LeadInOutTransformer,
    MergeLinesTransformer,
    MultiPassTransformer,
    Optimize,
    OverscanTransformer,
    Smooth,
    TabOpsTransformer,
)

_localedir = Path(__file__).parent.parent / "locale"
_t = gettext.translation(
    "post_processors", localedir=_localedir, fallback=True
)
_ = _t.gettext

ADDON_NAME = "post_processors"


@hookimpl
def register_transformers(transformer_registry):
    """Register transformers with the transformer registry."""
    transformer_registry.register(CropTransformer, addon_name=ADDON_NAME)
    transformer_registry.register(LeadInOutTransformer, addon_name=ADDON_NAME)
    transformer_registry.register(MergeLinesTransformer, addon_name=ADDON_NAME)
    transformer_registry.register(MultiPassTransformer, addon_name=ADDON_NAME)
    transformer_registry.register(Optimize, addon_name=ADDON_NAME)
    transformer_registry.register(OverscanTransformer, addon_name=ADDON_NAME)
    transformer_registry.register(Smooth, addon_name=ADDON_NAME)
    transformer_registry.register(TabOpsTransformer, addon_name=ADDON_NAME)
