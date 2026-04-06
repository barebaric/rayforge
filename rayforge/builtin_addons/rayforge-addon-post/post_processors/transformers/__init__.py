"""
Post-processing transformers for toolpath optimization.

This module provides all built-in transformers that can be applied
to Ops objects for various post-processing operations.
"""

from .crop_transformer import CropTransformer
from .merge_lines_transformer import MergeLinesTransformer
from .multipass_transformer import MultiPassTransformer
from .optimize_transformer import (
    Optimize,
    greedy_order_segments,
    two_opt,
    kdtree_order_segments,
)
from .overscan_transformer import OverscanTransformer
from .smooth_transformer import Smooth
from .tabs_transformer import TabOpsTransformer

__all__ = [
    "CropTransformer",
    "MergeLinesTransformer",
    "MultiPassTransformer",
    "Optimize",
    "OverscanTransformer",
    "Smooth",
    "TabOpsTransformer",
    "greedy_order_segments",
    "two_opt",
    "kdtree_order_segments",
]
