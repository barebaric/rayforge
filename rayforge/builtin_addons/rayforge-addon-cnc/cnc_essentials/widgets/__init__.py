"""
CNC Essentials UI Widgets.
"""

from .pages import (
    AdaptiveClearPage,
    HelixPlungePage,
    ProfileInnerPage,
    ProfileOuterPage,
    SlotPage,
    ToroidalClearPage,
)

ASSEMBLER_WIDGETS = {
    "adaptive_clearing": AdaptiveClearPage,
    "helix": HelixPlungePage,
    "spiral": HelixPlungePage,
    "ramp": HelixPlungePage,
    "toroidal_clear": ToroidalClearPage,
    "slot": SlotPage,
    "profile_inner": ProfileInnerPage,
    "profile_outer": ProfileOuterPage,
}

__all__ = [
    "ASSEMBLER_WIDGETS",
    "AdaptiveClearPage",
    "HelixPlungePage",
    "ProfileInnerPage",
    "ProfileOuterPage",
    "SlotPage",
    "ToroidalClearPage",
]
