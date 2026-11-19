"""
Knife Essentials UI Widgets.
"""

from .pages import DragKnifePage, KnifeStepSettingsPage, TangentialKnifePage

ASSEMBLER_WIDGETS = {
    "drag_knife": DragKnifePage,
    "tangential_knife": TangentialKnifePage,
}

__all__ = [
    "ASSEMBLER_WIDGETS",
    "DragKnifePage",
    "KnifeStepSettingsPage",
    "TangentialKnifePage",
]
