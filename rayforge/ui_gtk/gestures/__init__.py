"""Configurable mouse gesture system."""

from .model import (
    BUTTON_MIDDLE,
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    GestureContext,
    GestureKind,
    GestureSlot,
    GestureSpec,
    normalize_modifiers,
)
from .registry import GestureRegistry, gesture_registry
from .router import GestureRouter

__all__ = [
    "BUTTON_MIDDLE",
    "BUTTON_PRIMARY",
    "BUTTON_SECONDARY",
    "GestureContext",
    "GestureKind",
    "GestureRegistry",
    "GestureRouter",
    "GestureSlot",
    "GestureSpec",
    "gesture_registry",
    "normalize_modifiers",
]
