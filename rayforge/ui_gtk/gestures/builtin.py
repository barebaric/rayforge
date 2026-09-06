"""Registers the gesture contexts and slots provided by core."""

from gettext import gettext as _

from gi.repository import Gdk

from .model import (
    BUTTON_MIDDLE,
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    GestureContext,
    GestureKind,
    GestureSlot,
    GestureSpec,
)
from .registry import gesture_registry

CANVAS2D_CONTEXT = GestureContext(
    id="canvas2d",
    label=_("2D Canvas"),
    description=_("Navigation gestures on the 2D work surface."),
)

CANVAS3D_CONTEXT = GestureContext(
    id="canvas3d",
    label=_("3D Canvas"),
    description=_("Navigation gestures on the 3D preview."),
)


def register_builtin_contexts() -> None:
    gesture_registry.register_context(CANVAS2D_CONTEXT)
    for slot in _canvas2d_slots():
        gesture_registry.register_slot(slot)
    gesture_registry.register_context(CANVAS3D_CONTEXT)
    for slot in _canvas3d_slots():
        gesture_registry.register_slot(slot)


def _canvas2d_slots() -> list[GestureSlot]:
    return [
        GestureSlot(
            id="pan",
            context_id="canvas2d",
            label=_("Pan the view"),
            description=_(
                "Hold the mouse button down and move to pan the view."
            ),
            continuous=True,
            default_binding=GestureSpec(
                GestureKind.DRAG, button=BUTTON_MIDDLE
            ),
        ),
        GestureSlot(
            id="zoom",
            context_id="canvas2d",
            label=_("Zoom the view"),
            description=_("Use the mouse wheel to zoom."),
            continuous=True,
            default_binding=GestureSpec(GestureKind.SCROLL),
            allow_unassign=False,
        ),
        GestureSlot(
            id="context_menu",
            context_id="canvas2d",
            label=_("Open the context menu"),
            default_binding=GestureSpec(
                GestureKind.CLICK, button=BUTTON_SECONDARY
            ),
        ),
        GestureSlot(
            id="reset_view",
            context_id="canvas2d",
            label=_("Reset the view"),
            description=_("Fits the work area into the view."),
            default_binding=None,
        ),
    ]


def _canvas3d_slots() -> list[GestureSlot]:
    return [
        GestureSlot(
            id="orbit",
            context_id="canvas3d",
            label=_("Orbit the camera"),
            description=_(
                "Hold the mouse button down and move to orbit around "
                "the scene."
            ),
            continuous=True,
            default_binding=GestureSpec(
                GestureKind.DRAG, button=BUTTON_MIDDLE
            ),
        ),
        GestureSlot(
            id="pan",
            context_id="canvas3d",
            label=_("Pan the camera"),
            description=_(
                "Hold the mouse button down and move to pan the scene."
            ),
            continuous=True,
            default_binding=GestureSpec(
                GestureKind.DRAG,
                button=BUTTON_MIDDLE,
                modifiers=Gdk.ModifierType.SHIFT_MASK,
            ),
        ),
        GestureSlot(
            id="z_rotate",
            context_id="canvas3d",
            label=_("Rotate around the Z axis"),
            continuous=True,
            default_binding=GestureSpec(
                GestureKind.DRAG, button=BUTTON_PRIMARY
            ),
        ),
        GestureSlot(
            id="zoom",
            context_id="canvas3d",
            label=_("Zoom the view"),
            description=_("Use the mouse wheel to zoom."),
            continuous=True,
            default_binding=GestureSpec(GestureKind.SCROLL),
            allow_unassign=False,
        ),
    ]
