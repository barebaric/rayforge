import sys
from gi.repository import Gdk


def primary_modifier_mask() -> Gdk.ModifierType:
    if sys.platform == "darwin":
        return (
            Gdk.ModifierType.META_MASK
            | Gdk.ModifierType.SUPER_MASK
            | Gdk.ModifierType.MOD2_MASK
        )
    return Gdk.ModifierType.CONTROL_MASK


def is_primary_modifier(state: Gdk.ModifierType) -> bool:
    return bool(state & primary_modifier_mask())


def is_primary_keyval(keyval: int) -> bool:
    if sys.platform == "darwin":
        command_key = getattr(Gdk, "KEY_Command", None)
        primary_keys = [
            Gdk.KEY_Meta_L,
            Gdk.KEY_Meta_R,
            Gdk.KEY_Super_L,
            Gdk.KEY_Super_R,
        ]
        if command_key is not None:
            primary_keys.append(command_key)
        return keyval in primary_keys
    return keyval in (Gdk.KEY_Control_L, Gdk.KEY_Control_R)


def primary_accel() -> str:
    if sys.platform == "darwin":
        return "<Meta>"
    return "<Primary>"
