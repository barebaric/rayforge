import sys

from gi.repository import Gdk


if sys.platform == "darwin":
    PRIMARY_MODIFIER_MASK = Gdk.ModifierType(0)
    for mask_name in ("META_MASK", "SUPER_MASK", "MOD2_MASK"):
        mask = getattr(Gdk.ModifierType, mask_name, None)
        if mask is not None:
            PRIMARY_MODIFIER_MASK |= mask
    if PRIMARY_MODIFIER_MASK == 0:
        PRIMARY_MODIFIER_MASK = Gdk.ModifierType.CONTROL_MASK
    PRIMARY_ACCEL = "<Meta>"
else:
    PRIMARY_MODIFIER_MASK = Gdk.ModifierType.CONTROL_MASK
    PRIMARY_ACCEL = "<Primary>"


def is_primary_modifier(state: Gdk.ModifierType) -> bool:
    return bool(state & PRIMARY_MODIFIER_MASK)


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
