"""Data model for the configurable mouse gesture system."""

from dataclasses import dataclass
from enum import Enum
from gettext import gettext as _

from gi.repository import Gdk

BUTTON_PRIMARY = Gdk.BUTTON_PRIMARY
BUTTON_MIDDLE = Gdk.BUTTON_MIDDLE
BUTTON_SECONDARY = Gdk.BUTTON_SECONDARY

_BUTTON_NAMES = {
    BUTTON_PRIMARY: "primary",
    BUTTON_MIDDLE: "middle",
    BUTTON_SECONDARY: "secondary",
}
_BUTTONS_BY_NAME = {name: num for num, name in _BUTTON_NAMES.items()}

_MATCHED_MODIFIERS = (
    Gdk.ModifierType.SHIFT_MASK,
    Gdk.ModifierType.CONTROL_MASK,
    Gdk.ModifierType.ALT_MASK,
)

_MODIFIER_NAMES = {
    Gdk.ModifierType.SHIFT_MASK: "shift",
    Gdk.ModifierType.CONTROL_MASK: "ctrl",
    Gdk.ModifierType.ALT_MASK: "alt",
}
_MODIFIERS_BY_NAME = {name: mod for mod, name in _MODIFIER_NAMES.items()}


class GestureKind(Enum):
    """The type of physical input that makes up a gesture."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    DRAG = "drag"
    SCROLL = "scroll"


def normalize_modifiers(state) -> Gdk.ModifierType:
    """
    Reduces a raw event modifier mask to the modifiers that take part
    in gesture matching (shift, ctrl, alt).
    """
    result = 0
    for mod in _MATCHED_MODIFIERS:
        if state & mod:
            result |= mod
    return Gdk.ModifierType(result)


@dataclass(frozen=True)
class GestureSpec:
    """
    Identifies a physical mouse gesture.

    A gesture is a combination of an input kind (click, drag, scroll),
    a mouse button, and a modifier key mask. Instances are serialized
    to plain strings in the config, e.g. ``"drag+shift+middle"``.
    """

    kind: GestureKind
    button: int | None = None
    modifiers: Gdk.ModifierType = Gdk.ModifierType(0)

    def __post_init__(self):
        if self.kind is GestureKind.SCROLL:
            object.__setattr__(self, "button", None)
            object.__setattr__(self, "modifiers", Gdk.ModifierType(0))

    def to_config_string(self) -> str:
        parts = [self.kind.value]
        parts.extend(self._modifier_names())
        if self.button is not None:
            parts.append(_BUTTON_NAMES.get(self.button, str(self.button)))
        return "+".join(parts)

    @classmethod
    def from_config_string(cls, value: str) -> "GestureSpec":
        """
        Parses a gesture spec string, raising ValueError on malformed
        input.
        """
        tokens = value.strip().lower().split("+")
        if not tokens or not tokens[0]:
            raise ValueError(f"Empty gesture spec: {value!r}")
        try:
            kind = GestureKind(tokens[0])
        except ValueError:
            raise ValueError(f"Unknown gesture kind in {value!r}")
        modifiers = 0
        button: int | None = None
        for token in tokens[1:]:
            if token in _MODIFIERS_BY_NAME:
                modifiers |= _MODIFIERS_BY_NAME[token]
            elif token in _BUTTONS_BY_NAME:
                if button is not None:
                    raise ValueError(
                        f"Multiple buttons in gesture spec {value!r}"
                    )
                button = _BUTTONS_BY_NAME[token]
            elif token.isdigit() and int(token) in _BUTTON_NAMES:
                button = int(token)
            else:
                raise ValueError(
                    f"Unknown token {token!r} in gesture spec {value!r}"
                )
        if kind is GestureKind.SCROLL:
            if button is not None or modifiers:
                raise ValueError(
                    f"Scroll gesture must not carry a button or "
                    f"modifiers: {value!r}"
                )
        elif button is None:
            raise ValueError(f"Missing button in gesture spec {value!r}")
        return cls(
            kind=kind,
            button=button,
            modifiers=Gdk.ModifierType(modifiers),
        )

    def _modifier_names(self) -> list[str]:
        return [
            name
            for mod, name in _MODIFIER_NAMES.items()
            if self.modifiers & mod
        ]

    def display_label(self) -> str:
        """Returns a human-readable label for this gesture."""
        kind_labels = {
            GestureKind.CLICK: _("Click"),
            GestureKind.DOUBLE_CLICK: _("Double-click"),
            GestureKind.DRAG: _("Drag"),
            GestureKind.SCROLL: _("Scroll"),
        }
        if self.kind is GestureKind.SCROLL:
            return kind_labels[self.kind]
        button_labels = {
            BUTTON_PRIMARY: _("Left"),
            BUTTON_MIDDLE: _("Middle"),
            BUTTON_SECONDARY: _("Right"),
        }
        modifier_labels = {
            Gdk.ModifierType.SHIFT_MASK: _("Shift"),
            Gdk.ModifierType.CONTROL_MASK: _("Ctrl"),
            Gdk.ModifierType.ALT_MASK: _("Alt"),
        }
        parts = [
            modifier_labels[mod]
            for mod in _MATCHED_MODIFIERS
            if self.modifiers & mod
        ]
        parts.append(button_labels.get(self.button, str(self.button)))
        return f"{'+'.join(parts)} {kind_labels[self.kind]}"


@dataclass(frozen=True)
class GestureSlot:
    """
    A rebindable gesture "meaning" within a gesture context.

    A slot is identified by ``context_id`` + ``id``. It carries the
    metadata needed by the settings UI and the default binding that
    applies while the user has not customized it.
    """

    id: str
    context_id: str
    label: str
    description: str = ""
    continuous: bool = False
    default_binding: GestureSpec | None = None
    allow_unassign: bool = True


@dataclass(frozen=True)
class GestureContext:
    """
    A named interaction area that owns gesture slots, e.g. the 2D
    canvas or the 3D canvas. Addons may register additional contexts
    for their own views.
    """

    id: str
    label: str
    description: str = ""
