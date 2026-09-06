"""Capture popover for recording a mouse gesture binding."""

import logging
from gettext import gettext as _
from typing import Any, Callable

from gi.repository import Gdk, Gtk

from ..gestures.model import (
    GestureKind,
    GestureSlot,
    GestureSpec,
    normalize_modifiers,
)
from ..gestures.registry import gesture_registry

logger = logging.getLogger(__name__)


class GestureBindingPopover(Gtk.Popover):
    """
    Records a physical gesture and assigns it to a gesture slot.

    The popover shows a capture area that records the next mouse
    button click or scroll wheel event together with the modifier
    keys held down. Escape closes the popover without recording.
    """

    def __init__(
        self,
        config: Any,
        context_id: str,
        slot: GestureSlot,
        on_applied: Callable | None = None,
    ):
        super().__init__()
        self._config = config
        self._context_id = context_id
        self._slot = slot
        self._on_applied = on_applied

        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=12,
            margin_bottom=12,
            margin_start=12,
            margin_end=12,
        )
        self.set_child(box)
        box.append(self._build_capture_area())
        box.append(self._build_warning_label())
        box.append(self._build_buttons())
        self._setup_key_controller()

    def _build_capture_area(self) -> Gtk.Widget:
        frame = Gtk.Frame()
        capture = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            valign=Gtk.Align.CENTER,
            halign=Gtk.Align.CENTER,
            margin_top=24,
            margin_bottom=24,
            margin_start=18,
            margin_end=18,
        )
        frame.set_child(capture)
        capture.append(Gtk.Label.new(_("Press a mouse button now, or scroll")))
        default = self._slot.default_binding
        if default is not None:
            capture.append(
                Gtk.Label.new(_("Default: %s") % default.display_label())
            )

        click = Gtk.GestureClick.new()
        click.set_button(0)
        click.connect("pressed", self._on_capture_pressed)
        capture.add_controller(click)

        scroll = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        scroll.connect("scroll", self._on_capture_scroll)
        capture.add_controller(scroll)

        return frame

    def _build_warning_label(self) -> Gtk.Widget:
        self._warning_label = Gtk.Label(wrap=True, xalign=0.0, visible=False)
        self._warning_label.add_css_class("error")
        return self._warning_label

    def _build_buttons(self) -> Gtk.Widget:
        box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            halign=Gtk.Align.END,
            homogeneous=True,
        )
        if self._slot.allow_unassign:
            unassign = Gtk.Button(label=_("Unassign"), valign=Gtk.Align.CENTER)
            unassign.connect("clicked", self._on_unassign_clicked)
            box.append(unassign)
        reset = Gtk.Button(label=_("Reset to default"))
        reset.connect("clicked", self._on_reset_clicked)
        box.append(reset)
        return box

    def _setup_key_controller(self) -> None:
        key = Gtk.EventControllerKey.new()
        key.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key)

    def _on_key_pressed(
        self,
        controller: Gtk.EventControllerKey,
        keyval: int,
        keycode: int,
        state: Gdk.ModifierType,
    ) -> bool:
        if keyval == Gdk.KEY_Escape:
            self.popdown()
            return True
        return False

    def _on_capture_pressed(
        self,
        gesture: Gtk.GestureClick,
        n_press: int,
        x: float,
        y: float,
    ) -> None:
        spec = GestureSpec(
            GestureKind.CLICK,
            button=gesture.get_current_button(),
            modifiers=normalize_modifiers(gesture.get_current_event_state()),
        )
        self._apply(spec)

    def _on_capture_scroll(
        self,
        controller: Gtk.EventControllerScroll,
        dx: float,
        dy: float,
    ) -> None:
        self._apply(GestureSpec(GestureKind.SCROLL))

    def _apply(self, spec: GestureSpec) -> None:
        conflict = gesture_registry.find_binding_conflict(
            self._config,
            self._context_id,
            spec,
            exclude_slot_id=self._slot.id,
        )
        if conflict is not None:
            self._show_warning(conflict.label)
            return
        self._config.set_gesture_binding(
            self._context_id, self._slot.id, spec.to_config_string()
        )
        logger.info(
            f"Bound gesture '{self._context_id}/{self._slot.id}' to "
            f"{spec.to_config_string()}"
        )
        if self._on_applied:
            self._on_applied()
        self.popdown()

    def _show_warning(self, conflict_label: str) -> None:
        self._warning_label.set_text(
            _("Already used for: %s") % conflict_label
        )
        self._warning_label.set_visible(True)

    def _on_unassign_clicked(self, button: Gtk.Button) -> None:
        self._config.set_gesture_binding(self._context_id, self._slot.id, None)
        self.popdown()

    def _on_reset_clicked(self, button: Gtk.Button) -> None:
        self._config.reset_gesture_binding(self._context_id, self._slot.id)
        self.popdown()
