"""Dispatches raw GTK input to the configured gesture bindings."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gi.repository import Gtk

from .builtin import register_builtin_contexts
from .model import GestureKind, GestureSpec, normalize_modifiers
from .registry import gesture_registry

logger = logging.getLogger(__name__)

register_builtin_contexts()


@dataclass
class DragHandlers:
    """Callbacks invoked over the lifetime of a continuous drag."""

    begin: Callable | None = None
    update: Callable | None = None
    end: Callable | None = None


class GestureRouter:
    """
    Routes the raw mouse input of a widget through the gesture
    bindings configured for a gesture context.

    The router attaches one any-button drag gesture, one any-button
    click gesture, and a scroll event controller to the widget. On
    each input it resolves the physical gesture (button + modifiers)
    against the bindings registered for its context:

    - On a match, the router claims the input sequence and forwards
      the events to the handlers registered for the matching slot.
    - Without a match, the sequence is denied so that any other
      gestures on the widget (e.g. element selection on a canvas)
      handle it unchanged.
    """

    def __init__(
        self,
        context_id: str,
        widget: Gtk.Widget,
        config_provider: Callable[[], Any] | None = None,
    ):
        self.context_id = context_id
        self._widget = widget
        self._config_provider = config_provider or self._default_config
        self._drag_handlers: dict[str, DragHandlers] = {}
        self._click_handlers: dict[str, Callable] = {}
        self._scroll_handlers: dict[str, Callable] = {}
        self._active_drag_slot: str | None = None
        self._setup_controllers()

    @staticmethod
    def _default_config() -> Any:
        from ...context import get_context

        return get_context().config

    def _setup_controllers(self) -> None:
        self._drag_gesture = Gtk.GestureDrag.new()
        self._drag_gesture.set_button(0)
        self._drag_gesture.connect("drag-begin", self._on_drag_begin)
        self._drag_gesture.connect("drag-update", self._on_drag_update)
        self._drag_gesture.connect("drag-end", self._on_drag_end)
        self._widget.add_controller(self._drag_gesture)

        self._click_gesture = Gtk.GestureClick.new()
        self._click_gesture.set_button(0)
        self._click_gesture.connect("pressed", self._on_click_pressed)
        self._widget.add_controller(self._click_gesture)

        self._scroll_controller = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        self._scroll_controller.connect("scroll", self._on_scroll)
        self._widget.add_controller(self._scroll_controller)

    def register_drag(
        self,
        slot_id: str,
        begin: Callable | None = None,
        update: Callable | None = None,
        end: Callable | None = None,
    ) -> None:
        """
        Registers the handlers for a continuous drag slot. The
        callbacks use the ``Gtk.GestureDrag`` signal signatures:
        ``begin(gesture, x, y)``, ``update(gesture, dx, dy)`` and
        ``end(gesture, dx, dy)``.
        """
        self._drag_handlers[slot_id] = DragHandlers(
            begin=begin, update=update, end=end
        )

    def register_click(self, slot_id: str, invoke: Callable) -> None:
        """
        Registers the handler for a discrete click slot with the
        signature ``invoke(gesture, n_press, x, y)``.
        """
        self._click_handlers[slot_id] = invoke

    def register_scroll(self, slot_id: str, scroll: Callable) -> None:
        """
        Registers the handler for a scroll slot with the signature
        ``scroll(controller, dx, dy)``.
        """
        self._scroll_handlers[slot_id] = scroll

    def _resolve_slot(
        self, spec: GestureSpec, handler_keys: dict
    ) -> str | None:
        config = self._config_provider()
        for slot in gesture_registry.get_slots(self.context_id):
            if slot.id not in handler_keys:
                continue
            binding = gesture_registry.resolve_binding(
                config, slot.context_id, slot.id
            )
            if binding == spec:
                return slot.id
        return None

    def _on_drag_begin(self, gesture, x: float, y: float):
        spec = GestureSpec(
            GestureKind.DRAG,
            button=gesture.get_current_button(),
            modifiers=normalize_modifiers(gesture.get_current_event_state()),
        )
        slot_id = self._resolve_slot(spec, self._drag_handlers)
        if slot_id is None:
            self._active_drag_slot = None
            gesture.set_state(Gtk.EventSequenceState.DENIED)
            return
        logger.debug(f"Gesture '{self.context_id}/{slot_id}' drag begin")
        self._active_drag_slot = slot_id
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        handler = self._drag_handlers[slot_id]
        if handler.begin:
            handler.begin(gesture, x, y)

    def _on_drag_update(self, gesture, offset_x: float, offset_y: float):
        slot_id = self._active_drag_slot
        if slot_id is None:
            return
        handler = self._drag_handlers[slot_id]
        if handler.update:
            handler.update(gesture, offset_x, offset_y)

    def _on_drag_end(self, gesture, offset_x: float, offset_y: float):
        slot_id = self._active_drag_slot
        self._active_drag_slot = None
        if slot_id is None:
            return
        logger.debug(f"Gesture '{self.context_id}/{slot_id}' drag end")
        handler = self._drag_handlers[slot_id]
        if handler.end:
            handler.end(gesture, offset_x, offset_y)

    def _on_click_pressed(self, gesture, n_press: int, x: float, y: float):
        spec = GestureSpec(
            GestureKind.CLICK,
            button=gesture.get_current_button(),
            modifiers=normalize_modifiers(gesture.get_current_event_state()),
        )
        slot_id = self._resolve_slot(spec, self._click_handlers)
        if slot_id is None:
            gesture.set_state(Gtk.EventSequenceState.DENIED)
            return
        logger.debug(f"Gesture '{self.context_id}/{slot_id}' click")
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        self._click_handlers[slot_id](gesture, n_press, x, y)

    def _on_scroll(self, controller, dx: float, dy: float):
        spec = GestureSpec(
            GestureKind.SCROLL,
            modifiers=normalize_modifiers(
                controller.get_current_event_state()
            ),
        )
        slot_id = self._resolve_slot(spec, self._scroll_handlers)
        if slot_id is None:
            return
        self._scroll_handlers[slot_id](controller, dx, dy)
