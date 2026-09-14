"""Registry for gesture contexts and slots contributed by core/addons."""

import logging
from typing import Any

from blinker import Signal

from .model import GestureContext, GestureSlot, GestureSpec

logger = logging.getLogger(__name__)

_MISSING = object()


class GestureRegistry:
    """
    Collects the gesture contexts and slots known to the application.

    Core registers the built-in canvas contexts at import time; addons
    can register their own contexts and slots. Implements the
    :class:`~rayforge.addon_mgr.addon_manager.AddonRegistry` protocol
    so that contributions are removed automatically when their addon
    is unloaded.

    Emits the :attr:`changed` signal whenever contexts or slots are
    added or removed.
    """

    def __init__(self) -> None:
        self._contexts: dict[str, tuple[GestureContext, str]] = {}
        self._slots: dict[tuple[str, str], tuple[GestureSlot, str]] = {}
        self.changed = Signal()

    def register_context(
        self, context: GestureContext, addon_name: str = ""
    ) -> None:
        """
        Register a gesture context. Re-registering an identical context
        is a no-op, so addon reloads do not produce duplicates.
        """
        existing = self._contexts.get(context.id)
        if existing and existing[0] == context:
            return
        self._contexts[context.id] = (context, addon_name)
        logger.debug(f"Registered gesture context '{context.id}'")
        self.changed.send(self)

    def register_slot(self, slot: GestureSlot, addon_name: str = "") -> None:
        """
        Register a gesture slot. Re-registering an identical slot is a
        no-op, so addon reloads do not produce duplicates.
        """
        key = (slot.context_id, slot.id)
        existing = self._slots.get(key)
        if existing and existing[0] == slot:
            return
        if slot.context_id not in self._contexts:
            logger.warning(
                f"Gesture slot '{slot.id}' registered for unknown "
                f"context '{slot.context_id}'"
            )
        self._slots[key] = (slot, addon_name)
        logger.debug(f"Registered gesture slot '{slot.context_id}/{slot.id}'")
        self.changed.send(self)

    def get_contexts(self) -> list[GestureContext]:
        """Return all registered contexts in insertion order."""
        return [context for context, _ in self._contexts.values()]

    def get_context(self, context_id: str) -> GestureContext | None:
        entry = self._contexts.get(context_id)
        return entry[0] if entry else None

    def get_slots(self, context_id: str) -> list[GestureSlot]:
        """Return all slots belonging to a context, in insertion order."""
        return [
            slot
            for (ctx, _), (slot, _) in self._slots.items()
            if ctx == context_id
        ]

    def get_slot(self, context_id: str, slot_id: str) -> GestureSlot | None:
        entry = self._slots.get((context_id, slot_id))
        return entry[0] if entry else None

    def resolve_binding(
        self, config: Any, context_id: str, slot_id: str
    ) -> GestureSpec | None:
        """
        Determines the effective gesture binding for a slot.

        The config may override the slot's default with an explicit
        spec string, or unassign the slot by storing ``None``. Stored
        values that cannot be parsed fall back to the default.
        """
        slot = self.get_slot(context_id, slot_id)
        if slot is None:
            return None
        stored = (
            getattr(config, "gesture_bindings", {})
            .get(context_id, {})
            .get(slot_id, _MISSING)
        )
        if stored is _MISSING:
            return slot.default_binding
        if stored is None:
            return None
        try:
            return GestureSpec.from_config_string(stored)
        except ValueError as e:
            logger.warning(
                f"Invalid stored gesture binding '{context_id}/{slot_id}': {e}"
            )
            return slot.default_binding

    def find_binding_conflict(
        self,
        config: Any,
        context_id: str,
        spec: GestureSpec,
        exclude_slot_id: str | None = None,
    ) -> GestureSlot | None:
        """
        Returns another slot in the context that is already bound to
        the given gesture, or None if the gesture is free.
        """
        for slot in self.get_slots(context_id):
            if slot.id == exclude_slot_id:
                continue
            binding = self.resolve_binding(config, context_id, slot.id)
            if binding == spec:
                return slot
        return None

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Remove all contexts and slots registered by the named addon.

        Returns:
            The number of removed entries.
        """
        before = len(self._contexts) + len(self._slots)
        self._contexts = {
            key: entry
            for key, entry in self._contexts.items()
            if entry[1] != addon_name
        }
        self._slots = {
            key: entry
            for key, entry in self._slots.items()
            if entry[1] != addon_name
        }
        removed = before - (len(self._contexts) + len(self._slots))
        if removed:
            logger.info(
                f"Removed {removed} gesture registrations from '{addon_name}'"
            )
            self.changed.send(self)
        return removed


gesture_registry = GestureRegistry()
