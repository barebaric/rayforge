from __future__ import annotations

from .base import ImageEditOp


class ImageEditRegistry:
    """Registry for image editing operations, keyed by their NAME."""

    def __init__(self):
        self._ops: dict[str, type[ImageEditOp]] = {}
        self._addon_items: dict[str, set[str]] = {}

    def register(
        self, op_class: type[ImageEditOp], addon_name: str | None = None
    ) -> None:
        name = op_class.NAME
        self._ops[name] = op_class
        if addon_name:
            self._addon_items.setdefault(addon_name, set()).add(name)

    def unregister(self, name: str) -> bool:
        if name not in self._ops:
            return False
        del self._ops[name]
        for items in self._addon_items.values():
            items.discard(name)
        return True

    def unregister_all_from_addon(self, addon_name: str) -> int:
        items = self._addon_items.pop(addon_name, set())
        count = 0
        for name in items:
            if self._ops.pop(name, None) is not None:
                count += 1
        return count

    def get(self, name: str) -> type[ImageEditOp] | None:
        return self._ops.get(name)

    def all_ops(self) -> dict[str, type[ImageEditOp]]:
        return self._ops.copy()


image_edit_registry = ImageEditRegistry()
