"""App-level entity state kept beside the raygeo document model.

The raygeo registry owns the document geometry, but a few per-entity
attributes rayforge manages (array bookkeeping, text edits, fill
colors, waypoint roles) have no mutation path on the raygeo model:
its Python entity objects are immutable snapshots. This module keeps
those attributes as an overlay store addressed by entity id. It is
seeded from the registry whenever a sketch is loaded or cloned and
merged back into the serialized entity dictionaries on save.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from raygeo.sketch.entities import Entity
    from raygeo.sketch.registry import EntityRegistry

import copy
from enum import Enum
from typing import TYPE_CHECKING


class WaypointType(Enum):
    """Handle role of a point for connected bezier geometry."""

    SHARP = "sharp"
    SMOOTH = "smooth"
    SYMMETRIC = "symmetric"


_TEXT_FIELDS = ("content", "font_family", "font_size", "bold", "italic")


class EntityFlags:
    """Overlay store for entity attributes rayforge owns."""

    def __init__(self) -> None:
        self.invisible: dict[int, bool] = {}
        self.array_copy: dict[int, bool] = {}
        self.waypoints: dict[int, WaypointType] = {}
        self.fill_colors: dict[int, tuple] = {}
        self.text: dict[int, dict[str, Any]] = {}
        self._registry: EntityRegistry | None = None

    def absorb(self, registry: EntityRegistry) -> None:
        """Seeds the overlay from the raygeo registry's storage."""
        self._registry = registry
        self.invisible.clear()
        self.array_copy.clear()
        self.fill_colors.clear()
        self.text.clear()
        self.waypoints.clear()
        for pid in registry.point_ids():
            kind = registry.waypoint_type(pid)
            if kind is not None and kind != WaypointType.SHARP.value:
                self.waypoints[pid] = WaypointType(kind)
        for eid in registry.entity_ids():
            entity = registry.get_entity_object(eid)
            if entity is None:
                continue
            if entity.invisible:
                self.invisible[eid] = True
            if entity.array_copy:
                self.array_copy[eid] = True
            if entity.fill_color is not None:
                self.fill_colors[eid] = tuple(entity.fill_color)
            if entity.type_name() == "text_box":
                self.text[eid] = {
                    "content": entity.content,
                    "font_family": entity.font_family,
                    "font_size": entity.font_size,
                    "bold": entity.bold,
                    "italic": entity.italic,
                }

    def _entity(self, eid: int) -> Entity | None:
        if self._registry is None:
            return None
        return self._registry.get_entity_object(eid)

    def __deepcopy__(self, memo: dict) -> EntityFlags:
        clone = EntityFlags()
        clone.invisible = dict(self.invisible)
        clone.array_copy = dict(self.array_copy)
        clone.waypoints = dict(self.waypoints)
        clone.fill_colors = dict(self.fill_colors)
        clone.text = copy.deepcopy(self.text, memo)
        return clone

    def drop_ids(self, entity_ids: set[int], point_ids: set[int]) -> None:
        """Forgets the flags of removed entities and points."""
        for eid in entity_ids:
            self.invisible.pop(eid, None)
            self.array_copy.pop(eid, None)
            self.fill_colors.pop(eid, None)
            self.text.pop(eid, None)
        for pid in point_ids:
            self.waypoints.pop(pid, None)

    def is_invisible(self, eid: int) -> bool:
        return self.invisible.get(eid, False)

    def set_invisible(self, eid: int, value: bool) -> None:
        if value:
            self.invisible[eid] = True
        else:
            self.invisible.pop(eid, None)
        entity = self._entity(eid)
        if entity is not None:
            entity.invisible = value

    def is_array_copy(self, eid: int) -> bool:
        return self.array_copy.get(eid, False)

    def set_array_copy(self, eid: int, value: bool) -> None:
        if value:
            self.array_copy[eid] = True
        else:
            self.array_copy.pop(eid, None)
        entity = self._entity(eid)
        if entity is not None:
            entity.array_copy = value

    def waypoint_type(self, pid: int) -> WaypointType:
        return self.waypoints.get(pid, WaypointType.SHARP)

    def set_waypoint_type(self, pid: int, waypoint: WaypointType) -> None:
        if waypoint is WaypointType.SHARP:
            self.waypoints.pop(pid, None)
        else:
            self.waypoints[pid] = waypoint

    def fill_color(self, eid: int) -> tuple | None:
        return self.fill_colors.get(eid)

    def set_fill_color(self, eid: int, color: tuple | None) -> None:
        if color is None:
            self.fill_colors.pop(eid, None)
        else:
            self.fill_colors[eid] = tuple(color)
        entity = self._entity(eid)
        if entity is not None:
            entity.set_fill_color(color)

    def text_properties(self, eid: int) -> dict[str, Any]:
        """The text box's content and font properties, overlay first."""
        return dict(self.text.get(eid, {}))

    def set_text_property(
        self, eid: int, key: str, value: str | float | bool
    ) -> None:
        if key not in _TEXT_FIELDS:
            raise ValueError(f"Unknown text property: {key}")
        self.text.setdefault(eid, {})[key] = value
        entity = self._entity(eid)
        if entity is None or entity.type_name() != "text_box":
            return
        if key == "content":
            entity.content = value
            return
        from raygeo.geo.shape.text import FontConfig

        fc = entity.font_config
        kwargs = {
            "family": fc.family,
            "size": fc.size,
            "bold": fc.bold,
            "italic": fc.italic,
        }
        field = {
            "font_family": "family",
            "font_size": "size",
            "bold": "bold",
            "italic": "italic",
        }[key]
        kwargs[field] = value
        entity.font_config = FontConfig(**kwargs)

    def text_content(self, eid: int, default: str = "") -> str:
        return self.text.get(eid, {}).get("content", default)

    def snapshot(self) -> dict[str, Any]:
        """Captures the overlay for undo bookkeeping."""
        return {
            "invisible": dict(self.invisible),
            "array_copy": dict(self.array_copy),
            "waypoints": {pid: wt.value for pid, wt in self.waypoints.items()},
            "fill_colors": dict(self.fill_colors),
            "text": copy.deepcopy(self.text),
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Restores an overlay captured by ``snapshot``."""
        self.invisible = dict(state.get("invisible", {}))
        self.array_copy = dict(state.get("array_copy", {}))
        self.waypoints = {
            pid: WaypointType(kind)
            for pid, kind in state.get("waypoints", {}).items()
        }
        self.fill_colors = dict(state.get("fill_colors", {}))
        self.text = copy.deepcopy(state.get("text", {}))

    def apply_to_registry_dict(self, registry_dict: dict[str, Any]) -> None:
        """Merges the overlay into a raygeo registry dictionary for
        serialization. Entities absent from the dictionaries are
        skipped."""
        for point_dict in registry_dict.get("points", []):
            waypoint = self.waypoints.get(point_dict.get("id"))
            if waypoint is not None:
                point_dict["waypoint_type"] = waypoint.value
        for entity_dict in registry_dict.get("entities", []):
            eid = entity_dict.get("id")
            if eid is None:
                continue
            if self.invisible.get(eid):
                entity_dict["invisible"] = True
            if self.array_copy.get(eid):
                entity_dict["array_copy"] = True
            color = self.fill_colors.get(eid)
            if color is not None:
                entity_dict["fill_color"] = list(color)
            overlay = self.text.get(eid)
            if overlay:
                entity_dict.setdefault("content", "")
                entity_dict["content"] = overlay.get(
                    "content", entity_dict.get("content", "")
                )
                font = entity_dict.setdefault("font_config", {})
                for key in ("font_family", "font_size", "bold", "italic"):
                    if key in overlay:
                        font[key] = overlay[key]
