from __future__ import annotations

import logging
from typing import Any

from .params import SketchArrayMode

logger = logging.getLogger(__name__)


class PatternDefinition:
    """
    Persistent definition of a sketch pattern ("master" object).

    A member is a *group* of entities (the whole shape the user picked
    as seed), not a single entity. Groups keep their identity across
    deletions: removing part of a member leaves a smaller group, never
    orphaned fragments that regenerate as broken copies.

    ``self.members`` holds ``(slot, [entity_id, ...])`` pairs; slot 0 is
    the template member, slots 1..N-1 correspond to the placements.
    """

    def __init__(
        self,
        uid: str,
        mode: SketchArrayMode,
        guide_circle_id: int,
        members: list[tuple[int, list[int]]] | None = None,
        count: int = 6,
        total_angle_deg: float = 360.0,
        rotate_copies: bool = True,
    ):
        self.uid = uid
        self.mode = mode
        self.guide_circle_id = guide_circle_id
        self.members: list[tuple[int, list[int]]] = [
            (slot, list(eids)) for slot, eids in (members or [])
        ]
        self.count = count
        self.total_angle_deg = total_angle_deg
        self.rotate_copies = rotate_copies

    def living_members(self, registry: Any) -> list[tuple[int, list[int]]]:
        """
        Returns (slot, [entity_id, ...]) pairs for members with at least
        one surviving entity, pruned to surviving entities only.
        """
        living: list[tuple[int, list[int]]] = []
        for slot, eids in self.members:
            alive = [
                eid for eid in eids if registry.get_entity(eid) is not None
            ]
            if alive:
                living.append((slot, alive))
        return sorted(living)

    def living_entity_ids(self, registry: Any) -> list[int]:
        """Flat list of all surviving member entity IDs."""
        return [
            eid
            for _slot, eids in self.living_members(registry)
            for eid in eids
        ]

    def occupied_slots(self, registry: Any) -> set[int]:
        """Returns the slot numbers of surviving members."""
        return {slot for slot, _eids in self.living_members(registry)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "mode": self.mode.value,
            "guide_circle_id": self.guide_circle_id,
            "members": [[slot, list(eids)] for slot, eids in self.members],
            "count": self.count,
            "total_angle_deg": self.total_angle_deg,
            "rotate_copies": self.rotate_copies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatternDefinition:
        members: list[tuple[int, list[int]]] = []
        for entry in data.get("members", []):
            slot, eids = entry
            members.append((int(slot), [int(eid) for eid in eids]))
        if not members and "entity_ids" in data:
            # Legacy flat format: every entity was treated as its own
            # single-entity member.
            legacy_slots = data.get(
                "entity_slots", range(len(data["entity_ids"]))
            )
            members = [
                (int(slot), [int(eid)])
                for slot, eid in zip(legacy_slots, data["entity_ids"])
            ]
        return cls(
            uid=data["uid"],
            mode=SketchArrayMode(data.get("mode", "circular")),
            guide_circle_id=data["guide_circle_id"],
            members=members,
            count=data.get("count", 6),
            total_angle_deg=data.get("total_angle_deg", 360.0),
            rotate_copies=data.get("rotate_copies", True),
        )


def find_pattern_for_entity(
    patterns: list[PatternDefinition], entity_id: int
) -> PatternDefinition | None:
    """Returns the pattern whose master circle is the given entity."""
    for pattern in patterns:
        if pattern.guide_circle_id == entity_id:
            return pattern
    return None
