from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..entities import Entity


class PlacementKind(Enum):
    ROTATION = auto()
    TRANSLATION = auto()


@dataclass(frozen=True)
class InstancePlacement:
    """Describes how one pattern instance is derived from the template."""

    kind: PlacementKind
    angle: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)
    delta: tuple[float, float] = (0.0, 0.0)

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        if self.kind == PlacementKind.ROTATION:
            ca = math.cos(self.angle)
            sa = math.sin(self.angle)
            dx = x - self.center[0]
            dy = y - self.center[1]
            return (
                self.center[0] + ca * dx - sa * dy,
                self.center[1] + sa * dx + ca * dy,
            )
        return (x + self.delta[0], y + self.delta[1])

    def transform_offset(self, dx: float, dy: float) -> tuple[float, float]:
        """Transforms a relative offset (e.g. a bezier control point)."""
        if self.kind == PlacementKind.ROTATION:
            ca = math.cos(self.angle)
            sa = math.sin(self.angle)
            return (ca * dx - sa * dy, sa * dx + ca * dy)
        return (dx, dy)


class PatternStrategy(ABC):
    """
    Base class for sketch pattern strategies.

    A strategy computes the per-instance placements for a pattern and the
    optional "master" construction geometry that carries the pattern's
    definition (e.g. the guide circle of a circular array). Adding a new
    array type (e.g. linear) means adding a strategy subclass plus its
    parameter dataclass; commands and tools stay generic.
    """

    #: Whether the pattern is defined relative to an explicit center point.
    needs_center_point: ClassVar[bool] = False

    def __init__(self, params: Any):
        self.params = params

    @abstractmethod
    def calculate_placements(
        self, seed_center: tuple[float, float]
    ) -> list[InstancePlacement]:
        """
        Returns placements for instances 1..N-1 (instance 0 is the
        template member).

        Args:
            seed_center: Center of the template geometry's bounding box.
        """

    def create_master_geometry(
        self,
        center_pid: int | None,
        radius_pt_pid: int | None,
    ) -> tuple[list[Any], list[Entity], list[Constraint]]:
        """
        Returns the master construction geometry (points, entities,
        constraints) that visualizes and carries the pattern definition.
        Instances are equal static copies; only the master is special.
        """
        return [], [], []

    def build_linkage_constraints(
        self,
        instances: list[tuple[int, dict[int, int]]],
        center_pid: int | None,
    ) -> list[Constraint]:
        """
        Returns constraints tying each instance's cloned points back to
        their template points, so editing any member propagates to the
        whole pattern while deleting a member affects only itself.

        Args:
            instances: One (slot, {template_point_id: copy_point_id})
                pair per instance. The slot determines the instance's
                angle; slot 1 is the first placement.
            center_pid: ID of the pattern's center point, if any.
        """
        return []
