from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from raygeo.geo.types import Point

from ..types import EntityID
from .base import Constraint

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class RotationalConstraint(Constraint):
    """
    Links a target point (p2) to a source point (p1) through a rotation
    of `value` radians around a center point.

    Used by pattern commands to keep array copies parametrically attached
    to the template member: editing any instance propagates to the whole
    pattern through the solver. Deleting a member deletes only its own
    constraints, so the remaining instances are never redistributed.
    """

    def __init__(
        self,
        center: EntityID,
        p1: EntityID,
        p2: EntityID,
        value: str | float,
        expression: str | None = None,
        user_visible: bool = False,
    ):
        super().__init__(user_visible=user_visible)
        self.center: EntityID = center
        self.p1: EntityID = p1
        self.p2: EntityID = p2

        if expression is not None:
            self.expression = expression
            self.value = float(value)
        elif isinstance(value, str):
            self.expression = value
            self.value = 0.0
        else:
            self.expression = None
            self.value = float(value)

    @classmethod
    def get_type_key(cls) -> str:
        return "rotational"

    @staticmethod
    def get_type_name() -> str:
        """Returns to human-readable name of this constraint type."""
        return _("Rotational Pattern")

    def get_title(self) -> str:
        """Returns a human-readable title for this constraint."""
        angle_deg = math.degrees(self.value)
        return f"{self.get_type_name()} {angle_deg:.1f}°"

    def to_dict(self) -> dict[str, Any]:
        data = {
            "type": "RotationalConstraint",
            "center": self.center,
            "p1": self.p1,
            "p2": self.p2,
            "value": self.value,
            "user_visible": self.user_visible,
        }
        if self.expression:
            data["expression"] = self.expression
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RotationalConstraint:
        return cls(
            center=data["center"],
            p1=data["p1"],
            p2=data["p2"],
            value=data["value"],
            expression=data.get("expression"),
            user_visible=data.get("user_visible", False),
        )

    def error(
        self, reg: EntityRegistry, params: ParameterContext
    ) -> list[float]:
        c = reg.get_point(self.center)
        s = reg.get_point(self.p1)
        t = reg.get_point(self.p2)

        ca = math.cos(self.value)
        sa = math.sin(self.value)
        dx = s.x - c.x
        dy = s.y - c.y

        # Target position: rotate (s - c) by angle around c.
        ex = t.x - (c.x + ca * dx - sa * dy)
        ey = t.y - (c.y + sa * dx + ca * dy)
        return [ex, ey]

    def gradient(
        self, reg: EntityRegistry, params: ParameterContext
    ) -> dict[EntityID, list[Point]]:
        ca = math.cos(self.value)
        sa = math.sin(self.value)

        # Row 0 (x residual): t.x - c.x - ca*(s.x - c.x) + sa*(s.y - c.y)
        # Row 1 (y residual): t.y - c.y - sa*(s.x - c.x) - ca*(s.y - c.y)
        return {
            self.p2: [(1.0, 0.0), (0.0, 1.0)],
            self.p1: [(-ca, sa), (-sa, -ca)],
            self.center: [(-1.0 + ca, -sa), (sa, -1.0 + ca)],
        }
