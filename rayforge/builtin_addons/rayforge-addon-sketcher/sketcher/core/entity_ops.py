"""App-level entity helpers.

The entity model itself is owned by raygeo; this module carries the
offset-plan description, the reference base-class defaults the
commands rely on, and small formatting helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raygeo.sketch.entities import Entity, OffsetPlan

if TYPE_CHECKING:
    from .registry import EntityRegistry


__all__ = ["Entity", "EntityOps", "OffsetPlan", "quantize"]


def quantize(value: float) -> float:
    return round(value, 6)


_MIN_OFFSET_RADIUS = 1e-6


class EntityOps:
    """Reference base defaults over the compiled entity classes.

    The compiled base carries state; this mixin restores the
    reference base-class *signatures* the commands rely on (mirror
    and placement transforms take the app's axis/placement objects),
    plus deepcopy support for the duplicate command.
    """

    def get_point_ids(self) -> list:
        return super().get_point_ids()  # type: ignore[attr-defined]

    def set_fill_color(self, color) -> None:
        return super().set_fill_color(color)  # type: ignore[attr-defined]

    def preview_polylines(
        self, registry: EntityRegistry, offset: float
    ) -> list:
        return super().preview_polylines(  # type: ignore[attr-defined]
            registry,
            offset,
        )

    def to_dict(self) -> dict:
        return super().to_dict()  # type: ignore[attr-defined]

    @staticmethod
    def _signature_tuple(signature) -> tuple:
        """The compiled signature is a JSON dict; the reference
        signature contract is a tuple."""
        if isinstance(signature, dict):
            return tuple(sorted(signature.items()))
        return tuple(signature)

    def mirror(self, axis) -> None:
        """Entity-internal mirror handling; the default is a no-op."""

    def transform_offsets(self, placement) -> None:
        """Entity-internal placement handling; the default is a no-op."""

    def rewrite_offsets_from(self, template, placement) -> None:
        """Re-derives copy state from a template; the default is a no-op."""

    def as_offset_item(self, sketch) -> Any:
        return None

    def geometry_signature(self, registry: EntityRegistry) -> tuple:
        """Quantized defining-point positions; subclasses extend with
        entity-internal state such as control-point offsets."""
        return tuple(
            (quantize(p.x), quantize(p.y))
            for p in (registry.get_point(pid) for pid in self.get_point_ids())
            if p is not None
        )

    def to_polylines(self, registry: EntityRegistry, tolerance: float = 0.1):
        """Samples this entity into one or more polylines in model
        coordinates. The default is the single ``to_polyline``
        polyline."""
        return [
            self.to_polyline(  # type: ignore[attr-defined]
                registry,
                tolerance,
            )
        ]

    def __deepcopy__(self, memo):
        clone = type(self).from_dict(  # type: ignore[attr-defined]
            self.to_dict()
        )
        memo[id(self)] = clone
        return clone

    def __copy__(self):
        return self.__deepcopy__({})
