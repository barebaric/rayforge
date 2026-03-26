from typing import Iterator, List, Optional, Set, Tuple, TYPE_CHECKING

from rayforge.core.geo.types import Point as GeoPoint
from ..types import SnapPoint, SnapLine, SnapLineType, DragContext
from ..engine import SnapLineProducer

if TYPE_CHECKING:
    from ...registry import EntityRegistry


class EquidistantLinesProducer(SnapLineProducer):
    def __init__(
        self,
        spacing_tolerance: float = 0.5,
        max_spacing: float = 100.0,
        include_construction: bool = True,
    ) -> None:
        self._spacing_tolerance: float = spacing_tolerance
        self._max_spacing: float = max_spacing
        self._include_construction: bool = include_construction

    def produce(
        self,
        registry: "EntityRegistry",
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapLine]:
        return iter(())

    def produce_points(
        self,
        registry: "EntityRegistry",
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapPoint]:
        x, y = drag_position

        v_coords, axis_x = self._collect_aligned_coords_with_axis(
            registry, drag_context, fixed_x=x, threshold=threshold
        )
        h_coords, axis_y = self._collect_aligned_coords_with_axis(
            registry, drag_context, fixed_y=y, threshold=threshold
        )

        for snap_coord, spacing, pattern in self._find_equidistant_snaps(
            v_coords, y, threshold
        ):
            yield SnapPoint(
                x=axis_x,
                y=snap_coord,
                line_type=SnapLineType.EQUIDISTANT,
                spacing=spacing,
                is_horizontal=True,
                pattern_coords=pattern,
                axis_coord=axis_x,
            )

        for snap_coord, spacing, pattern in self._find_equidistant_snaps(
            h_coords, x, threshold
        ):
            yield SnapPoint(
                x=snap_coord,
                y=axis_y,
                line_type=SnapLineType.EQUIDISTANT,
                spacing=spacing,
                is_horizontal=False,
                pattern_coords=pattern,
                axis_coord=axis_y,
            )

    def _collect_aligned_coords_with_axis(
        self,
        registry: "EntityRegistry",
        drag_context: DragContext,
        fixed_x: Optional[float] = None,
        fixed_y: Optional[float] = None,
        threshold: float = 0.0,
    ) -> Tuple[List[float], float]:
        coords: List[float] = []
        axis_values: List[float] = []

        for point in registry.points:
            if drag_context.is_point_dragged(point.id):
                continue

            if fixed_x is not None:
                if abs(point.x - fixed_x) > threshold:
                    continue
                coords.append(point.y)
                axis_values.append(point.x)
            elif fixed_y is not None:
                if abs(point.y - fixed_y) > threshold:
                    continue
                coords.append(point.x)
                axis_values.append(point.y)

        axis_coord = (
            sum(axis_values) / len(axis_values) if axis_values else 0.0
        )
        return sorted(set(coords)), axis_coord

    def _find_equidistant_snaps(
        self,
        aligned_coords: List[float],
        drag_coord: float,
        threshold: float,
    ) -> Iterator[Tuple[float, float, Tuple[float, ...]]]:
        if len(aligned_coords) < 2:
            return

        seen_snaps: Set[float] = set()

        spacings: Set[float] = set()
        for i in range(len(aligned_coords) - 1):
            spacing = aligned_coords[i + 1] - aligned_coords[i]
            if spacing > 1e-6 and spacing <= self._max_spacing:
                spacings.add(round(spacing, 6))

        for spacing in spacings:
            for base_coord in aligned_coords:
                n = round((drag_coord - base_coord) / spacing)
                snap_coord = base_coord + n * spacing

                if snap_coord in seen_snaps:
                    continue
                if abs(snap_coord - drag_coord) > threshold:
                    continue

                pattern_coords = self._build_pattern(
                    aligned_coords, snap_coord, spacing
                )

                if len(pattern_coords) >= 3:
                    seen_snaps.add(snap_coord)
                    yield (snap_coord, spacing, pattern_coords)

    def _build_pattern(
        self,
        aligned_coords: List[float],
        snap_coord: float,
        spacing: float,
    ) -> Tuple[float, ...]:
        all_coords: Set[float] = set(aligned_coords)
        all_coords.add(snap_coord)

        pattern: List[float] = []

        coord = snap_coord
        while any(
            abs(coord - c) < self._spacing_tolerance for c in all_coords
        ):
            pattern.append(coord)
            coord -= spacing
            if coord < min(all_coords) - spacing:
                break

        coord = snap_coord + spacing
        while any(
            abs(coord - c) < self._spacing_tolerance for c in all_coords
        ):
            pattern.append(coord)
            coord += spacing
            if coord > max(all_coords) + spacing:
                break

        return tuple(sorted(pattern))
