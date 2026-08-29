import logging
from collections.abc import Iterator

from raygeo.geo.types import Point as GeoPoint

from ..registry import EntityRegistry
from .spatial import SnapLineIndex
from .types import (
    DragContext,
    SnapLine,
    SnapPoint,
    SnapResult,
)

logger = logging.getLogger(__name__)


class SnapLineProducer:
    def produce(
        self,
        registry: EntityRegistry,
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapLine]:
        raise NotImplementedError

    def produce_points(
        self,
        registry: EntityRegistry,
        drag_position: GeoPoint,
        drag_context: DragContext,
        threshold: float,
    ) -> Iterator[SnapPoint]:
        return iter(())


class SnapEngine:
    DEFAULT_THRESHOLD = 5.0

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self._producers: list[SnapLineProducer] = []
        self._threshold: float = threshold
        self._index: SnapLineIndex = SnapLineIndex()
        self._last_query_pos: GeoPoint | None = None
        self._enabled: bool = True
        self._cached_fingerprint: tuple | None = None
        self._cached_drag_context_key: tuple | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def register_producer(self, producer: SnapLineProducer) -> None:
        self._producers.append(producer)

    def unregister_producer(self, producer: SnapLineProducer) -> None:
        if producer in self._producers:
            self._producers.remove(producer)

    def clear_producers(self) -> None:
        self._producers.clear()

    @staticmethod
    def _geometry_fingerprint(registry: EntityRegistry) -> tuple:
        """
        Cheap identity of the geometry the snap index was built from.

        Structural changes bump the registry version, but solving,
        drags and array syncs also move points IN PLACE without any
        version signal; only the positions themselves detect that.
        The O(n) fingerprint is compared on every query so the
        expensive producer run only happens when geometry changed.
        Positions are quantized to 6 decimals: the solver leaves
        residual noise of roughly 1e-9 on every solve, which must not
        count as a geometry change.
        """
        return (
            registry._entity_version,
            tuple((round(p.x, 6), round(p.y, 6)) for p in registry.points),
        )

    def rebuild_index(
        self,
        registry: EntityRegistry,
        drag_position: GeoPoint,
        drag_context: DragContext,
    ) -> None:
        fingerprint = self._geometry_fingerprint(registry)
        drag_key = (
            frozenset(drag_context.dragged_point_ids),
            frozenset(drag_context.dragged_entity_ids),
        )
        if (
            fingerprint == self._cached_fingerprint
            and drag_key == self._cached_drag_context_key
        ):
            return

        self._index.clear()

        for producer in self._producers:
            try:
                self._index.add_all(
                    producer.produce(
                        registry,
                        drag_position,
                        drag_context,
                        self._threshold,
                    )
                )
            except Exception as e:  # noqa: BLE001 - addon producer boundary
                logger.warning(f"SnapLineProducer error: {e}")

        self._last_query_pos = drag_position
        self._cached_fingerprint = fingerprint
        self._cached_drag_context_key = drag_key

    def _get_snap_points(
        self,
        registry: EntityRegistry,
        drag_position: GeoPoint,
        drag_context: DragContext,
    ) -> list[SnapPoint]:
        points: list[SnapPoint] = []
        for producer in self._producers:
            try:
                points.extend(
                    producer.produce_points(
                        registry,
                        drag_position,
                        drag_context,
                        self._threshold,
                    )
                )
            except Exception as e:  # noqa: BLE001 - addon producer boundary
                logger.warning(f"SnapLineProducer error: {e}")
        return points

    def query(
        self,
        registry: EntityRegistry,
        position: GeoPoint,
        drag_context: DragContext | None = None,
    ) -> SnapResult:
        if not self._enabled:
            return SnapResult.no_snap(position)

        if drag_context is None:
            drag_context = DragContext()

        self.rebuild_index(registry, position, drag_context)

        x, y = position

        snap_points = self._get_snap_points(registry, position, drag_context)
        point_result = self._find_nearest_snap_point(x, y, snap_points)
        if point_result is not None:
            snap_point, dist = point_result
            crossing_lines = self._find_crossing_lines(x, y, snap_point)
            return SnapResult.from_snap_point(snap_point, dist, crossing_lines)

        best_h, best_v = self._find_best_lines_for_both_axes(x, y)
        if best_h is not None or best_v is not None:
            snap_x = best_v.coordinate if best_v else x
            snap_y = best_h.coordinate if best_h else y
            snap_lines: list[SnapLine] = []
            if best_h:
                snap_lines.append(best_h)
            if best_v:
                snap_lines.append(best_v)
            dist = max(
                best_h.distance_to(x, y) if best_h else 0,
                best_v.distance_to(x, y) if best_v else 0,
            )
            return SnapResult(
                snapped=True,
                position=(snap_x, snap_y),
                snap_lines=snap_lines,
                distance=dist,
            )

        return SnapResult.no_snap(position)

    def is_snap_result_current(self, registry: EntityRegistry) -> bool:
        """
        True if the registry geometry is unchanged since the last
        query. During a drag, the solve and the array sync move
        geometry AFTER the snap query but BEFORE the frame is drawn:
        a result computed against the older geometry draws snap lines
        detached from any geometry, which flash up during fast drags.
        Callers must not render such a stale result.
        """
        return self._cached_fingerprint == self._geometry_fingerprint(registry)

    def _find_best_lines_for_both_axes(
        self, x: float, y: float
    ) -> tuple[SnapLine | None, SnapLine | None]:
        best_h: SnapLine | None = None
        best_h_dist: float = self._threshold
        best_v: SnapLine | None = None
        best_v_dist: float = self._threshold

        for indexed in self._index._horizontal:
            if indexed.snap_line is None:
                continue
            dist = abs(y - indexed.coordinate)
            if dist < best_h_dist:
                best_h_dist = dist
                best_h = indexed.snap_line

        for indexed in self._index._vertical:
            if indexed.snap_line is None:
                continue
            dist = abs(x - indexed.coordinate)
            if dist < best_v_dist:
                best_v_dist = dist
                best_v = indexed.snap_line

        return (best_h, best_v)

    def _find_nearest_snap_point(
        self, x: float, y: float, snap_points: list[SnapPoint]
    ) -> tuple[SnapPoint, float] | None:
        best_point: SnapPoint | None = None
        best_dist: float = self._threshold
        best_priority: int = -1

        for sp in snap_points:
            dx = x - sp.x
            dy = y - sp.y
            dist = (dx * dx + dy * dy) ** 0.5
            priority = sp.line_type.priority

            if dist > self._threshold:
                continue

            if priority > best_priority or (
                priority == best_priority and dist < best_dist
            ):
                best_dist = dist
                best_point = sp
                best_priority = priority

        if best_point is not None:
            return (best_point, best_dist)
        return None

    def _find_crossing_lines(
        self, x: float, y: float, snap_point: SnapPoint
    ) -> list[SnapLine]:
        crossing: list[SnapLine] = []
        for sl in self._get_all_lines():
            if (
                sl.is_horizontal
                and abs(sl.coordinate - snap_point.y) < 1e-6
                or (
                    not sl.is_horizontal
                    and abs(sl.coordinate - snap_point.x) < 1e-6
                )
            ):
                crossing.append(sl)
        return crossing

    def _get_all_lines(self) -> list[SnapLine]:
        lines: list[SnapLine] = []
        for indexed in self._index._horizontal:
            if indexed.snap_line is not None:
                lines.append(indexed.snap_line)
        for indexed in self._index._vertical:
            if indexed.snap_line is not None:
                lines.append(indexed.snap_line)
        return lines

    def get_visible_snap_lines(
        self,
        registry: EntityRegistry,
        position: GeoPoint,
        drag_context: DragContext | None = None,
    ) -> list[SnapLine]:
        if not self._enabled:
            return []

        if drag_context is None:
            drag_context = DragContext()

        self.rebuild_index(registry, position, drag_context)

        return self._get_all_lines()
