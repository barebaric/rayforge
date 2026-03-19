from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional

from ..entities import Bezier, Point
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..registry import EntityRegistry
    from ..sketch import Sketch


class BezierPreviewState(PreviewState):
    """Preview state for bezier tool's multi-click workflow."""

    def __init__(
        self,
        start_id: int,
        start_temp: bool,
        cp1_id: Optional[int] = None,
        cp1_temp: bool = False,
        cp2_id: Optional[int] = None,
        cp2_temp: bool = False,
        end_id: Optional[int] = None,
        end_temp: bool = False,
        temp_entity_id: Optional[int] = None,
    ):
        self.start_id = start_id
        self.start_temp = start_temp
        self.cp1_id = cp1_id
        self.cp1_temp = cp1_temp
        self.cp2_id = cp2_id
        self.cp2_temp = cp2_temp
        self.end_id = end_id
        self.end_temp = end_temp
        self.temp_entity_id = temp_entity_id

    def get_preview_point_ids(self) -> set[int]:
        result: set[int] = set()
        if self.cp1_temp and self.cp1_id is not None:
            result.add(self.cp1_id)
        if self.cp2_temp and self.cp2_id is not None:
            result.add(self.cp2_id)
        if self.end_temp and self.end_id is not None:
            result.add(self.end_id)
        return result

    @property
    def has_cp1(self) -> bool:
        return self.cp1_id is not None

    @property
    def has_cp2(self) -> bool:
        return self.cp2_id is not None

    def get_dimensions(
        self, registry: "EntityRegistry"
    ) -> List[DimensionData]:
        return []


class BezierCommand(SketchChangeCommand):
    """A command to create a cubic bezier curve."""

    def __init__(
        self,
        sketch: Sketch,
        start_id: int,
        cp1_id: int,
        cp2_id: int,
        end_pos: tuple[float, float],
        end_pid: Optional[int] = None,
        is_start_temp: bool = False,
        is_cp1_temp: bool = False,
        is_cp2_temp: bool = False,
    ):
        super().__init__(sketch, _("Add Bezier"))
        self.start_id = start_id
        self.cp1_id = cp1_id
        self.cp2_id = cp2_id
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_start_temp = is_start_temp
        self.is_cp1_temp = is_cp1_temp
        self.is_cp2_temp = is_cp2_temp
        self.add_cmd: Optional[AddItemsCommand] = None
        self._committed_end_id: Optional[int] = None

    @property
    def committed_end_id(self) -> Optional[int]:
        return self._committed_end_id

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: Optional[int] = None,
        **kwargs,
    ) -> BezierPreviewState:
        if snapped_pid is not None:
            start_id = snapped_pid
            start_temp = False
        else:
            start_id = registry.add_point(x, y)
            start_temp = True

        return BezierPreviewState(
            start_id=start_id,
            start_temp=start_temp,
        )

    @staticmethod
    def update_preview(
        registry: EntityRegistry,
        preview_state: PreviewState,
        x: float,
        y: float,
    ) -> None:
        if not isinstance(preview_state, BezierPreviewState):
            raise AttributeError("Expected BezierPreviewState")

        if preview_state.cp1_id is None:
            return

        try:
            registry.get_point(preview_state.cp1_id)
        except IndexError:
            return

        if preview_state.end_id is not None:
            end = registry.get_point(preview_state.end_id)
            end.x = x
            end.y = y

    @staticmethod
    def set_cp1(
        registry: EntityRegistry,
        preview_state: BezierPreviewState,
        x: float,
        y: float,
        snapped_pid: Optional[int] = None,
    ) -> None:
        if snapped_pid is not None and snapped_pid != preview_state.start_id:
            cp1_id = snapped_pid
            cp1_temp = False
        else:
            cp1_id = registry.add_point(x, y)
            cp1_temp = True

        preview_state.cp1_id = cp1_id
        preview_state.cp1_temp = cp1_temp

    @staticmethod
    def set_cp2_and_preview(
        registry: EntityRegistry,
        preview_state: BezierPreviewState,
        x: float,
        y: float,
        snapped_pid: Optional[int] = None,
    ) -> None:
        if preview_state.cp1_id is None:
            return

        if snapped_pid is not None and snapped_pid not in (
            preview_state.start_id,
            preview_state.cp1_id,
        ):
            cp2_id = snapped_pid
            cp2_temp = False
        else:
            cp2_id = registry.add_point(x, y)
            cp2_temp = True

        end_id = registry.add_point(x, y)
        end_temp = True

        temp_entity_id = registry.add_bezier(
            preview_state.start_id,
            preview_state.cp1_id,
            cp2_id,
            end_id,
        )

        preview_state.cp2_id = cp2_id
        preview_state.cp2_temp = cp2_temp
        preview_state.end_id = end_id
        preview_state.end_temp = end_temp
        preview_state.temp_entity_id = temp_entity_id

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        if not isinstance(preview_state, BezierPreviewState):
            raise AttributeError("Expected BezierPreviewState")

        if preview_state.temp_entity_id is not None:
            registry.entities = [
                e
                for e in registry.entities
                if e.id != preview_state.temp_entity_id
            ]
            registry._entity_map = {e.id: e for e in registry.entities}

        if preview_state.end_temp and preview_state.end_id is not None:
            registry.points = [
                p for p in registry.points if p.id != preview_state.end_id
            ]

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()

        registry = self.sketch.registry

        try:
            registry.get_point(self.start_id)
            registry.get_point(self.cp1_id)
            registry.get_point(self.cp2_id)
        except IndexError:
            return

        final_x, final_y = self.end_pos
        if self.end_pid is not None:
            try:
                end_p = registry.get_point(self.end_pid)
                final_x, final_y = end_p.x, end_p.y
            except IndexError:
                pass

        new_point = None
        end_pid = self.end_pid

        if end_pid is None:
            temp_id = registry._id_counter
            end_pid = temp_id
            new_point = Point(temp_id, final_x, final_y)

        temp_bezier_id = registry._id_counter + (1 if new_point else 0)
        new_bezier = Bezier(
            temp_bezier_id,
            self.start_id,
            self.cp1_id,
            self.cp2_id,
            end_pid,
        )

        points_to_add: List[Point] = [new_point] if new_point else []

        if self.is_start_temp:
            try:
                p = registry.get_point(self.start_id)
                registry.points.remove(p)
                points_to_add.append(p)
            except (IndexError, ValueError):
                pass

        if self.is_cp1_temp:
            try:
                p = registry.get_point(self.cp1_id)
                registry.points.remove(p)
                points_to_add.append(p)
            except (IndexError, ValueError):
                pass

        if self.is_cp2_temp:
            try:
                p = registry.get_point(self.cp2_id)
                registry.points.remove(p)
                points_to_add.append(p)
            except (IndexError, ValueError):
                pass

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=points_to_add,
            entities=[new_bezier],
            constraints=[],
        )
        self.add_cmd._do_execute()
        self._committed_end_id = end_pid

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
