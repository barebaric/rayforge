import json
import logging
from gettext import gettext as _
from pathlib import Path
from typing import Any, ClassVar

from blinker import Signal
from raygeo.geo import Arc as GeoArc
from raygeo.geo import Bezier as GeoBezier
from raygeo.geo import Geometry
from raygeo.geo import Line as GeoLine
from raygeo.geo import Move as GeoMove
from raygeo.sketch import Sketch as RaygeoSketch
from raygeo.sketch.document import Fill
from raygeo.sketch.entities import TextBox as TextBoxEntity

from rayforge.core.asset import IAsset
from rayforge.core.expression import ExpressionMap
from rayforge.core.geometry_provider import IGeometryProvider
from rayforge.core.varset import VarSet
from rayforge.image.geo_renderer import render_geometry_to_png
from rayforge.image.structures import FillRenderData, FillStyle

from .template_functions import get_template_functions
from .types import EntityID, WaypointType

DEFAULT_FILL_COLOR = (0.85, 0.85, 0.85, 0.7)
FORMAT_VERSION = 1
_DEFAULT_VARSET_TITLE = _("Sketch Parameters")
_DEFAULT_VARSET_DESCRIPTION = _(
    "Parameters that control this sketch's geometry"
)

logger = logging.getLogger(__name__)


class Sketch(IAsset, IGeometryProvider):
    """Host adapter around the raygeo sketch document.

    The document owns the registry, parameters, constraints, fills and
    arrays. This class adds the rayforge asset contract, signals,
    text-template resolution, thumbnails and array command integration.
    """

    is_addable: ClassVar[bool] = True
    asset_type_name: ClassVar[str] = "sketch"
    display_icon_name: ClassVar[str] = "sketch-edit-symbolic"
    is_reorderable: ClassVar[bool] = False
    is_draggable_to_canvas: ClassVar[bool] = True
    type_display_name: ClassVar[str] = _("Sketch")
    can_edit: ClassVar[bool] = True
    add_action: ClassVar[str | None] = "add-sketch"
    activate_action: ClassVar[str | None] = "activate-sketch"
    edit_item_action: ClassVar[str | None] = "edit-sketch-item"

    _solving: bool = False

    def __init__(self, name: str = "New Sketch") -> None:
        self._updated = Signal()
        self.document = RaygeoSketch(name)
        self.input_parameters = VarSet(
            title=_DEFAULT_VARSET_TITLE,
            description=_DEFAULT_VARSET_DESCRIPTION,
        )
        self._last_solve_values: dict[str, Any] = {}
        self._resolved_text_cache: dict[EntityID, tuple[str, str | None]] = {}
        self._solved_ctx: dict[str, Any] | None = None

    def __getattr__(self, name: str):
        document = self.__dict__.get("document")
        if document is None:
            raise AttributeError(name)
        return getattr(document, name)

    @property
    def registry(self):
        return self.document.registry

    @registry.setter
    def registry(self, value):
        self.document.registry = value

    @property
    def params(self):
        return self.document.params

    @params.setter
    def params(self, value):
        self.document.params = value

    @property
    def constraints(self):
        return self.document.constraints

    @constraints.setter
    def constraints(self, value):
        self.document.constraints = value

    @property
    def fills(self):
        return self.document.fills

    @fills.setter
    def fills(self, value):
        self.document.fills = value

    @property
    def arrays(self):
        return self.document.arrays

    @arrays.setter
    def arrays(self, value):
        self.document.arrays = value

    @property
    def uid(self) -> str:
        return self.document.uid

    @uid.setter
    def uid(self, value: str) -> None:
        self.document.uid = value

    @property
    def origin_id(self) -> EntityID:
        return self.document.origin_id

    @origin_id.setter
    def origin_id(self, value: EntityID) -> None:
        self.document.origin_id = value

    @property
    def updated(self) -> "Signal":
        return self._updated

    @property
    def name(self) -> str:
        return self.document.name

    @name.setter
    def name(self, value: str):
        if self.document.name != value:
            self.document.name = value
            self._updated.send(self)

    @property
    def hidden(self) -> bool:
        return self.document.hidden

    @hidden.setter
    def hidden(self, value: bool):
        if self.document.hidden != value:
            self.document.hidden = value
            self._updated.send(self)

    def set_hidden(self, value: bool):
        self.hidden = value

    @property
    def provider_type_name(self) -> str:
        return "sketch"

    @property
    def renderer(self):
        from ..image.renderer import SKETCH_RENDERER

        return SKETCH_RENDERER

    @property
    def is_empty(self) -> bool:
        return self.document.is_empty()

    @property
    def is_fully_constrained(self) -> bool:
        return self.document.is_fully_constrained()

    @property
    def conflicting_constraints(self) -> list:
        return self.document.conflicting_constraints()

    @property
    def has_conflicts(self) -> bool:
        return self.document.has_conflicts()

    def notify_update(self):
        if self._solving:
            return
        self._updated.send(self)

    def get_reference_point(self) -> tuple[float, float]:
        origin = self.registry.get_point(self.origin_id)
        return (origin.x, origin.y)

    def get_geometry(
        self,
        params: dict[str, Any] | None = None,
        *,
        resolved_text_cache: dict | None = None,
    ) -> tuple[Geometry, list[FillRenderData]]:
        clone = self._clone_for_geometry()
        if resolved_text_cache is not None:
            clone._resolved_text_cache = dict(resolved_text_cache)
        clone.solve(variable_overrides=params)
        if resolved_text_cache is not None:
            for key, value in resolved_text_cache.items():
                if key not in clone._resolved_text_cache:
                    clone._resolved_text_cache[key] = value
        geo = clone.to_geometry()
        fills = clone.get_fill_render_data()
        if resolved_text_cache is not None:
            resolved_text_cache.update(clone._resolved_text_cache)
        return geo, fills

    def _clone_for_geometry(self) -> "Sketch":
        clone = Sketch.__new__(Sketch)
        clone._updated = Signal()
        clone.document = self.document.clone_for_geometry()
        clone.input_parameters = self.input_parameters
        clone._last_solve_values = {}
        clone._resolved_text_cache = {}
        clone._solved_ctx = None
        return clone

    def get_thumbnail(self, size: int) -> bytes | None:
        try:
            return render_geometry_to_png(self.to_geometry(), size)
        except Exception:
            logger.exception("Failed to generate sketch thumbnail")
            return None

    def to_dict(self, include_input_values: bool = False) -> dict[str, Any]:
        data = self.document.to_dict()
        data["type"] = self.asset_type_name
        data["input_parameters"] = self.input_parameters.to_dict(
            include_value=include_input_values,
            include_metadata=False,
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sketch":
        file_version = data.get("version", 1)
        if file_version != FORMAT_VERSION:
            logger.warning(
                "Sketch file version %s differs from current version %s; "
                "loading may produce incorrect results",
                file_version,
                FORMAT_VERSION,
            )
        sketch = cls.__new__(cls)
        sketch._updated = Signal()
        sketch.document = RaygeoSketch.from_dict(data)
        sketch.input_parameters = VarSet(
            title=_DEFAULT_VARSET_TITLE,
            description=_DEFAULT_VARSET_DESCRIPTION,
        )
        if "input_parameters" in data:
            sketch.input_parameters = VarSet.from_dict(
                data["input_parameters"]
            )
            sketch.input_parameters.title = _DEFAULT_VARSET_TITLE
            sketch.input_parameters.description = _DEFAULT_VARSET_DESCRIPTION
        sketch._last_solve_values = {}
        sketch._resolved_text_cache = {}
        sketch._solved_ctx = None
        return sketch

    @classmethod
    def from_file(cls, file_path: str | Path) -> "Sketch":
        with open(file_path, "r") as file:
            data = json.load(file)
        return cls.from_dict(data)

    @classmethod
    def from_geometry(cls, geometry: Geometry) -> "Sketch":
        sketch = cls()
        if geometry.data is None or len(geometry.data) == 0:
            return sketch

        point_map: dict[tuple[float, float], EntityID] = {}

        def get_or_add_point(x: float, y: float) -> EntityID:
            key = (round(x, 6), round(y, 6))
            if key not in point_map:
                point_map[key] = sketch.add_point(x, y)
            return point_map[key]

        current_x, current_y = 0.0, 0.0
        current_pid: EntityID | None = None

        for cmd in geometry.iter_typed_commands():
            end_x, end_y = cmd.end[0], cmd.end[1]
            if isinstance(cmd, GeoMove):
                current_x, current_y = end_x, end_y
                current_pid = get_or_add_point(end_x, end_y)
            elif isinstance(cmd, GeoLine):
                if current_pid is None:
                    current_pid = get_or_add_point(current_x, current_y)
                end_pid = get_or_add_point(end_x, end_y)
                sketch.add_line(current_pid, end_pid)
                current_pid = end_pid
                current_x, current_y = end_x, end_y
            elif isinstance(cmd, GeoArc):
                if current_pid is None:
                    current_pid = get_or_add_point(current_x, current_y)
                end_pid = get_or_add_point(end_x, end_y)
                i_offset, j_offset, _ = cmd.center_offset
                center_x = current_x + i_offset
                center_y = current_y + j_offset
                center_pid = get_or_add_point(center_x, center_y)
                sketch.add_arc(
                    current_pid,
                    end_pid,
                    center_pid,
                    clockwise=cmd.clockwise,
                )
                current_pid = end_pid
                current_x, current_y = end_x, end_y
            elif isinstance(cmd, GeoBezier):
                if current_pid is None:
                    current_pid = get_or_add_point(current_x, current_y)
                cp1_x, cp1_y, _ = cmd.control1
                cp2_x, cp2_y, _ = cmd.control2
                start_pt = sketch.registry.get_point(current_pid)
                if start_pt:
                    start_pt.waypoint_type = WaypointType.SMOOTH
                end_pid = get_or_add_point(end_x, end_y)
                end_pt = sketch.registry.get_point(end_pid)
                if end_pt:
                    end_pt.waypoint_type = WaypointType.SMOOTH
                cp1_offset = (cp1_x - start_pt.x, cp1_y - start_pt.y)
                cp2_offset = (cp2_x - end_pt.x, cp2_y - end_pt.y)
                sketch.add_bezier(
                    current_pid, end_pid, cp1=cp1_offset, cp2=cp2_offset
                )
                current_pid = end_pid
                current_x, current_y = end_x, end_y
        return sketch

    def capture_undo_state(self) -> dict[str, Any]:
        return self.document.capture_undo_state()

    def apply_undo_state(self, state: dict[str, Any]) -> None:
        self.document.apply_undo_state(state)

    def _validate_and_cleanup_fills(self):
        self.document.validate_fills()

    def _find_all_closed_loops(self):
        return self.document.find_all_closed_loops()

    def move_point(self, pid: EntityID, x: float, y: float) -> bool:
        try:
            point = self.registry.get_point(pid)
        except IndexError:
            return False
        if point.fixed or point.constrained:
            return False
        point.x = x
        point.y = y
        return self.solve()

    def solve(
        self,
        extra_constraints: list | None = None,
        update_constraint_status: bool = True,
        variable_overrides: dict[str, Any] | None = None,
        excluded_constraints: set[int] | None = None,
        point_scope: set[EntityID] | None = None,
    ) -> bool:
        if self._solving:
            return True
        self._solving = True
        self._resolved_text_cache = {}
        try:
            initial_values: dict[str, Any] = {}
            if self.input_parameters:
                initial_values.update(self.input_parameters.get_values())
            result = self.document.solve(
                extra_constraints=extra_constraints,
                update_constraint_status=update_constraint_status,
                variable_overrides=variable_overrides,
                excluded_constraints=(
                    set(excluded_constraints)
                    if excluded_constraints is not None
                    else None
                ),
                point_scope=(
                    set(point_scope) if point_scope is not None else None
                ),
                initial_values=initial_values,
                template_functions=get_template_functions(),
            )
            self.sync_arrays()
            self.document.finish_solve()
            self._last_solve_values = dict(self.document.last_solve_values)
            self._solved_ctx = (
                dict(self.document.solved_ctx)
                if self.document.solved_ctx is not None
                else None
            )
        finally:
            self._solving = False
        return result

    def sync_arrays(self) -> None:
        for array in self.arrays:
            guide_sig = array.guide_signature(self.registry)
            if not guide_sig:
                continue
            template_sig = array.template_signature(self.registry)
            if array.signatures_changed(guide_sig, template_sig):
                self._reapply_array(array)
                template_sig = array.template_signature(self.registry)
            array.update_caches(guide_sig, template_sig)

    def _reapply_array(self, array_def) -> None:
        from .commands.edit_array import EditArrayCommand

        cmd = EditArrayCommand(
            self,
            array_def,
            array_def.make_strategy(self.registry),
            force_full_regen=True,
            capture_snapshot=False,
            old_frame=array_def._cached_guide_frame,
        )
        cmd.execute()

    def _resolve_text_content(self, entity: TextBoxEntity) -> str | None:
        cached = self._resolved_text_cache.get(entity.id)
        if isinstance(cached, (tuple, list)):
            source, resolved = cached
            if source == entity.content:
                return resolved

        if not entity.content:
            self._resolved_text_cache[entity.id] = ("", None)
            return None

        try:
            if self._solved_ctx is not None:
                ctx = self._solved_ctx
            else:
                solve_params = type(self.params).from_dict(
                    self.params.to_dict()
                )
                initial_values = dict(self._last_solve_values)
                if not initial_values and self.input_parameters:
                    initial_values.update(self.input_parameters.get_values())
                solve_params.evaluate_all(initial_values=initial_values)
                ctx: dict[str, Any] = dict(solve_params.get_all_values())
                ctx.update(get_template_functions())

            expr_map = ExpressionMap(ctx)
            resolved = expr_map.format(entity.content)
            self._resolved_text_cache[entity.id] = (
                entity.content,
                resolved,
            )
            return resolved
        except (KeyError, IndexError, ValueError) as e:
            logger.debug(
                "Template resolution failed for '%s': %s", entity.content, e
            )
            self._resolved_text_cache[entity.id] = (entity.content, None)
            return None

    def to_geometry(self) -> Geometry:
        resolved_text: dict[EntityID, str | None] = {}
        for entity in self.registry.entities():
            if isinstance(entity, TextBoxEntity):
                resolved_text[entity.id] = self._resolve_text_content(entity)
        return self.document.to_geometry(resolved_text)

    def _create_fill_geometry(
        self, fill: Fill, exclude_ids: set[EntityID]
    ) -> Geometry | None:
        return self.document.create_fill_geometry(fill, exclude_ids)

    def get_fill_render_data(
        self, exclude_ids: set[EntityID] | None = None
    ) -> list[FillRenderData]:
        if exclude_ids is None:
            exclude_ids = set()

        render_data = []
        for fill in self.fills:
            if not fill.boundary:
                continue
            geo = self._create_fill_geometry(fill, exclude_ids)
            if geo is not None:
                render_data.append(
                    FillRenderData(
                        geometry=geo,
                        style=fill.style,
                        color=fill.color,
                        gradient_stops=fill.gradient_stops,
                        gradient_angle=fill.gradient_angle,
                    )
                )

        for entity in self.registry.entities():
            if entity.id in exclude_ids:
                continue
            if not entity.construction and isinstance(entity, TextBoxEntity):
                resolved = self._resolve_text_content(entity)
                text_geo = entity.to_geometry(
                    self.registry, resolved_content=resolved
                )
                if text_geo:
                    color = (
                        entity.fill_color
                        if entity.fill_color is not None
                        else DEFAULT_FILL_COLOR
                    )
                    render_data.append(
                        FillRenderData(
                            geometry=text_geo,
                            style=FillStyle.SOLID,
                            color=color,
                        )
                    )

        return render_data


def find_array_for_entity(arrays: list, entity_id: int):
    for arr in arrays:
        if arr.guide_entity_id == entity_id:
            return arr
    return None
