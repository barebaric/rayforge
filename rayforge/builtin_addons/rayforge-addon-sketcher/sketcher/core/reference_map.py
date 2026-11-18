"""Typed rewrites of the point/entity id references on model objects."""

from __future__ import annotations

from collections.abc import Iterable

from raygeo.sketch import constraints as c
from raygeo.sketch import entities as e
from raygeo.sketch.types import EntityID

IdMap = dict[EntityID, EntityID]


def _map(value: EntityID, id_map: IdMap) -> EntityID:
    return id_map.get(value, value)


def _map_opt(value: EntityID | None, id_map: IdMap) -> EntityID | None:
    if value is None:
        return None
    return id_map.get(value, value)


def _map_all(values: Iterable[EntityID], id_map: IdMap) -> list[EntityID]:
    return [id_map.get(value, value) for value in values]


def remap_references(obj: object, id_map: IdMap) -> None:
    """Rewrites every id reference carried by *obj* through *id_map*."""
    if isinstance(obj, e.Line):
        obj.p1_idx = _map(obj.p1_idx, id_map)
        obj.p2_idx = _map(obj.p2_idx, id_map)
    elif isinstance(obj, e.Arc):
        obj.start_idx = _map(obj.start_idx, id_map)
        obj.end_idx = _map(obj.end_idx, id_map)
        obj.center_idx = _map(obj.center_idx, id_map)
    elif isinstance(obj, e.Circle):
        obj.center_idx = _map(obj.center_idx, id_map)
        obj.radius_pt_idx = _map(obj.radius_pt_idx, id_map)
    elif isinstance(obj, e.Ellipse):
        obj.center_idx = _map(obj.center_idx, id_map)
        obj.radius_x_pt_idx = _map(obj.radius_x_pt_idx, id_map)
        obj.radius_y_pt_idx = _map(obj.radius_y_pt_idx, id_map)
        obj.helper_line_ids = _map_all(obj.helper_line_ids, id_map)
    elif isinstance(obj, e.Bezier):
        obj.start_idx = _map(obj.start_idx, id_map)
        obj.end_idx = _map(obj.end_idx, id_map)
    elif isinstance(obj, e.Polygon):
        obj.center_idx = _map(obj.center_idx, id_map)
        obj.handle_idx = _map(obj.handle_idx, id_map)
    elif isinstance(obj, e.TextBox):
        obj.origin_id = _map(obj.origin_id, id_map)
        obj.width_id = _map(obj.width_id, id_map)
        obj.height_id = _map(obj.height_id, id_map)
        obj.construction_line_ids = _map_all(obj.construction_line_ids, id_map)
    elif isinstance(obj, c.AngleConstraint):
        obj.e1_id = _map(obj.e1_id, id_map)
        obj.e2_id = _map(obj.e2_id, id_map)
        obj.e1_far_idx = _map_opt(obj.e1_far_idx, id_map)
        obj.e2_far_idx = _map_opt(obj.e2_far_idx, id_map)
    elif isinstance(obj, c.AspectRatioConstraint):
        obj.p1 = _map(obj.p1, id_map)
        obj.p2 = _map(obj.p2, id_map)
        obj.p3 = _map(obj.p3, id_map)
        obj.p4 = _map(obj.p4, id_map)
    elif isinstance(obj, c.CoincidentConstraint):
        obj.p1 = _map(obj.p1, id_map)
        obj.p2 = _map(obj.p2, id_map)
    elif isinstance(obj, c.CollinearConstraint):
        obj.p1 = _map(obj.p1, id_map)
        obj.p2 = _map(obj.p2, id_map)
        obj.p3 = _map(obj.p3, id_map)
    elif isinstance(obj, c.DiameterConstraint):
        obj.circle_id = _map(obj.circle_id, id_map)
    elif isinstance(obj, c.DistanceConstraint):
        obj.p1 = _map(obj.p1, id_map)
        obj.p2 = _map(obj.p2, id_map)
    elif isinstance(obj, c.DragConstraint):
        obj.point_id = _map(obj.point_id, id_map)
    elif isinstance(obj, c.EqualDistanceConstraint):
        obj.p1 = _map(obj.p1, id_map)
        obj.p2 = _map(obj.p2, id_map)
        obj.p3 = _map(obj.p3, id_map)
        obj.p4 = _map(obj.p4, id_map)
    elif isinstance(obj, c.EqualLengthConstraint):
        obj.entity_ids = _map_all(obj.entity_ids, id_map)
    elif isinstance(obj, c.HorizontalConstraint):
        obj.p1 = _map(obj.p1, id_map)
        obj.p2 = _map(obj.p2, id_map)
    elif isinstance(obj, c.ParallelogramConstraint):
        obj.p_origin = _map(obj.p_origin, id_map)
        obj.p_width = _map(obj.p_width, id_map)
        obj.p_height = _map(obj.p_height, id_map)
        obj.p4 = _map(obj.p4, id_map)
    elif isinstance(obj, c.PerpendicularConstraint):
        obj.e1_id = _map(obj.e1_id, id_map)
        obj.e2_id = _map(obj.e2_id, id_map)
    elif isinstance(obj, (c.PointOnCurveConstraint, c.PointOnLineConstraint)):
        obj.point_id = _map(obj.point_id, id_map)
        obj.shape_id = _map(obj.shape_id, id_map)
    elif isinstance(obj, c.RadiusConstraint):
        obj.entity_id = _map(obj.entity_id, id_map)
    elif isinstance(obj, c.RotationalConstraint):
        obj.center = _map(obj.center, id_map)
        obj.p1 = _map(obj.p1, id_map)
        obj.p2 = _map(obj.p2, id_map)
    elif isinstance(obj, c.SymmetryConstraint):
        obj.p1 = _map(obj.p1, id_map)
        obj.p2 = _map(obj.p2, id_map)
        obj.center = _map_opt(obj.center, id_map)
    elif isinstance(obj, c.TangentConstraint):
        obj.line_id = _map(obj.line_id, id_map)
        obj.shape_id = _map(obj.shape_id, id_map)
    elif isinstance(obj, c.VerticalConstraint):
        obj.p1 = _map(obj.p1, id_map)
        obj.p2 = _map(obj.p2, id_map)
