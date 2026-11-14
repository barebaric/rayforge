"""Construction and state transforms for raygeo sketch entities.

The raygeo entity objects are immutable snapshots: geometry is
mutated through the registry (``set_state``, ``set_point_position``)
and new entities are built with plain constructors and inserted with
``add_entity``. This module concentrates the constructor knowledge
per entity kind so commands, arrays and groups can clone, place and
re-derive entities without reaching into serialization dicts.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Protocol

from raygeo.geo.shape.polygon import get_polygon_signed_area
from raygeo.geo.shape.text import FontConfig
from raygeo.sketch.entities.arc import Arc
from raygeo.sketch.entities.bezier import Bezier
from raygeo.sketch.entities.circle import Circle
from raygeo.sketch.entities.ellipse import Ellipse
from raygeo.sketch.entities.line import Line
from raygeo.sketch.entities.polygon import Polygon
from raygeo.sketch.entities.text_box import TextBox

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry


class PlacementTransform(Protocol):
    """Anything that can map absolute points and relative offsets
    (e.g. an array placement or a mirror axis) into new coordinates."""

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        """Transforms an absolute point."""
        ...

    def transform_offset(self, dx: float, dy: float) -> tuple[float, float]:
        """Transforms a relative offset (e.g. a bezier control point)."""
        ...


def flips_chirality(placement: PlacementTransform) -> bool:
    """Returns True when the placement mirrors the plane."""
    ex = placement.transform_offset(1.0, 0.0)
    ey = placement.transform_offset(0.0, 1.0)
    return ex[0] * ey[1] - ex[1] * ey[0] < 0


def clone_entity(
    registry: EntityRegistry,
    entity: Any,
    pid_map: dict[int, int],
    placement: PlacementTransform | None = None,
) -> Any:
    """Builds an unattached copy of the entity with its point ids
    remapped through ``pid_map``. Helper references (ellipse axis
    lines, text box construction lines) are not copied: a copy must
    not retain references to another member's helper geometry. A
    placement transforms the entity-internal relative state (bezier
    control offsets, polygon chirality)."""
    construction = entity.construction
    kind = entity.type_name()
    if kind == "line":
        return Line(
            pid_map[entity.p1_idx],
            pid_map[entity.p2_idx],
            construction,
        )
    if kind == "arc":
        return Arc(
            pid_map[entity.start_idx],
            pid_map[entity.end_idx],
            pid_map[entity.center_idx],
            entity.clockwise,
            construction,
        )
    if kind == "circle":
        return Circle(
            pid_map[entity.center_idx],
            pid_map[entity.radius_pt_idx],
            construction,
        )
    if kind == "ellipse":
        return Ellipse(
            pid_map[entity.center_idx],
            pid_map[entity.radius_x_point_idx],
            pid_map[entity.radius_y_point_idx],
            construction,
        )
    if kind == "bezier":
        cp1 = _transformed_offset(entity.cp1_offset, placement)
        cp2 = _transformed_offset(entity.cp2_offset, placement)
        return Bezier(
            pid_map[entity.start_idx],
            pid_map[entity.end_idx],
            construction,
            cp1=cp1,
            cp2=cp2,
        )
    if kind == "polygon":
        vertices = [tuple(vertex) for vertex in entity.vertices]
        rings = [list(ring) for ring in entity.rings]
        if placement is not None and flips_chirality(placement):
            rings = [[(u, -v) for u, v in ring] for ring in rings]
        return Polygon(
            pid_map[entity.center_idx],
            pid_map[entity.handle_idx],
            rings[0],
            entity.closed,
            construction,
            rings=rings if len(rings) > 1 else None,
        )
    if kind == "text_box":
        return TextBox(
            pid_map[entity.origin_id],
            pid_map[entity.width_id],
            pid_map[entity.height_id],
            entity.content,
            entity.font_family,
            entity.font_size,
            entity.bold,
            entity.italic,
        )
    raise ValueError(f"Unsupported entity type: {kind}")


def entity_from_dict(entity_dict: dict) -> Any:
    """Builds an unattached entity from its reference dictionary,
    preserving its id and point ids. The registry assigns a fresh id
    on insertion."""
    kind = entity_dict.get("type")
    construction = entity_dict.get("construction", False)
    if kind == "line":
        entity = Line(
            entity_dict["p1_idx"], entity_dict["p2_idx"], construction
        )
    elif kind == "arc":
        entity = Arc(
            entity_dict["start_idx"],
            entity_dict["end_idx"],
            entity_dict["center_idx"],
            entity_dict.get("clockwise", False),
            construction,
        )
    elif kind == "circle":
        entity = Circle(
            entity_dict["center_idx"],
            entity_dict["radius_pt_idx"],
            construction,
        )
    elif kind == "ellipse":
        entity = Ellipse(
            entity_dict["center_idx"],
            entity_dict["radius_x_pt_idx"],
            entity_dict["radius_y_pt_idx"],
            construction,
        )
    elif kind == "bezier":
        cp1 = None
        if "cp1_dx" in entity_dict and "cp1_dy" in entity_dict:
            cp1 = (entity_dict["cp1_dx"], entity_dict["cp1_dy"])
        cp2 = None
        if "cp2_dx" in entity_dict and "cp2_dy" in entity_dict:
            cp2 = (entity_dict["cp2_dx"], entity_dict["cp2_dy"])
        entity = Bezier(
            entity_dict["start_idx"],
            entity_dict["end_idx"],
            construction,
            cp1=cp1,
            cp2=cp2,
        )
    elif kind == "polygon":
        rings = entity_dict.get("rings")
        entity = Polygon(
            entity_dict["center_idx"],
            entity_dict["handle_idx"],
            [tuple(vertex) for vertex in entity_dict["vertices"]],
            entity_dict.get("closed", False),
            construction,
            rings=rings,
        )
    elif kind == "text_box":
        font = entity_dict.get("font_config", {})
        entity = TextBox(
            entity_dict["origin_id"],
            entity_dict["width_id"],
            entity_dict["height_id"],
            entity_dict.get("content", ""),
            font.get("font_family", "sans-serif"),
            font.get("font_size", 10.0),
            font.get("bold", False),
            font.get("italic", False),
        )
    else:
        raise ValueError(f"Unsupported entity type: {kind}")
    if "id" in entity_dict:
        entity.id = entity_dict["id"]
    return entity


def _transformed_offset(
    offset: tuple[float, float] | None,
    placement: PlacementTransform | None,
) -> tuple[float, float] | None:
    if offset is None:
        return None
    if placement is None:
        return (offset[0], offset[1])
    return placement.transform_offset(offset[0], offset[1])


def apply_placement_to_state(
    registry: EntityRegistry, eid: int, placement: PlacementTransform
) -> None:
    """Transforms the entity-internal relative state (bezier control
    offsets, polygon chirality) by the placement. Defining points are
    moved by the caller (see ``EntityGroup.apply_placement``)."""
    entity = registry.get_entity_object(eid)
    if entity is None:
        return
    kind = entity.type_name()
    if kind == "bezier":
        state: dict[str, Any] = {}
        cp1 = _transformed_offset(entity.cp1_offset, placement)
        cp2 = _transformed_offset(entity.cp2_offset, placement)
        if entity.cp1_offset is not None and cp1 is not None:
            state["cp1"] = [cp1[0], cp1[1]]
        if entity.cp2_offset is not None and cp2 is not None:
            state["cp2"] = [cp2[0], cp2[1]]
        if state:
            registry.set_state(eid, state)
    elif kind == "polygon":
        if flips_chirality(placement):
            _flip_polygon_rings(registry, eid, entity)


def rewrite_state_from(
    registry: EntityRegistry,
    copy_eid: int,
    template_eid: int,
    placement: PlacementTransform,
) -> None:
    """Re-derives the copy entity's internal state from a template
    entity of the same kind: the template's state transformed by the
    placement. The template is the source of truth; the copy's own
    state is never read."""
    template = registry.get_entity_object(template_eid)
    copy = registry.get_entity_object(copy_eid)
    if template is None or copy is None:
        return
    kind = template.type_name()
    if kind == "bezier":
        cp1 = _transformed_offset(template.cp1_offset, placement)
        cp2 = _transformed_offset(template.cp2_offset, placement)
        state: dict[str, Any] = {}
        if cp1 is not None:
            state["cp1"] = [cp1[0], cp1[1]]
        if cp2 is not None:
            state["cp2"] = [cp2[0], cp2[1]]
        registry.set_state(copy_eid, state)
    elif kind == "polygon":
        rings = [list(ring) for ring in template.rings]
        if flips_chirality(placement):
            rings = [[(u, -v) for u, v in ring] for ring in rings]
        state = {
            "vertices": [list(vertex) for vertex in rings[0]],
            "closed": template.closed,
        }
        if len(rings) > 1:
            state["rings"] = [list(ring) for ring in rings[1:]]
        registry.set_state(copy_eid, state)
    elif kind == "text_box":
        registry.set_state(
            copy_eid,
            {
                "content": template.content,
                "font_config": template.font_config.to_dict(),
            },
        )


def _flip_polygon_rings(
    registry: EntityRegistry, eid: int, entity: Polygon
) -> None:
    """Negates the local v component of every ring (mirroring the
    outline with the frame) and normalizes ring winding."""
    rings = [[(u, -v) for u, v in ring] for ring in entity.rings]
    state: dict[str, Any] = {
        "vertices": [list(vertex) for vertex in rings[0]],
        "closed": entity.closed,
    }
    if len(rings) > 1:
        state["rings"] = [list(ring) for ring in rings[1:]]
    registry.set_state(eid, state)


def mirrored_rings(rings: list[list[tuple[float, float]]]) -> list:
    """Negates the local v component of every ring, for a mirror
    transform of the frame."""
    return [[(u, -v) for u, v in ring] for ring in rings]


def normalize_ring_winding(
    closed: bool, rings: list[list[tuple[float, float]]]
) -> list[list[tuple[float, float]]]:
    """Enforces the winding convention on closed shapes: ring 0 CCW,
    hole rings CW."""
    if closed:
        rings = [list(ring) for ring in rings]
        if rings[0] and get_polygon_signed_area(rings[0]) < 0:
            rings[0].reverse()
        for ring in rings[1:]:
            if ring and get_polygon_signed_area(ring) > 0:
                ring.reverse()
    return rings


def text_box_fourth_corner(
    registry: EntityRegistry, entity: TextBox
) -> int | None:
    """The fourth frame corner point id of a text box, found through
    its construction lines."""
    for eid in registry.helper_ids(entity.id):
        if registry.entity_type(eid) != "line":
            continue
        endpoints = registry.entity_endpoint_ids(eid)
        if not endpoints or len(endpoints) != 2:
            continue
        p1, p2 = endpoints
        if p1 == entity.width_id and p2 not in (
            entity.origin_id,
            entity.height_id,
        ):
            return p2
        if p2 == entity.width_id and p1 not in (
            entity.origin_id,
            entity.height_id,
        ):
            return p1
    return None


def text_box_natural_size(
    content: str,
    font_family: str,
    font_size: float,
    bold: bool,
    italic: bool,
) -> tuple[float, float]:
    """The natural (width, height) of the text content, derived from
    the actual geometry bounding box so the frame matches the ink
    extent exactly. raygeo returns Y-UP geometry."""
    from raygeo.geo.shape.text import text_to_geometry

    if not content:
        return 10.0, font_size

    font = FontConfig(
        family=font_family, size=font_size, bold=bold, italic=italic
    )
    geo = text_to_geometry(content, font_config=font)
    _, geo_min_y, _, geo_max_y = geo.rect()
    geo_height = geo_max_y - geo_min_y
    _, _, geo_max_x, _ = geo.rect()
    return (geo_max_x, geo_height)


def polygon_world_vertices(
    registry: EntityRegistry, entity: Polygon
) -> list[tuple[float, float]]:
    """The outer ring in world coordinates."""
    return polygon_world_rings(registry, entity)[0]


def polygon_world_rings(
    registry: EntityRegistry, entity: Polygon
) -> list[list[tuple[float, float]]]:
    """Every ring in world coordinates (outer first)."""
    return [list(ring) for ring in entity.get_world_rings(registry)]
