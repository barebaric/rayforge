"""Round-trip tests for the legacy sketch dictionary migration.

Old .rfs files carry the legacy registry/constraint dictionary
shape. ``Sketch.from_dict`` must keep loading them, and a migrated
sketch must serialize back to dictionaries the legacy reader shape
can express (same type keys, same fields).
"""

import pytest
from raygeo.sketch.constraints.horizontal import HorizontalConstraint
from raygeo.sketch.constraints.radius import RadiusConstraint
from sketcher.core.legacy import migrate_sketch_data
from sketcher.core.sketch import Sketch

LEGACY_REGISTRY = {
    "points": [
        {"id": 0, "x": 0.0, "y": 0.0, "fixed": True},
        {"id": 1, "x": 10.0, "y": 0.0, "fixed": False},
        {"id": 2, "x": 10.0, "y": 5.0, "waypoint_type": "smooth"},
    ],
    "entities": [
        {
            "id": 3,
            "type": "line",
            "p1_idx": 0,
            "p2_idx": 1,
            "construction": False,
        },
        {
            "id": 4,
            "type": "circle",
            "center_idx": 0,
            "radius_pt_idx": 1,
            "construction": True,
        },
        {
            "id": 5,
            "type": "text_box",
            "origin_id": 0,
            "width_id": 1,
            "height_id": 2,
            "content": "",
            "font_config": {
                "font_family": "serif",
                "font_size": 12.0,
                "bold": True,
                "italic": False,
            },
            "construction_line_ids": [],
        },
    ],
    "id_counter": 6,
}

LEGACY_CONSTRAINTS = [
    {
        "type": "HorizontalConstraint",
        "p1": 0,
        "p2": 1,
        "user_visible": True,
    },
    {
        "type": "RadiusConstraint",
        "entity_id": 4,
        "value": 10.0,
        "user_visible": True,
    },
    {
        "type": "DistanceConstraint",
        "p1": 0,
        "p2": 1,
        "value": 10.0,
        "expression": "width",
        "user_visible": True,
    },
    {
        "type": "SymmetryConstraint",
        "p1": 1,
        "p2": 2,
        "center": None,
        "axis": 3,
        "user_visible": True,
    },
    {},
]

LEGACY_SKETCH = {
    "version": 1,
    "uid": "legacy-uid",
    "type": "sketch",
    "name": "Legacy",
    "params": {"width": {"expression": "10.0"}},
    "registry": LEGACY_REGISTRY,
    "constraints": LEGACY_CONSTRAINTS,
    "origin_id": 0,
    "hidden": False,
}


def test_migrate_merges_constraints_into_registry():
    migrated = migrate_sketch_data(LEGACY_SKETCH)
    constraints = migrated["registry"]["constraints"]
    types = [c.get("type") for c in constraints]
    assert types == [
        "HorizontalConstraint",
        "RadiusConstraint",
        "DistanceConstraint",
        "SymmetryConstraint",
    ]
    # The symmetry constraint's explicit None center is preserved
    # verbatim (raygeo resolves the axis and ignores a null center).
    symmetry = constraints[3]
    assert symmetry["center"] is None
    assert symmetry["axis"] == 3


def test_migrate_does_not_mutate_input():
    import copy

    data = copy.deepcopy(LEGACY_SKETCH)
    migrate_sketch_data(data)
    assert data["registry"] == LEGACY_REGISTRY
    assert data["constraints"] == LEGACY_CONSTRAINTS


def test_load_legacy_sketch():
    sketch = Sketch.from_dict(dict(LEGACY_SKETCH))
    assert sketch.registry.entity_count() == 3
    assert len(sketch.constraints) == 4
    # The waypoint type survives into the app-level overlay.
    from sketcher.core.entity_flags import WaypointType

    assert sketch.flags.waypoint_type(2) is WaypointType.SMOOTH
    assert sketch.uid == "legacy-uid"


def test_loaded_constraints_are_functional():
    sketch = Sketch.from_dict(dict(LEGACY_SKETCH))
    horizontal = sketch.constraints[0]
    assert isinstance(horizontal, HorizontalConstraint)
    assert (horizontal.p1, horizontal.p2) == (0, 1)
    radius = sketch.constraints[1]
    assert isinstance(radius, RadiusConstraint)
    assert radius.value == 10.0
    assert radius.constrains_radius(sketch.registry, 4)
    distance = sketch.constraints[2]
    assert distance.expression == "width"


def test_roundtrip_preserves_geometry_and_constraints():
    sketch = Sketch.from_dict(dict(LEGACY_SKETCH))
    data = sketch.to_dict()
    reloaded = Sketch.from_dict(data)
    assert reloaded.registry.entity_count() == 3
    assert len(reloaded.constraints) == 4
    for original, loaded in zip(sketch.constraints, reloaded.constraints):
        assert type(original) is type(loaded)
    assert reloaded.flags.waypoint_type(2) is not None


def test_saved_constraints_keep_expression_and_target():
    sketch = Sketch.from_dict(dict(LEGACY_SKETCH))
    constraint_dicts = sketch.to_dict()["constraints"]
    by_type = {c["type"]: c for c in constraint_dicts}
    assert by_type["DistanceConstraint"]["expression"] == "width"
    radius = by_type["RadiusConstraint"]
    # Radius keeps its resolved points so the same loader accepts it.
    assert "value" in radius
    assert radius["value"] == 10.0


def test_skips_unknown_constraint_types():
    data = dict(LEGACY_SKETCH)
    data["constraints"] = [
        {"type": "HorizontalConstraint", "p1": 0, "p2": 1},
        {"type": "MysteryConstraint", "p1": 0},
        {},
    ]
    sketch = Sketch.from_dict(data)
    assert len(sketch.constraints) == 1


def test_legacy_solve_still_constrains():
    sketch = Sketch.from_dict(dict(LEGACY_SKETCH))
    sketch.params.set("width", "10.0")
    assert sketch.solve()
    horizontal = sketch.constraints[0]
    _ = horizontal
    p1 = sketch.registry.get_point(1)
    assert p1 is not None
    assert p1[1] == pytest.approx(0.0, abs=1e-5)
