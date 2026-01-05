import math
import uuid
import pytest
from pathlib import Path
from rayforge.core.geo import Geometry
from rayforge.core.geo.constants import (
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Z,
    COL_I,
    COL_J,
    COL_CW,
)
from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.sketch import Fill
from rayforge.core.sketcher.constraints import (
    EqualDistanceConstraint,
    PointOnLineConstraint,
    PerpendicularConstraint,
    EqualLengthConstraint,
    SymmetryConstraint,
)
from rayforge.core.varset import FloatVar


def test_sketch_workflow():
    s = Sketch()

    # 1. Define params
    s.set_param("side", 10.0)

    # 2. Add geometry (approximate square)
    p1 = s.add_point(0, 0, fixed=True)
    p2 = s.add_point(5, 0)
    p3 = s.add_point(5, 5)
    p4 = s.add_point(0, 5)

    s.add_line(p1, p2)
    s.add_line(p2, p3)
    s.add_line(p3, p4)
    s.add_line(p4, p1)

    # 3. Constrain
    s.constrain_horizontal(p1, p2)
    s.constrain_vertical(p2, p3)
    s.constrain_horizontal(p4, p3)
    s.constrain_vertical(p1, p4)

    s.constrain_distance(p1, p2, "side")
    s.constrain_distance(p2, p3, "side")

    # 4. Solve
    assert s.solve() is True

    # 5. Check geometry export
    geo = s.to_geometry()
    assert isinstance(geo, Geometry)
    assert len(geo) == 8  # 4 moves + 4 lines (simple export)

    # Check bounding box is approx 10x10
    min_x, min_y, max_x, max_y = geo.rect()
    assert min_x == pytest.approx(0.0)
    assert min_y == pytest.approx(0.0)
    assert max_x == pytest.approx(10.0, abs=1e-4)
    assert max_y == pytest.approx(10.0, abs=1e-4)


def test_sketch_is_empty():
    """
    Verifies the is_empty property logic:
    - True for a new sketch (only origin point).
    - True if only points are added (points are not drawable entities).
    - False once a drawable entity (Line/Arc/Circle) is added.
    """
    s = Sketch()

    # 1. Initially empty (contains origin point, but no entities)
    assert s.is_empty is True
    # Verify internal state: Origin exists, so points list is not empty
    assert len(s.registry.points) == 1
    assert len(s.registry.entities) == 0

    # 2. Adding standalone points doesn't change emptiness regarding drawable
    # geometry
    p1 = s.add_point(10, 0)
    assert s.is_empty is True

    # 3. Adding an entity (Line) makes it not empty
    s.add_line(s.origin_id, p1)
    assert s.is_empty is False


def test_sketch_arc_export():
    s = Sketch()
    # Simple quarter circle arc
    p1 = s.add_point(10, 0)  # Start
    p2 = s.add_point(0, 10)  # End
    c = s.add_point(0, 0)  # Center

    s.add_arc(p1, p2, c, clockwise=False)

    geo = s.to_geometry()
    data = geo.data
    assert data is not None

    # Should contain a MoveTo(10,0) and ArcTo(0,10)
    arc_rows = data[data[:, COL_TYPE] == CMD_TYPE_ARC]
    assert len(arc_rows) == 1

    arc_row = arc_rows[0]
    assert (arc_row[COL_X : COL_Z + 1] == (0.0, 10.0, 0.0)).all()
    # Check offsets. Center (0,0) relative to Start (10,0) is (-10, 0)
    assert (arc_row[COL_I : COL_J + 1] == (-10.0, 0.0)).all()
    assert arc_row[COL_CW] == 0.0


def test_sketch_circle_workflow():
    s = Sketch()
    s.set_param("diam", 20.0)

    center = s.add_point(10, 10, fixed=True)
    radius_pt = s.add_point(15, 10)  # Initial radius is 5
    circ_id = s.add_circle(center, radius_pt)

    s.constrain_diameter(circ_id, "diam")

    assert s.solve() is True

    # After solve, radius should be 10, diameter 20.
    # The radius point should be 10 units away from center.
    p = s.registry.get_point(radius_pt)
    c = s.registry.get_point(center)
    dist = ((p.x - c.x) ** 2 + (p.y - c.y) ** 2) ** 0.5
    assert dist == pytest.approx(10.0)

    geo = s.to_geometry()
    # Should export as two semi-circles -> 2 ArcToCommands
    assert isinstance(geo, Geometry)
    data = geo.data
    assert data is not None
    arcs = data[data[:, COL_TYPE] == CMD_TYPE_ARC]
    assert len(arcs) == 2


def test_sketch_construction_geometry_is_ignored_on_roundtrip():
    """
    Tests that geometry marked 'construction' is not exported, even after a
    full serialization and deserialization cycle.
    """
    s = Sketch()
    assert s.uid is not None

    # Add a regular line that should be exported
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    s.add_line(p1, p2, construction=False)

    # Add various construction entities that should be ignored
    p3 = s.add_point(0, 10)
    p4 = s.add_point(10, 10)
    s.add_line(p3, p4, construction=True)

    p5_s = s.add_point(20, 0)
    p5_e = s.add_point(30, 10)
    p5_c = s.add_point(20, 10)
    s.add_arc(p5_s, p5_e, p5_c, construction=True)

    p6_c = s.add_point(40, 0)
    p6_r = s.add_point(45, 0)
    s.add_circle(p6_c, p6_r, construction=True)

    # 1. Serialize the sketch to a dictionary
    sketch_data = s.to_dict()
    assert "uid" in sketch_data

    # Verify that the data contains the correct flags
    construction_entities = [
        e for e in sketch_data["registry"]["entities"] if e["construction"]
    ]
    assert len(construction_entities) == 3

    # 2. Create a new sketch from the serialized data
    s2 = Sketch.from_dict(sketch_data)
    assert s2.uid == s.uid

    # 3. Generate the geometry from the new sketch
    geo = s2.to_geometry()

    # The simple export creates a MoveTo and a LineTo for each line.
    # We only have one non-construction line.
    # So we expect exactly 2 commands total after the round trip.
    assert len(geo) == 2


def test_sketch_equal_length_workflow():
    """Test a full workflow using an equal length constraint."""
    s = Sketch()

    # Line 1 will be fixed at length 10
    p1 = s.add_point(0, 0, fixed=True)
    p2 = s.add_point(10, 0)
    l1 = s.add_line(p1, p2)
    s.constrain_horizontal(p1, p2)
    s.constrain_distance(p1, p2, 10.0)

    # Line 2 will start at length 5 and should be solved to 10
    p3 = s.add_point(20, 0, fixed=True)
    p4 = s.add_point(25, 0)
    l2 = s.add_line(p3, p4)
    s.constrain_horizontal(p3, p4)

    # Apply the Equal Length constraint
    s.constrain_equal_length([l1, l2])

    assert s.solve() is True

    # Check that p4 has moved to make line 2 have length 10
    pt4 = s.registry.get_point(p4)
    pt3 = s.registry.get_point(p3)
    dist = ((pt4.x - pt3.x) ** 2 + (pt4.y - pt3.y) ** 2) ** 0.5
    assert dist == pytest.approx(10.0)
    assert pt4.x == pytest.approx(30.0)
    assert pt4.y == pytest.approx(0.0)


def test_sketch_symmetry_workflow():
    """Test full workflow for adding symmetry constraints."""
    s = Sketch()

    # --- Test Point Symmetry ---
    # Center at (0,0), P1 at (-10, 0), P2 at (10, 5) [wrong y]
    c = s.add_point(0, 0, fixed=True)
    p1 = s.add_point(-10, 0)
    p2 = s.add_point(10, 5)

    # Constrain P1, P2 symmetric to C
    s.constrain_symmetry([c, p1, p2], [])

    s.solve()

    pt1 = s.registry.get_point(p1)
    pt2 = s.registry.get_point(p2)

    # Assert X symmetry (sum of x relative to center should be 0)
    # They started at -10 and 10, so they should stay there or move
    # symmetrically.
    assert (pt1.x + pt2.x) == pytest.approx(0.0, abs=1e-4)

    # Assert Y symmetry
    # Since center Y is 0, P1.y + P2.y should equal 0.
    # Initial: 0 and 5. Solver will move them to approx -2.5 and 2.5
    assert (pt1.y + pt2.y) == pytest.approx(0.0, abs=1e-4)
    # Check they are actually separated and symmetric, not just both at 0
    assert pt2.y == pytest.approx(2.5, abs=1.0)
    assert pt1.y == pytest.approx(-2.5, abs=1.0)

    # --- Test Line Symmetry ---
    s2 = Sketch()
    # Axis on Y-axis
    l1 = s2.add_point(0, 10, fixed=True)
    l2 = s2.add_point(0, 20, fixed=True)
    axis = s2.add_line(l1, l2)

    # P3 at (-5, 15), P4 at (5, 16) [wrong y]
    p3 = s2.add_point(-5, 15)
    p4 = s2.add_point(5, 16)

    s2.constrain_symmetry([p3, p4], [axis])
    s2.solve()

    pt3 = s2.registry.get_point(p3)
    pt4 = s2.registry.get_point(p4)
    assert pt4.y == pytest.approx(pt3.y, abs=1e-4)


def test_sketch_parameter_updates():
    """Test that changing a parameter and re-solving updates geometry."""
    s = Sketch()
    s.set_param("len", 10.0)

    p1 = s.add_point(0, 0, fixed=True)
    p2 = s.add_point(5, 0)
    s.constrain_distance(p1, p2, "len")

    assert s.solve() is True
    assert s.registry.get_point(p2).x == pytest.approx(10.0)

    # Change param
    s.set_param("len", 20.0)
    s.solve()
    assert s.registry.get_point(p2).x == pytest.approx(20.0)


def test_solve_with_variable_overrides():
    """
    Tests that variable overrides in solve() work correctly and are temporary.
    """
    # 1. Setup a sketch with a parameter
    s = Sketch()
    s.set_param("width", 10.0)
    p1 = s.add_point(0, 0, fixed=True)
    p2 = s.add_point(1, 0)  # Initial position doesn't matter much
    s.constrain_distance(p1, p2, "width")

    # 2. Solve with the default parameter value
    assert s.solve() is True
    pt2 = s.registry.get_point(p2)
    assert pt2.x == pytest.approx(10.0)
    # Check that the parameter context has the correct value
    assert s.params.get("width") == 10.0

    # 3. Solve again, this time with an override
    overrides = {"width": 25.0}
    assert s.solve(variable_overrides=overrides) is True
    pt2_override = s.registry.get_point(p2)
    # The geometry should reflect the overridden value
    assert pt2_override.x == pytest.approx(25.0)

    # 4. Check that the override was temporary
    assert s.params.get("width") == 10.0

    # 5. Solve again without overrides to confirm it uses the original value.
    pt2.x = 1.0
    pt2.y = 0.0
    assert s.solve() is True
    pt2_final = s.registry.get_point(p2)
    assert pt2_final.x == pytest.approx(10.0)


def test_solve_with_expression_override():
    """
    Tests that variable overrides can also accept string expressions.
    """
    s = Sketch()
    s.set_param("base_len", 10.0)
    s.set_param("width", "base_len")  # width = 10.0
    p1 = s.add_point(0, 0, fixed=True)
    p2 = s.add_point(1, 0)
    s.constrain_distance(p1, p2, "width")

    overrides = {"width": 30.0}
    assert s.solve(variable_overrides=overrides) is True
    assert s.registry.get_point(p2).x == pytest.approx(30.0)
    assert s.params.get("width") == 10.0  # Verify it was temporary


def test_sketch_constraint_shortcuts():
    """Verify all constraint shortcut methods properly register constraints."""
    s = Sketch()
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    p3 = s.add_point(0, 10)
    p4 = s.add_point(0, 14)
    c = s.add_point(5, 5)
    l1 = s.add_line(p1, p2)
    l2 = s.add_line(p3, p4)
    circ = s.add_circle(c, p1)

    # Call shortcuts not covered in main workflow test
    s.constrain_equal_distance(p1, p2, p3, p4)
    s.constrain_coincident(p1, p3)
    s.constrain_point_on_line(p3, l1)
    s.constrain_perpendicular(l1, l2)
    s.constrain_diameter(circ, 20.0)
    s.constrain_equal_length([l1, circ])
    s.constrain_symmetry([p1, p2, p3], [])  # Point symmetry
    s.constrain_symmetry([p3, p4], [l1])  # Line symmetry

    assert len(s.constraints) == 8
    assert isinstance(s.constraints[0], EqualDistanceConstraint)
    assert isinstance(s.constraints[2], PointOnLineConstraint)
    assert isinstance(s.constraints[3], PerpendicularConstraint)
    assert isinstance(s.constraints[5], EqualLengthConstraint)
    assert isinstance(s.constraints[6], SymmetryConstraint)
    assert isinstance(s.constraints[7], SymmetryConstraint)


def test_sketch_serialization_from_file():
    """
    Tests that a sketch can be loaded from a file, serialized back to a
    dictionary, and re-loaded from that dictionary without data loss.
    """
    # 1. Locate the file relative to this test file
    test_dir = Path(__file__).parent
    file_path = test_dir / "rect.rfs"

    if not file_path.exists():
        pytest.skip(f"Test data file not found: {file_path}")

    # 2. Load the sketch from the project file
    sketch1 = Sketch.from_file(file_path)
    assert sketch1.uid is not None
    assert isinstance(sketch1.uid, str)

    # 3. Serialize the loaded sketch back into a dictionary
    data_from_sketch1 = sketch1.to_dict()
    assert "uid" in data_from_sketch1
    assert data_from_sketch1["uid"] == sketch1.uid

    # 4. Create a second sketch instance from the serialized dictionary
    sketch2 = Sketch.from_dict(data_from_sketch1)
    assert sketch2.uid == sketch1.uid

    # 5. Serialize the second sketch
    data_from_sketch2 = sketch2.to_dict()

    # 6. The serialized data from both sketches must be identical.
    # This proves that the `from_dict` -> `to_dict` round trip is perfect.
    assert data_from_sketch1 == data_from_sketch2, (
        "Serialization round-trip failed"
    )

    # 7. As a final check, ensure both sketches are functionally equivalent
    # by solving and comparing a key result.
    assert sketch1.solve() is True
    assert sketch2.solve() is True
    p_final_s1 = sketch1.registry.get_point(8)  # A point from the sketch
    p_final_s2 = sketch2.registry.get_point(8)
    assert p_final_s1.pos() == pytest.approx(p_final_s2.pos())


@pytest.fixture
def setup_sketch_for_validation():
    """Provides a sketch with a variety of geometry for validation tests."""
    s = Sketch()
    # Points
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    p3 = s.add_point(0, 10)
    p_ext = s.add_point(5, 5)
    # Entities
    l1 = s.add_line(p1, p2)
    l2 = s.add_line(p1, p3)
    # Arc points are separate to avoid endpoint conflicts
    arc_s = s.add_point(20, 0)
    arc_e = s.add_point(0, 20)
    arc_c = s.add_point(0, 0)
    a1 = s.add_arc(arc_s, arc_e, arc_c)
    c1 = s.add_circle(p1, p2)

    return s, {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p_ext": p_ext,
        "l1": l1,
        "l2": l2,
        "a1": a1,
        "c1": c1,
    }


def test_sketch_supports_constraint(setup_sketch_for_validation):
    """Tests the logic of the `supports_constraint` method."""
    s, ids = setup_sketch_for_validation
    p1, p2, p3, p_ext = ids["p1"], ids["p2"], ids["p3"], ids["p_ext"]
    l1, l2, a1, c1 = ids["l1"], ids["l2"], ids["a1"], ids["c1"]

    # Test "dist"
    assert s.supports_constraint("dist", [p1, p2], []) is True
    assert s.supports_constraint("dist", [], [l1]) is True
    assert s.supports_constraint("dist", [p1], []) is False
    assert s.supports_constraint("dist", [p1, p2, p3], []) is False
    assert s.supports_constraint("dist", [p1], [l1]) is False
    assert s.supports_constraint("dist", [], [a1]) is False
    assert s.supports_constraint("dist", [], [l1, l2]) is False

    # Test "horiz" and "vert"
    for c_type in ("horiz", "vert"):
        # Valid cases
        assert s.supports_constraint(c_type, [p1, p2], []) is True
        assert s.supports_constraint(c_type, [], [l1]) is True
        assert s.supports_constraint(c_type, [], [l1, l2]) is True  # FIXED
        # Invalid cases
        assert s.supports_constraint(c_type, [p1], []) is False
        assert s.supports_constraint(c_type, [p1, p2, p3], []) is False
        assert s.supports_constraint(c_type, [p1], [l1]) is False
        assert s.supports_constraint(c_type, [], [a1]) is False

    # Test "radius"
    assert s.supports_constraint("radius", [], [a1]) is True
    assert s.supports_constraint("radius", [], [c1]) is True
    assert s.supports_constraint("radius", [], [l1]) is False
    assert s.supports_constraint("radius", [p1], [a1]) is False
    assert s.supports_constraint("radius", [], [a1, c1]) is False

    # Test "diameter"
    assert s.supports_constraint("diameter", [], [c1]) is True
    assert s.supports_constraint("diameter", [], [a1]) is False
    assert s.supports_constraint("diameter", [], [l1]) is False
    assert s.supports_constraint("diameter", [p1], [c1]) is False

    # Test "perp"
    assert s.supports_constraint("perp", [], [l1, l2]) is True
    assert s.supports_constraint("perp", [], [l1]) is False
    assert s.supports_constraint("perp", [], [l1, a1]) is True
    assert s.supports_constraint("perp", [], [a1, c1]) is True
    assert s.supports_constraint("perp", [], [l1, l2, l1]) is False

    # Test "tangent"
    assert s.supports_constraint("tangent", [], [l1, a1]) is True
    assert s.supports_constraint("tangent", [], [l1, c1]) is True
    assert s.supports_constraint("tangent", [], [l1, l2]) is False
    assert s.supports_constraint("tangent", [], [a1, c1]) is False
    assert s.supports_constraint("tangent", [], [l1]) is False
    assert s.supports_constraint("tangent", [], [a1]) is False

    # Test "equal"
    assert s.supports_constraint("equal", [], [l1, l2]) is True
    assert s.supports_constraint("equal", [], [l1, a1]) is True
    assert s.supports_constraint("equal", [], [l1, c1, a1]) is True
    assert s.supports_constraint("equal", [], [l1]) is False  # Needs >= 2
    assert s.supports_constraint("equal", [p1], [l1, l2]) is False
    assert s.supports_constraint("equal", [], []) is False

    # Test "align" (covers coincident and point-on-line)
    # Coincident (2 points)
    assert s.supports_constraint("align", [p1, p_ext], []) is True
    # Point-on-Shape (1 point, 1 shape)
    assert s.supports_constraint("align", [p_ext], [l1]) is True
    assert s.supports_constraint("align", [p_ext], [a1]) is True
    assert s.supports_constraint("align", [p_ext], [c1]) is True

    # Invalid: Endpoint on its own line
    assert s.supports_constraint("align", [p1], [l1]) is False
    # Invalid: Control point on its own shape
    arc_start_id = s.registry.get_entity(a1).start_idx
    assert s.supports_constraint("align", [arc_start_id], [a1]) is False
    circle_center_id = s.registry.get_entity(c1).center_idx
    assert s.supports_constraint("align", [circle_center_id], [c1]) is False

    # Invalid: Other combos
    assert s.supports_constraint("align", [p1, p2, p3], []) is False
    assert s.supports_constraint("align", [p_ext], [l1, l2]) is False

    # Test "coincident" (internal use)
    assert s.supports_constraint("coincident", [p1, p2], []) is True
    assert s.supports_constraint("coincident", [p1], []) is False
    assert s.supports_constraint("coincident", [p1], [l1]) is False

    # Test "point_on_line" (internal use, now means point-on-shape)
    assert s.supports_constraint("point_on_line", [p_ext], [l1]) is True
    assert s.supports_constraint("point_on_line", [p_ext], [a1]) is True
    assert s.supports_constraint("point_on_line", [p_ext], [c1]) is True
    # Invalid: Endpoint on its own line
    assert s.supports_constraint("point_on_line", [p1], [l1]) is False
    # Invalid: wrong number of items
    assert s.supports_constraint("point_on_line", [p1, p_ext], [l1]) is False
    assert s.supports_constraint("point_on_line", [p_ext], [l1, l2]) is False

    # Test "symmetry"
    # Case A: 3 Points
    assert s.supports_constraint("symmetry", [p1, p2, p3], []) is True
    # Case B: 2 Points + 1 Line
    assert s.supports_constraint("symmetry", [p1, p2], [l1]) is True
    # Invalid
    assert s.supports_constraint("symmetry", [p1, p2], []) is False
    assert s.supports_constraint("symmetry", [p1], [l1]) is False
    assert s.supports_constraint("symmetry", [p1, p2], [l1, l2]) is False


def test_sketch_initializes_with_empty_varset():
    """Test that a new Sketch has a valid, empty input_parameters VarSet."""
    sketch = Sketch()
    assert sketch.input_parameters is not None
    assert len(sketch.input_parameters) == 0


def test_bridge_from_varset_to_solver():
    """
    Verify that a value from input_parameters is correctly used by the solver.
    """
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)

    # 1. Define an input parameter
    width_var = FloatVar(key="width", label="Overall Width", default=123.45)
    sketch.input_parameters.add(width_var)

    # 2. Use it in a constraint
    sketch.constrain_distance(p1, p2, "width")

    # 3. Solve the sketch
    assert sketch.solve() is True

    # 4. Verify the result
    pt1 = sketch.registry.get_point(p1)
    pt2 = sketch.registry.get_point(p2)
    final_dist = math.hypot(pt2.x - pt1.x, pt2.y - pt1.y)

    assert final_dist == pytest.approx(123.45)


def test_input_parameter_overrides_internal_parameter():
    """
    Verify that if an input_parameter has the same key as an internal
    parameter, the input_parameter's value takes precedence.
    """
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)

    # 1. Set an internal parameter with a "wrong" value
    sketch.set_param("width", 999.9)

    # 2. Define an input parameter with the "correct" value
    width_var = FloatVar(key="width", label="Width", default=50.0)
    sketch.input_parameters.add(width_var)

    # 3. Use it in a constraint
    sketch.constrain_distance(p1, p2, "width")

    # 4. Solve the sketch
    assert sketch.solve() is True

    # 5. Verify the result uses the value from the input_parameter
    pt1 = sketch.registry.get_point(p1)
    pt2 = sketch.registry.get_point(p2)
    final_dist = math.hypot(pt2.x - pt1.x, pt2.y - pt1.y)

    assert final_dist == pytest.approx(50.0)


def test_solve_with_no_input_parameters():
    """
    Verify that a sketch with no input parameters still solves correctly.
    """
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 20)

    # Use a hard-coded value in the constraint
    sketch.constrain_distance(p1, p2, 25.0)
    assert sketch.solve() is True

    pt1 = sketch.registry.get_point(p1)
    pt2 = sketch.registry.get_point(p2)
    final_dist = math.hypot(pt2.x - pt1.x, pt2.y - pt1.y)

    assert final_dist == pytest.approx(25.0)


def test_sketch_serialization_roundtrip_with_input_parameters():
    """
    Verify that a sketch with input_parameters can be serialized and
    deserialized correctly, respecting the include_input_values flag.
    """
    # 1. Create a sketch and add an input parameter
    sketch = Sketch()
    assert sketch.uid is not None
    width_var = FloatVar(
        key="width", label="Overall Width", default=140.0, value=120.0
    )
    sketch.input_parameters.add(width_var)

    # Add some geometry that uses the parameter
    p1 = sketch.add_point(0, 0, fixed=True)
    p2 = sketch.add_point(100, 0)
    sketch.constrain_distance(p1, p2, "width")

    # 2. Serialize to dict WITH value (state)
    sketch_data_with_values = sketch.to_dict(include_input_values=True)

    # 3. Assert serialized state data is correct
    assert "uid" in sketch_data_with_values
    assert "input_parameters" in sketch_data_with_values
    var_data_with_value = sketch_data_with_values["input_parameters"]["vars"][
        0
    ]
    assert var_data_with_value["key"] == "width"
    assert var_data_with_value["default"] == 140.0
    assert "value" in var_data_with_value
    assert var_data_with_value["value"] == 120.0

    # 4. Serialize to dict WITHOUT value (definition)
    sketch_data_definition_only = sketch.to_dict(include_input_values=False)

    # 5. Assert serialized definition data is correct
    var_data_def_only = sketch_data_definition_only["input_parameters"][
        "vars"
    ][0]
    assert var_data_def_only["key"] == "width"
    assert var_data_def_only["default"] == 140.0
    assert "value" not in var_data_def_only

    # 6. Deserialize back into a new sketch from the state data
    new_sketch = Sketch.from_dict(sketch_data_with_values)
    assert new_sketch.uid == sketch.uid

    # 7. Assert the new sketch is reconstructed correctly
    reloaded_var = new_sketch.input_parameters.get("width")
    assert isinstance(reloaded_var, FloatVar)
    assert reloaded_var.default == 140.0
    assert reloaded_var.value == 120.0  # Value is restored

    # 8. Check: does it still solve correctly with restored value?
    assert new_sketch.solve() is True
    pt2_reloaded = new_sketch.registry.get_point(p2)
    final_dist = math.hypot(pt2_reloaded.x, pt2_reloaded.y)
    assert final_dist == pytest.approx(120.0)

    # 9. Test rehydration from definition-only data
    new_sketch_from_def = Sketch.from_dict(sketch_data_definition_only)
    reloaded_var_from_def = new_sketch_from_def.input_parameters.get("width")
    assert reloaded_var_from_def is not None
    # Value should be the default, not the original value
    assert reloaded_var_from_def.value == 140.0

    # 10. Check that it solves with the default value
    assert new_sketch_from_def.solve() is True
    pt2_reloaded_from_def = new_sketch_from_def.registry.get_point(p2)
    final_dist_from_def = math.hypot(
        pt2_reloaded_from_def.x, pt2_reloaded_from_def.y
    )
    assert final_dist_from_def == pytest.approx(140.0)


def test_sketch_deserialization_backward_compatibility():
    """
    Verify that a sketch dictionary without the 'input_parameters' and 'uid'
    keys can be loaded without crashing.
    """
    # 1. Create a dictionary representing an old file format
    old_sketch_data = {
        "params": {"expressions": {"width": "100"}},
        "registry": {
            "points": [
                {"id": 0, "x": 0.0, "y": 0.0, "fixed": True, "label": None}
            ],
            "entities": [],
        },
        "constraints": [],
        "origin_id": 0,
    }

    # 2. Try to load it
    try:
        sketch = Sketch.from_dict(old_sketch_data)
    except Exception as e:
        pytest.fail(f"Sketch.from_dict failed to load old format: {e}")

    # 3. Assert the result is a valid sketch
    assert sketch is not None
    assert isinstance(sketch, Sketch)
    # Assert a new UID was generated
    assert sketch.uid is not None
    assert isinstance(sketch.uid, str)
    # Assert an empty VarSet was created
    assert sketch.input_parameters is not None
    assert len(sketch.input_parameters) == 0
    # Assert fills list is initialized and empty for backward compatibility
    assert sketch.fills == []
    # Make sure other parts loaded correctly
    assert sketch.params.get("width") == 100.0
    assert sketch.name == ""


def test_sketch_serialization_roundtrip_with_fill():
    """
    Verify that a sketch with a Fill can be serialized and deserialized
    correctly.
    """
    # 1. Create a sketch with a simple closed shape
    s = Sketch()
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    p3 = s.add_point(10, 10)
    p4 = s.add_point(0, 10)
    s.add_line(p1, p2)
    s.add_line(p2, p3)
    s.add_line(p3, p4)
    s.add_line(p4, p1)

    # 2. Find the loop and create a Fill object for it
    loops = s._find_all_closed_loops()
    assert len(loops) == 1
    boundary = loops[0]
    fill_uid = str(uuid.uuid4())
    s.fills.append(Fill(uid=fill_uid, boundary=boundary))

    # 3. Serialize the sketch to a dictionary
    sketch_data = s.to_dict()
    assert "fills" in sketch_data
    assert len(sketch_data["fills"]) == 1
    fill_data = sketch_data["fills"][0]
    assert fill_data["uid"] == fill_uid
    # to_dict() should return lists for JSON compatibility
    expected_boundary = [list(item) for item in boundary]
    assert fill_data["boundary"] == expected_boundary

    # 4. Deserialize back into a new sketch
    new_sketch = Sketch.from_dict(sketch_data)
    assert new_sketch.uid == s.uid
    assert len(new_sketch.fills) == 1
    reloaded_fill = new_sketch.fills[0]
    assert isinstance(reloaded_fill, Fill)
    assert reloaded_fill.uid == fill_uid
    assert reloaded_fill.boundary == boundary


def test_fill_is_removed_when_boundary_is_broken():
    """
    Tests that a Fill is automatically removed if its boundary is no longer
    a valid closed loop after a geometry change.
    """
    # 1. Create a sketch with a filled square
    s = Sketch()
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    p3 = s.add_point(10, 10)
    p4 = s.add_point(0, 10)
    s.add_line(p1, p2)
    s.add_line(p2, p3)
    l3 = s.add_line(p3, p4)
    s.add_line(p4, p1)

    loops = s._find_all_closed_loops()
    assert len(loops) == 1
    s.fills.append(Fill(uid=str(uuid.uuid4()), boundary=loops[0]))
    assert len(s.fills) == 1

    # 2. Get the entity to remove and call the public removal method
    l3_entity = s.registry.get_entity(l3)
    assert l3_entity is not None
    s.remove_entities([l3_entity])

    # 3. Assert that the fill has been automatically removed by the method
    assert len(s.fills) == 0


def test_remove_point_if_unused():
    """
    Tests the remove_point_if_unused method.
    """
    s = Sketch()

    # 1. Test removing a point that is not used by any entity
    unused_pid = s.add_point(100, 100)
    assert s.remove_point_if_unused(unused_pid) is True
    assert unused_pid not in [p.id for p in s.registry.points]

    # 2. Test that None returns False and does nothing
    assert s.remove_point_if_unused(None) is False

    # 3. Test that a point used by an entity is not removed
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 0)
    s.add_line(p1, p2)
    assert s.remove_point_if_unused(p1) is False
    assert p1 in [p.id for p in s.registry.points]

    # 4. Test removing a point after its entity is removed
    p3 = s.add_point(20, 0)
    p4 = s.add_point(30, 0)
    l2 = s.add_line(p3, p4)
    entity = s.registry.get_entity(l2)
    if entity is not None:
        s.remove_entities([entity])
    assert s.remove_point_if_unused(p3) is True
    assert s.remove_point_if_unused(p4) is True
    assert p3 not in [p.id for p in s.registry.points]
    assert p4 not in [p.id for p in s.registry.points]

    # 5. Test that origin point is removed when not used by any entity
    # (in a new sketch with no entities, origin is considered unused)
    origin_pid = s.origin_id
    assert s.remove_point_if_unused(origin_pid) is True
    assert origin_pid not in [p.id for p in s.registry.points]

    # 6. Test that origin point is NOT removed when used by an entity
    s2 = Sketch()
    origin_pid2 = s2.origin_id
    p_other = s2.add_point(10, 0)
    s2.add_line(origin_pid2, p_other)
    assert s2.remove_point_if_unused(origin_pid2) is False
    assert origin_pid2 in [p.id for p in s2.registry.points]


class TestSketchLoopFindingHelpers:
    @pytest.fixture
    def cross_sketch(self):
        """A sketch with 4 lines meeting at the origin."""
        s = Sketch()
        p_center = s.origin_id
        p_right = s.add_point(10, 0)
        p_up = s.add_point(0, 10)
        p_left = s.add_point(-10, 0)
        p_down = s.add_point(0, -10)
        l_right = s.add_line(p_center, p_right)  # Angle 0
        l_up = s.add_line(p_center, p_up)  # Angle pi/2
        l_left = s.add_line(p_center, p_left)  # Angle pi
        l_down = s.add_line(p_center, p_down)  # Angle -pi/2
        return s, {
            "center": p_center,
            "l_right": l_right,
            "l_up": l_up,
            "l_left": l_left,
            "l_down": l_down,
        }

    def test_build_adjacency_list(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(10, 0)
        l1 = s.add_line(p1, p2)
        adj = s._build_adjacency_list()

        assert p1 in adj
        assert p2 in adj
        assert len(adj[p1]) == 1
        assert len(adj[p2]) == 1

        edge_from_p1 = adj[p1][0]
        assert edge_from_p1["id"] == l1
        assert edge_from_p1["to"] == p2
        assert edge_from_p1["fwd"] is True

        edge_from_p2 = adj[p2][0]
        assert edge_from_p2["id"] == l1
        assert edge_from_p2["to"] == p1
        assert edge_from_p2["fwd"] is False

    def test_sort_edges_by_angle(self, cross_sketch):
        s, ids = cross_sketch
        adj = s._build_adjacency_list()
        sorted_adj = s._sort_edges_by_angle(adj)

        center_edges = sorted_adj[ids["center"]]
        assert len(center_edges) == 4

        # Expected order: down (-pi/2), right (0), up (pi/2), left (pi)
        sorted_ids = [e["id"] for e in center_edges]
        assert sorted_ids == [
            ids["l_down"],
            ids["l_right"],
            ids["l_up"],
            ids["l_left"],
        ]

    def test_get_next_edge_ccw(self, cross_sketch):
        s, ids = cross_sketch
        adj = s._build_adjacency_list()
        sorted_adj = s._sort_edges_by_angle(adj)

        center_id = ids["center"]

        # To test finding the next edge in a loop, we simulate arriving
        # at the center.
        #
        # Scene: We arrive at Center from the Right.
        # The entity 'l_right' connects Center(0,0) and Right(10,0).
        # We traveled Right -> Center. This is the reverse direction of
        # l_right. So incoming_fwd = False.
        #
        # To continue the loop CCW (keeping face on left), we need to make a
        # LEFT turn at the Center.
        # In: (-1, 0) (Right -> Center)
        # Options: Up (0, 1), Left (-1, 0), Down (0, -1), Right (1, 0).
        # A left turn relative to (-1, 0) is Down (0, -1).
        # Note: In standard Cartesian coords:
        # Vector A=(-1,0). Vector B=(0,-1).
        #   Cross Z = (-1*-1) - 0 = 1 (Positive/CCW).
        # Vector B=(0,1) (Up). Cross Z = (-1*1) - 0 = -1 (Negative/CW).
        # So Down is the correct Left turn.

        next_edge = s._get_next_edge_ccw(
            center_id, ids["l_right"], False, sorted_adj
        )
        assert next_edge["id"] == ids["l_down"]

    def test_calculate_loop_signed_area(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(10, 0)
        p3 = s.add_point(10, 10)
        p4 = s.add_point(0, 10)
        l1 = s.add_line(p1, p2)
        l2 = s.add_line(p2, p3)
        l3 = s.add_line(p3, p4)
        l4 = s.add_line(p4, p1)

        # CCW Loop: P1->P2->P3->P4->P1
        ccw_loop = [(l1, True), (l2, True), (l3, True), (l4, True)]
        area_ccw = s._calculate_loop_signed_area(ccw_loop)
        assert area_ccw == pytest.approx(100.0)

        # CW Loop: P1->P4->P3->P2->P1
        # L4 is P4->P1. False is P1->P4.
        # L3 is P3->P4. False is P4->P3.
        # L2 is P2->P3. False is P3->P2.
        # L1 is P1->P2. False is P2->P1.
        cw_loop = [(l4, False), (l3, False), (l2, False), (l1, False)]
        area_cw = s._calculate_loop_signed_area(cw_loop)
        assert area_cw == pytest.approx(-100.0)

        # Degenerate Loop (back and forth)
        degen_loop = [(l1, True), (l1, False)]
        area_degen = s._calculate_loop_signed_area(degen_loop)
        assert area_degen == pytest.approx(0.0)


def test_sketch_find_all_closed_loops():
    """Tests the core graph traversal algorithm for finding faces."""

    def get_loop_ids(loops):
        """Helper to get sets of entity IDs for easy comparison."""
        return [set(item[0] for item in loop) for loop in loops]

    # --- Test Case 1: Simple Square ---
    s1 = Sketch()
    p1 = s1.add_point(0, 0)
    p2 = s1.add_point(10, 0)
    p3 = s1.add_point(10, 10)
    p4 = s1.add_point(0, 10)
    l1 = s1.add_line(p1, p2)
    l2 = s1.add_line(p2, p3)
    l3 = s1.add_line(p3, p4)
    l4 = s1.add_line(p4, p1)
    loops1 = s1._find_all_closed_loops()
    assert len(loops1) == 1
    assert get_loop_ids(loops1)[0] == {l1, l2, l3, l4}

    # --- Test Case 2: Dangling Edge ---
    s2 = Sketch()
    p1 = s2.add_point(0, 0)
    p2 = s2.add_point(10, 0)
    p3 = s2.add_point(10, 10)
    p4 = s2.add_point(0, 10)
    p5 = s2.add_point(20, 0)
    l1 = s2.add_line(p1, p2)
    l2 = s2.add_line(p2, p3)
    l3 = s2.add_line(p3, p4)
    l4 = s2.add_line(p4, p1)
    s2.add_line(p2, p5)  # Dangling line, no assignment needed
    loops2 = s2._find_all_closed_loops()
    assert len(loops2) == 1
    assert get_loop_ids(loops2)[0] == {l1, l2, l3, l4}

    # --- Test Case 3: Two Separate Shapes ---
    s3 = Sketch()
    # Shape A
    p1 = s3.add_point(0, 0)
    p2 = s3.add_point(1, 0)
    p3 = s3.add_point(1, 1)
    p4 = s3.add_point(0, 1)
    la1 = s3.add_line(p1, p2)
    la2 = s3.add_line(p2, p3)
    la3 = s3.add_line(p3, p4)
    la4 = s3.add_line(p4, p1)
    # Shape B
    p5 = s3.add_point(10, 10)
    p6 = s3.add_point(11, 10)
    p7 = s3.add_point(11, 11)
    p8 = s3.add_point(10, 11)
    lb1 = s3.add_line(p5, p6)
    lb2 = s3.add_line(p6, p7)
    lb3 = s3.add_line(p7, p8)
    lb4 = s3.add_line(p8, p5)
    loops3 = s3._find_all_closed_loops()
    assert len(loops3) == 2
    loop_sets3 = get_loop_ids(loops3)
    assert {la1, la2, la3, la4} in loop_sets3
    assert {lb1, lb2, lb3, lb4} in loop_sets3

    # --- Test Case 4: Figure-Eight (Shared Point/Edge) ---
    s4 = Sketch()
    # Left loop
    p1 = s4.add_point(-1, -1)
    p2 = s4.add_point(0, -1)
    p3 = s4.add_point(0, 0)  # Shared point
    p4 = s4.add_point(-1, 0)
    ll1 = s4.add_line(p1, p2)
    ll2 = s4.add_line(p2, p3)
    ll3 = s4.add_line(p3, p4)
    ll4 = s4.add_line(p4, p1)
    # Right loop
    p5 = s4.add_point(1, 0)
    p6 = s4.add_point(1, -1)
    rl1 = s4.add_line(p3, p5)
    rl2 = s4.add_line(p5, p6)
    rl3 = s4.add_line(p6, p2)
    loops4 = s4._find_all_closed_loops()
    assert len(loops4) == 2
    loop_sets4 = get_loop_ids(loops4)
    assert {ll1, ll2, ll3, ll4} in loop_sets4
    assert {rl1, rl2, rl3, ll2} in loop_sets4  # ll2 is shared edge

    # --- Test Case 5: Shape with an Arc (D-Shape) ---
    s5 = Sketch()
    p1 = s5.add_point(-10, 0)
    p2 = s5.add_point(10, 0)
    c = s5.add_point(0, 0)
    l1 = s5.add_line(p1, p2)
    a1 = s5.add_arc(p2, p1, c, clockwise=False)  # Semicircle
    loops5 = s5._find_all_closed_loops()
    assert len(loops5) == 1
    assert get_loop_ids(loops5)[0] == {l1, a1}

    # --- Test Case 6: Circle ---
    s6 = Sketch()
    c = s6.add_point(0, 0)
    r = s6.add_point(10, 0)
    c1 = s6.add_circle(c, r)
    loops6 = s6._find_all_closed_loops()
    assert len(loops6) == 1
    assert get_loop_ids(loops6)[0] == {c1}

    # --- Test Case 7: Shape with Hole ---
    s7 = Sketch()
    # Outer
    op1 = s7.add_point(0, 0)
    op2 = s7.add_point(10, 0)
    op3 = s7.add_point(10, 10)
    op4 = s7.add_point(0, 10)
    ol1 = s7.add_line(op1, op2)
    ol2 = s7.add_line(op2, op3)
    ol3 = s7.add_line(op3, op4)
    ol4 = s7.add_line(op4, op1)
    # Inner
    ip1 = s7.add_point(2, 2)
    ip2 = s7.add_point(8, 2)
    ip3 = s7.add_point(8, 8)
    ip4 = s7.add_point(2, 8)
    il1 = s7.add_line(ip1, ip2)
    il2 = s7.add_line(ip2, ip3)
    il3 = s7.add_line(ip3, ip4)
    il4 = s7.add_line(ip4, ip1)
    loops7 = s7._find_all_closed_loops()
    assert len(loops7) == 2
    loop_sets7 = get_loop_ids(loops7)
    assert {ol1, ol2, ol3, ol4} in loop_sets7
    assert {il1, il2, il3, il4} in loop_sets7


def test_find_all_closed_loops_with_circle_hole():
    """Tests finding loops when a circle is used as a hole."""
    s = Sketch()
    # Outer square
    p1 = s.add_point(-10, -10)
    p2 = s.add_point(10, -10)
    p3 = s.add_point(10, 10)
    p4 = s.add_point(-10, 10)
    l1 = s.add_line(p1, p2)
    l2 = s.add_line(p2, p3)
    l3 = s.add_line(p3, p4)
    l4 = s.add_line(p4, p1)
    # Inner circle
    c = s.add_point(0, 0)
    r = s.add_point(5, 0)
    c1 = s.add_circle(c, r)

    loops = s._find_all_closed_loops()
    assert len(loops) == 2

    loop_sets = [set(item[0] for item in loop) for loop in loops]
    assert {l1, l2, l3, l4} in loop_sets
    assert {c1} in loop_sets

    # Verify area calculation is correct for the circle loop
    circle_loop = next(loop for loop in loops if loop[0][0] == c1)
    area = s._calculate_loop_signed_area(circle_loop)
    assert area == pytest.approx(math.pi * 5**2)


def test_sketch_fill_geometry_generation_circle():
    """Test generating fill geometry for a single Circle loop."""
    sketch = Sketch()
    c = sketch.add_point(0, 0)
    r = sketch.add_point(10, 0)

    # Create circle
    cid = sketch.add_circle(c, r)

    # Define fill
    fill = Fill(uid="fill1", boundary=[(cid, True)])
    sketch.fills.append(fill)

    # Generate geometries
    geos = sketch.get_fill_geometries()

    assert len(geos) == 1
    geo = geos[0]
    data = geo.data
    assert data is not None

    # Check commands: should be 1 MoveTo + 2 ArcTo (semicircles)
    assert len(geo) == 3

    # Start at radius point (10, 0)
    assert (data[0, COL_X : COL_Z + 1] == (10.0, 0.0, 0.0)).all()

    # First arc to (-10, 0). Center is (0,0). Offset from (10,0) is (-10, 0).
    arc1 = data[1]
    assert arc1[COL_TYPE] == CMD_TYPE_ARC
    assert (arc1[COL_X : COL_Z + 1] == (-10.0, 0.0, 0.0)).all()
    assert (arc1[COL_I : COL_J + 1] == (-10.0, 0.0)).all()
    assert not bool(arc1[COL_CW])

    # Second arc to (10, 0). Offset from (-10,0) is (10, 0).
    arc2 = data[2]
    assert arc2[COL_TYPE] == CMD_TYPE_ARC
    assert (arc2[COL_X : COL_Z + 1] == (10.0, 0.0, 0.0)).all()
    assert (arc2[COL_I : COL_J + 1] == (10.0, 0.0)).all()
    assert not bool(arc2[COL_CW])


def test_sketch_fill_geometry_generation_rect():
    """Test generating fill geometry for a rectangle (multi-segment lines)."""
    sketch = Sketch()
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(10, 0)
    p3 = sketch.add_point(10, 10)
    p4 = sketch.add_point(0, 10)

    l1 = sketch.add_line(p1, p2)
    l2 = sketch.add_line(p2, p3)
    l3 = sketch.add_line(p3, p4)
    l4 = sketch.add_line(p4, p1)

    # Define fill loop
    boundary = [
        (l1, True),  # p1 -> p2
        (l2, True),  # p2 -> p3
        (l3, True),  # p3 -> p4
        (l4, True),  # p4 -> p1
    ]
    fill = Fill(uid="fill_rect", boundary=boundary)
    sketch.fills.append(fill)

    geos = sketch.get_fill_geometries()
    assert len(geos) == 1
    geo = geos[0]
    data = geo.data
    assert data is not None

    assert len(geo) == 5  # MoveTo + 4 LineTo

    # Check start
    assert (data[0, COL_X : COL_Z + 1] == (0.0, 0.0, 0.0)).all()

    # Check sequence
    assert data[1, COL_TYPE] == CMD_TYPE_LINE
    assert (data[1, COL_X : COL_Z + 1] == (10.0, 0.0, 0.0)).all()

    assert data[2, COL_TYPE] == CMD_TYPE_LINE
    assert (data[2, COL_X : COL_Z + 1] == (10.0, 10.0, 0.0)).all()

    assert data[3, COL_TYPE] == CMD_TYPE_LINE
    assert (data[3, COL_X : COL_Z + 1] == (0.0, 10.0, 0.0)).all()

    assert data[4, COL_TYPE] == CMD_TYPE_LINE
    assert (data[4, COL_X : COL_Z + 1] == (0.0, 0.0, 0.0)).all()


def test_sketch_fill_geometry_generation_arc_shape():
    """Test fill with mixed Lines and Arcs, including reverse traversal."""
    sketch = Sketch()

    # Shape: A semi-circle connected by a straight line.
    # Points: (-10, 0) and (10, 0). Center (0,0).

    p_left = sketch.add_point(-10, 0)
    p_right = sketch.add_point(10, 0)
    p_center = sketch.add_point(0, 0)

    # Line from Left to Right
    line_id = sketch.add_line(p_left, p_right)

    # Arc from Left to Right (top half). Clockwise.
    # Start: Left, End: Right, Center: Center.
    # If CW: goes Left -> Top -> Right.
    arc_id = sketch.add_arc(p_left, p_right, p_center, clockwise=True)

    # Loop definition:
    # 1. Line: Left -> Right (Forward)
    # 2. Arc: Right -> Left (Reverse of Arc definition Left->Right)

    boundary = [
        (line_id, True),  # (-10,0) -> (10,0)
        (arc_id, False),  # (10,0) -> (-10,0) via arc
    ]

    fill = Fill(uid="fill_arc", boundary=boundary)
    sketch.fills.append(fill)

    geos = sketch.get_fill_geometries()
    assert len(geos) == 1
    geo = geos[0]
    data = geo.data
    assert data is not None

    assert len(geo) == 3  # MoveTo + LineTo + ArcTo

    # 1. MoveTo Start of Line (-10, 0)
    assert (data[0, COL_X : COL_Z + 1] == (-10.0, 0.0, 0.0)).all()

    # 2. LineTo End of Line (10, 0)
    assert data[1, COL_TYPE] == CMD_TYPE_LINE
    assert (data[1, COL_X : COL_Z + 1] == (10.0, 0.0, 0.0)).all()

    # 3. ArcTo back to (-10, 0)
    # Entity is CW. Traversal is Reverse.
    # In `get_fill_geometries`: is_cw = not entity.clockwise if not fwd.
    # So is_cw should be False (CCW).

    cmd_arc = data[2]
    assert cmd_arc[COL_TYPE] == CMD_TYPE_ARC
    assert (cmd_arc[COL_X : COL_Z + 1] == (-10.0, 0.0, 0.0)).all()

    # Center offset relative to current point (10, 0). Center is (0,0).
    # Offset = (0 - 10, 0 - 0) = (-10, 0)
    assert (cmd_arc[COL_I : COL_J + 1] == (-10.0, 0.0)).all()

    # Direction check
    assert not bool(cmd_arc[COL_CW])
