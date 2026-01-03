import pytest
from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.constraints import CollinearConstraint


@pytest.fixture
def sketch_with_points():
    """Create a sketch with three points for testing."""
    s = Sketch()
    p1 = s.add_point(0, 0)
    p2 = s.add_point(10, 10)
    p3 = s.add_point(20, 0)
    return s, p1, p2, p3


def test_collinear_constraint_serialization():
    """Test that the constraint can be serialized and deserialized."""
    constraint = CollinearConstraint(1, 2, 3)
    data = constraint.to_dict()
    assert data == {"type": "CollinearConstraint", "p1": 1, "p2": 2, "p3": 3}
    new_constraint = CollinearConstraint.from_dict(data)
    assert isinstance(new_constraint, CollinearConstraint)
    assert new_constraint.p1 == 1
    assert new_constraint.p2 == 2
    assert new_constraint.p3 == 3


def test_collinear_constraint_error_zero(sketch_with_points):
    """Test the error is zero for perfectly collinear points."""
    sketch, p1, p2, p3 = sketch_with_points
    # Move p3 to be on the line p1-p2
    pt3 = sketch.registry.get_point(p3)
    pt3.x = 5
    pt3.y = 5

    constraint = CollinearConstraint(p1, p2, p3)
    error = constraint.error(sketch.registry, sketch.params)
    assert abs(error) < 1e-9


def test_collinear_constraint_error_non_zero(sketch_with_points):
    """Test the error is non-zero for non-collinear points."""
    sketch, p1, p2, p3 = sketch_with_points
    # p1=(0,0), p2=(10,10), p3=(20,0)
    constraint = CollinearConstraint(p1, p2, p3)
    error = constraint.error(sketch.registry, sketch.params)

    # Expected error:
    #   (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
    # = (10 - 0) * (0 - 0) - (10 - 0) * (20 - 0) = 0 - 10 * 20 = -200
    assert pytest.approx(error) == -200.0


def test_collinear_constraint_gradient(sketch_with_points):
    """Test the gradient calculation for the constraint."""
    sketch, p1, p2, p3 = sketch_with_points
    constraint = CollinearConstraint(p1, p2, p3)
    gradient = constraint.gradient(sketch.registry, sketch.params)

    pt1 = sketch.registry.get_point(p1)
    pt2 = sketch.registry.get_point(p2)
    pt3 = sketch.registry.get_point(p3)

    # dE/dp1x = p2y - p3y = 10 - 0 = 10
    # dE/dp1y = p3x - p2x = 20 - 10 = 10
    grad_p1 = (pt2.y - pt3.y, pt3.x - pt2.x)
    assert pytest.approx(gradient[p1][0]) == grad_p1

    # dE/dp2x = p3y - p1y = 0 - 0 = 0
    # dE/dp2y = -(p3x - p1x) = -(20 - 0) = -20
    grad_p2 = (pt3.y - pt1.y, -(pt3.x - pt1.x))
    assert pytest.approx(gradient[p2][0]) == grad_p2

    # dE/dp3x = -(p2y - p1y) = -(10 - 0) = -10
    # dE/dp3y = p2x - p1x = 10 - 0 = 10
    grad_p3 = (-(pt2.y - pt1.y), pt2.x - pt1.x)
    assert pytest.approx(gradient[p3][0]) == grad_p3


def test_collinear_solver_integration(sketch_with_points):
    """Test that applying the constraint and solving moves the point."""
    sketch, p1, p2, p3 = sketch_with_points

    # Fix p1 and p2 to define the line
    sketch.registry.get_point(p1).fixed = True
    sketch.registry.get_point(p2).fixed = True

    constraint = CollinearConstraint(p1, p2, p3)
    sketch.constraints.append(constraint)

    success = sketch.solve()
    assert success is True

    # After solving, p3 should be on the line defined by p1=(0,0) and
    # p2=(10,10). The solver will find the closest point, which in this
    # case will move p3.
    # The exact final position depends on the solver's path.
    # We verify the error is now zero.
    error = constraint.error(sketch.registry, sketch.params)
    assert abs(error) < 1e-6
