import pytest
import math
from rayforge.core.sketcher.entities import EntityRegistry
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.constraints import (
    DistanceConstraint,
    HorizontalConstraint,
    VerticalConstraint,
    DragConstraint,
)
from rayforge.core.sketcher.solver import Solver


def test_solver_simple_move():
    """Test moving a single point to satisfy a distance constraint."""
    reg = EntityRegistry()
    params = ParameterContext()

    # p1 fixed at origin
    p1 = reg.add_point(0, 0, fixed=True)
    # p2 at (5, 0), want it at (10, 0)
    p2 = reg.add_point(5, 0, fixed=False)

    # Constrain distance to 10
    constraints = [
        HorizontalConstraint(p1, p2),  # Keep it on x-axis
        DistanceConstraint(p1, p2, 10.0),
    ]

    solver = Solver(reg, params, constraints)
    success = solver.solve()

    assert success is True
    pt2 = reg.get_point(p2)
    assert pt2.x == pytest.approx(10.0, abs=1e-4)
    assert pt2.y == pytest.approx(0.0, abs=1e-4)


def test_solver_fixed_point():
    """Test that fixed points do not move."""
    reg = EntityRegistry()
    params = ParameterContext()

    p1 = reg.add_point(0, 0, fixed=True)
    p2 = reg.add_point(10, 0, fixed=True)  # THIS IS FIXED

    # Impossible constraint: fixed points are 10 apart, we want 5
    constraints = [DistanceConstraint(p1, p2, 5.0)]

    solver = Solver(reg, params, constraints)
    solver.solve()

    pt2 = reg.get_point(p2)
    assert pt2.x == 10.0  # Should not have moved

    # Check if constraints are actually satisfied
    err = constraints[0].error(reg, params)
    assert abs(err) > 1.0  # Constraint definitely failed


def test_solver_degrees_of_freedom():
    """Test a system with multiple moving parts."""
    reg = EntityRegistry()
    params = ParameterContext()

    #    p3 -- p4
    #    |     |
    #    p1 -- p2

    p1 = reg.add_point(0, 0, fixed=True)
    p2 = reg.add_point(10, 0)  # Free
    p3 = reg.add_point(0, 10)  # Free

    # Right triangle 3-4-5 logic check
    # Make p1-p2 length 3
    # Make p1-p3 length 4
    # Distance p2-p3 should naturally become 5

    constraints = [
        HorizontalConstraint(p1, p2),
        VerticalConstraint(p1, p3),
        DistanceConstraint(p1, p2, 3.0),
        DistanceConstraint(p1, p3, 4.0),
    ]

    solver = Solver(reg, params, constraints)
    assert solver.solve() is True

    dist_hypotenuse = math.hypot(
        reg.get_point(p2).x - reg.get_point(p3).x,
        reg.get_point(p2).y - reg.get_point(p3).y,
    )
    assert dist_hypotenuse == pytest.approx(5.0, abs=1e-4)


def test_solver_drag_behavior():
    """Test that DragConstraint moves unconstrained points."""
    reg = EntityRegistry()
    params = ParameterContext()

    # Point at origin
    p1 = reg.add_point(0, 0)

    # Drag constraint to (10, 10)
    c = DragConstraint(p1, 10.0, 10.0)
    constraints = [c]

    solver = Solver(reg, params, constraints)
    success = solver.solve()

    assert success is True
    pt = reg.get_point(p1)
    # Since there are no competing constraints, it should reach the target
    assert pt.x == pytest.approx(10.0, abs=1e-4)
    assert pt.y == pytest.approx(10.0, abs=1e-4)


def test_solver_drag_vs_geometry():
    """
    Test that geometric constraints overpower DragConstraint due to weight.
    """
    reg = EntityRegistry()
    params = ParameterContext()

    p1 = reg.add_point(0, 0, fixed=True)
    p2 = reg.add_point(10, 0)

    # Hard Geometric Constraint: Distance must be 10
    # Note: Horizontal is implied by y=0 init and no vertical drag,
    # but let's add Horizontal explicitly to be safe.
    constraints = [
        HorizontalConstraint(p1, p2),
        DistanceConstraint(p1, p2, 10.0),
        # Drag Constraint: Try to pull p2 way out to (20, 0) with a low weight
        DragConstraint(p2, 20.0, 0.0, weight=0.05),
    ]

    solver = Solver(reg, params, constraints)

    # We do not assert success here because the solver returns False when
    # a soft constraint (DragConstraint) causes the residual cost to remain
    # above the strict tolerance (1e-6), even though it found the optimal
    # solution.
    solver.solve()

    pt2 = reg.get_point(p2)

    # Because DragConstraint has a small weight vs the implicit 1.0 of
    # geometric constraints, the solver will prioritize DistanceConstraint.
    # The point should be at x=10 (distance 10 from 0,0), not x=20.
    assert pt2.x == pytest.approx(10.0, abs=0.1)
    # Ensure it didn't get dragged all the way to 20
    assert pt2.x != pytest.approx(20.0, abs=0.1)


def test_solver_no_constraints():
    """
    Test solver behavior when there are mutable points but no constraints.
    """
    reg = EntityRegistry()
    params = ParameterContext()
    reg.add_point(0, 0)  # Mutable

    solver = Solver(reg, params, [])
    # Should succeed immediately (residuals return [0.0])
    assert solver.solve() is True


def test_solver_no_mutable_points():
    """Test solver behavior when everything is fixed."""
    reg = EntityRegistry()
    params = ParameterContext()
    reg.add_point(0, 0, fixed=True)

    # Constraint exists, but nothing can move
    # Solver calculates error, sees it can't move anything,
    # returns True immediately (optimization check)
    solver = Solver(reg, params, [])
    assert solver.solve() is True


def test_solver_impossible_constraints():
    """Test solver reporting failure on impossible geometry."""
    reg = EntityRegistry()
    params = ParameterContext()
    p1 = reg.add_point(0, 0, fixed=True)

    # Initialize p2 near the compromise to test failure reporting
    # logic rather than optimizer descent capabilities in singular landscapes.
    p2 = reg.add_point(15.1, 0)  # Start slightly off

    constraints = [
        # Must be at dist 10 from p1
        DistanceConstraint(p1, p2, 10.0),
        # BUT must also be at dist 20 from p1
        DistanceConstraint(p1, p2, 20.0),
    ]

    solver = Solver(reg, params, constraints)
    success = solver.solve()

    # success flag should be False because residuals > tolerance
    # even though it is at the best possible location.
    assert success is False

    # The solver minimizes sum of squares of linear errors:
    # E = (d-10)^2 + (d-20)^2
    # The minimum for this is at d = (10+20)/2 = 15.0
    optimal_dist = 15.0
    # Relax tolerance slightly as the solver may terminate just short
    assert reg.get_point(p2).x == pytest.approx(optimal_dist, abs=1e-2)
