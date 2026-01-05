import pytest
import numpy as np
from scipy.optimize import check_grad
import math
from types import SimpleNamespace
from rayforge.core.sketcher.constraints import PerpendicularConstraint
from rayforge.core.sketcher.params import ParameterContext
from rayforge.core.sketcher.registry import EntityRegistry


@pytest.fixture
def setup_env():
    reg = EntityRegistry()
    params = ParameterContext()
    return reg, params


def test_perpendicular_constraint_line_line(setup_env):
    reg, params = setup_env

    # Line 1: Horizontal (0,0) -> (10,0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    # Line 2: Vertical (5,5) -> (5,15). Vector is (0, 10)
    p3 = reg.add_point(5, 5)
    p4 = reg.add_point(5, 15)
    l2 = reg.add_line(p3, p4)

    c = PerpendicularConstraint(l1, l2)
    # Dot product: (10, 0) . (0, 10) = 0
    assert c.error(reg, params) == pytest.approx(0.0)

    # Make Line 2 NOT perpendicular (Slope 1)
    # Move p4 to (15, 15). Vector (10, 10)
    reg.get_point(p4).x = 15

    # Dot product: (10, 0) . (10, 10) = 100
    assert c.error(reg, params) == pytest.approx(100.0)


def test_perpendicular_constraint_extended(setup_env):
    """Tests perpendicular constraint for Line-Circle and Circle-Circle."""
    reg, params = setup_env

    # --- Line-Circle Test ---
    # Circle at (10,10), radius 5
    c1_p = reg.add_point(10, 10)
    c1_r = reg.add_point(15, 10)
    circ1 = reg.add_circle(c1_p, c1_r)

    # Line passing through center (0,10) -> (20,10)
    l1_p1 = reg.add_point(0, 10)
    l1_p2 = reg.add_point(20, 10)
    line1 = reg.add_line(l1_p1, l1_p2)

    # Error is cross product of (L2-L1) and (C-L1)
    # (20,0) x (10-0, 10-10) = 20*0 - 10*0 = 0
    lc_constraint = PerpendicularConstraint(line1, circ1)
    assert lc_constraint.error(reg, params) == pytest.approx(0.0)

    # Move line so it doesn't pass through center
    reg.get_point(l1_p1).y = 0
    reg.get_point(l1_p2).y = 0
    # Line is now (0,0)->(20,0). Center is (10,10).
    # Vector L2-L1: (20, 0)
    # Vector C-L1: (10, 10)
    # Cross product: (20 * 10) - (10 * 0) = 200
    assert lc_constraint.error(reg, params) == pytest.approx(200.0)

    # --- Circle-Circle Test ---
    # C1 at (0,0), radius 3 (r^2=9)
    c2_p = reg.add_point(0, 0)
    c2_r = reg.add_point(3, 0)
    circ2 = reg.add_circle(c2_p, c2_r)

    # C2 at (5,0), radius 4 (r^2=16), distance between centers = 5 (d^2=25)
    c3_p = reg.add_point(5, 0)
    c3_r = reg.add_point(9, 0)  # 5+4
    circ3 = reg.add_circle(c3_p, c3_r)

    # Error is r1^2 + r2^2 - d^2 = 9 + 16 - 25 = 0
    cc_constraint = PerpendicularConstraint(circ2, circ3)
    assert cc_constraint.error(reg, params) == pytest.approx(0.0)

    # Move C2 center to (6,0), d^2 = 36
    reg.get_point(c3_p).x = 6.0
    # Radius of circ3 changes! New radius is (9-6)=3, so r2^2=9.
    # Error = r1^2 + new_r2^2 - d^2 = 9 + 9 - 36 = -18
    assert cc_constraint.error(reg, params) == pytest.approx(-18.0)


def test_perpendicular_constraint_invalid_type(setup_env):
    reg, params = setup_env
    p1 = reg.add_point(0, 0)
    l1 = reg.add_line(p1, p1)  # Dummy line

    # Pass Point ID instead of Line ID
    c = PerpendicularConstraint(l1, p1)
    assert c.error(reg, params) == 0.0


def test_perpendicular_constraint_gradient(setup_env):
    reg, params = setup_env
    p0 = reg.add_point(0, 0)
    p1 = reg.add_point(10, 2)
    p2 = reg.add_point(5, 10)
    p3 = reg.add_point(3, -5)
    l1 = reg.add_line(p0, p1)
    l2 = reg.add_line(p2, p3)
    mutable_pids = [p0, p1, p2, p3]

    constraint = PerpendicularConstraint(e1_id=l1, e2_id=l2)
    pid_to_idx_map = {pid: i for i, pid in enumerate(mutable_pids)}

    def update_state_from_vec(x_vec):
        for pid, i in pid_to_idx_map.items():
            pt = reg.get_point(pid)
            pt.x = x_vec[i * 2]
            pt.y = x_vec[i * 2 + 1]

    def func_wrapper(x_vec):
        update_state_from_vec(x_vec)
        return constraint.error(reg, params)

    def grad_wrapper(x_vec):
        update_state_from_vec(x_vec)
        grad_map = constraint.gradient(reg, params)
        grad_vec = np.zeros_like(x_vec)
        for pid, grads in grad_map.items():
            if pid in pid_to_idx_map:
                idx = pid_to_idx_map[pid] * 2
                dx, dy = grads[0]
                grad_vec[idx] = dx
                grad_vec[idx + 1] = dy
        return grad_vec

    x0 = np.array([0, 0, 10, 2, 5, 10, 3, -5], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_perpendicular_gradient_with_shared_points(setup_env):
    reg, params = setup_env
    p0 = reg.add_point(0, 0)  # p0: C1 center AND C2 radius point
    p1 = reg.add_point(3, 0)  # p1: C1 radius point
    p2 = reg.add_point(5, 0)  # p2: C2 center
    c1 = reg.add_circle(p0, p1)
    c2 = reg.add_circle(p2, p0)
    mutable_pids = [p0, p1, p2]

    constraint = PerpendicularConstraint(e1_id=c1, e2_id=c2)
    pid_to_idx_map = {pid: i for i, pid in enumerate(mutable_pids)}

    def update_state_from_vec(x_vec):
        for pid, i in pid_to_idx_map.items():
            pt = reg.get_point(pid)
            pt.x = x_vec[i * 2]
            pt.y = x_vec[i * 2 + 1]

    def func_wrapper(x_vec):
        update_state_from_vec(x_vec)
        return constraint.error(reg, params)

    def grad_wrapper(x_vec):
        update_state_from_vec(x_vec)
        grad_map = constraint.gradient(reg, params)
        grad_vec = np.zeros_like(x_vec)
        for pid, grads in grad_map.items():
            if pid in pid_to_idx_map:
                idx = pid_to_idx_map[pid] * 2
                dx, dy = grads[0]
                grad_vec[idx] = dx
                grad_vec[idx + 1] = dy
        return grad_vec

    x0 = np.array([0, 0, 3, 0, 5, 0], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_perpendicular_constraint_line_circle_gradient(setup_env):
    reg, params = setup_env
    l1p1 = reg.add_point(0, 5)
    l1p2 = reg.add_point(20, 5)
    c1p = reg.add_point(10, 10)
    c1r = reg.add_point(15, 10)
    line1 = reg.add_line(l1p1, l1p2)
    circ1 = reg.add_circle(c1p, c1r)
    mutable_pids = [l1p1, l1p2, c1p]

    constraint = PerpendicularConstraint(line1, circ1)
    pid_to_idx_map = {pid: i for i, pid in enumerate(mutable_pids)}

    def update_state_from_vec(x_vec):
        for pid, i in pid_to_idx_map.items():
            pt = reg.get_point(pid)
            pt.x = x_vec[i * 2]
            pt.y = x_vec[i * 2 + 1]

    def func_wrapper(x_vec):
        update_state_from_vec(x_vec)
        return constraint.error(reg, params)

    def grad_wrapper(x_vec):
        update_state_from_vec(x_vec)
        grad_map = constraint.gradient(reg, params)
        grad_vec = np.zeros_like(x_vec)
        for pid, grads in grad_map.items():
            if pid in pid_to_idx_map:
                idx = pid_to_idx_map[pid] * 2
                dx, dy = grads[0]
                grad_vec[idx] = dx
                grad_vec[idx + 1] = dy
        return grad_vec

    x0 = np.array([0, 5, 20, 5, 10, 10], dtype=float)
    diff = check_grad(func_wrapper, grad_wrapper, x0, epsilon=1e-6)
    assert diff < 1e-5


def test_perpendicular_constraint_serialization_round_trip(setup_env):
    reg, params = setup_env

    # Line 1: Horizontal (0,0) -> (10,0)
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    l1 = reg.add_line(p1, p2)

    # Line 2: Vertical (5,5) -> (5,15)
    p3 = reg.add_point(5, 5)
    p4 = reg.add_point(5, 15)
    l2 = reg.add_line(p3, p4)

    # Create original constraint
    original = PerpendicularConstraint(l1, l2)

    # Serialize to dict
    serialized = original.to_dict()

    # Deserialize from dict
    restored = PerpendicularConstraint.from_dict(serialized)

    # Check that the restored constraint has the same error
    assert original.error(reg, params) == restored.error(reg, params)


def test_perpendicular_is_hit(setup_env):
    reg, params = setup_env

    def to_screen(pos):
        return pos

    mock_element = SimpleNamespace()
    threshold = 15.0

    # --- Test Line-Line case ---
    l1p1 = reg.add_point(0, 50)
    l1p2 = reg.add_point(100, 50)
    l1 = reg.add_line(l1p1, l1p2)

    l2p1 = reg.add_point(50, 0)
    l2p2 = reg.add_point(50, 100)
    l2 = reg.add_line(l2p1, l2p2)
    c_ll = PerpendicularConstraint(l1, l2)

    # Visuals are centered at intersection (50, 50)
    # The dot is placed at mid-angle with radius 0.6 * 16 = 9.6
    # ang1 ~ 0 (from (100,50)), ang2 ~ PI/2 (from (50,100))
    # mid-angle is ~PI/4. Dot pos ~ (50+9.6*cos(PI/4), 50+9.6*sin(PI/4))
    # ~ (50+6.8, 50+6.8) = (56.8, 56.8)
    hit_x, hit_y = (
        50 + 9.6 * math.cos(math.pi / 4),
        50 + 9.6 * math.sin(math.pi / 4),
    )
    assert (
        c_ll.is_hit(hit_x, hit_y, reg, to_screen, mock_element, threshold)
        is True
    )
    assert c_ll.is_hit(0, 0, reg, to_screen, mock_element, threshold) is False

    # --- Test Line-Circle case ---
    c1p = reg.add_point(150, 50)
    c1r = reg.add_point(160, 50)
    circ1 = reg.add_circle(c1p, c1r)
    c_lc = PerpendicularConstraint(l1, circ1)

    # Visuals are a box at the intersection of line and circle.
    # No intersection, so it defaults to the center of the circle (150, 50).
    assert (
        c_lc.is_hit(150, 50, reg, to_screen, mock_element, threshold) is True
    )
    assert c_lc.is_hit(0, 0, reg, to_screen, mock_element, threshold) is False
