import pytest
import numpy as np
import math
import copy
from rayforge.core.matrix import Matrix


class TestMatrix:
    def test_initialization(self):
        # Default initialization should be identity
        m1 = Matrix()
        assert m1 == Matrix(np.identity(3))

        # Initialization from list
        list_data = [[1, 2, 3], [4, 5, 6], [0, 0, 1]]
        m2 = Matrix(list_data)
        assert np.array_equal(m2.m, np.array(list_data))

        # Initialization from another Matrix
        m3 = Matrix(m2)
        assert m3 == m2
        assert m3 is not m2  # Should be a new instance
        assert m3.m is not m2.m  # Internal array should be a copy

        # Invalid initialization
        with pytest.raises(ValueError):
            Matrix([[1, 2], [3, 4]])  # Wrong shape

    def test_equality(self):
        m1 = Matrix.translation(10, 20)
        m2 = Matrix.translation(10, 20)
        m3 = Matrix.translation(10, 21)
        assert m1 == m2
        assert m1 != m3
        assert m1 != "not a matrix"

    def test_copying(self):
        m1 = Matrix.rotation(45)

        # Test shallow copy
        m2 = copy.copy(m1)
        assert m1 == m2
        assert m1 is not m2
        assert m1.m is not m2.m

        # Test deep copy
        m3 = copy.deepcopy(m1)
        assert m1 == m3
        assert m1 is not m3
        assert m1.m is not m3.m

    def test_representation(self):
        m = Matrix.translation(10, -20.5)
        # Test __repr__
        m_repr = repr(m)
        m_from_repr = eval(m_repr)
        assert m == m_from_repr

        # Test __str__
        assert str(m) == str(
            np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -20.5], [0.0, 0.0, 1.0]])
        )

    def test_identity(self):
        ident = Matrix.identity()
        p = (123, 456)
        assert ident.transform_point(p) == pytest.approx(p)

    def test_is_identity(self):
        """Test the is_identity() method."""
        assert Matrix.identity().is_identity() is True
        assert Matrix().is_identity() is True
        # Test near-identity for floating point robustness
        m_near = Matrix([[1, 0, 1e-10], [0, 1, 0], [0, 0, 1]])
        assert m_near.is_identity() is True
        # Test non-identity
        assert Matrix.translation(1, 0).is_identity() is False
        assert Matrix.rotation(1).is_identity() is False

    def test_translation(self):
        m = Matrix.translation(50, -30)
        assert m.transform_point((0, 0)) == pytest.approx((50, -30))
        assert m.transform_point((10, 10)) == pytest.approx((60, -20))

    def test_get_translation(self):
        m1 = Matrix.translation(123, -456)
        assert m1.get_translation() == pytest.approx((123, -456))

        # Composition: T @ R @ S applies S -> R -> T
        m2 = (
            Matrix.translation(10, 20)
            @ Matrix.rotation(45)
            @ Matrix.scale(2, 2)
        )
        tx, ty = m2.get_translation()
        assert tx == pytest.approx(10)
        assert ty == pytest.approx(20)

    def test_without_translation(self):
        """Tests the without_translation() method."""
        m1 = Matrix.translation(100, 200) @ Matrix.rotation(45)
        assert m1.get_translation() != (0, 0)

        m2 = m1.without_translation()
        assert m2.get_translation() == pytest.approx((0, 0))

        # Check that the linear part is unchanged
        assert np.allclose(m1.m[0:2, 0:2], m2.m[0:2, 0:2])

        # Check against a manually created matrix
        m_linear_only = Matrix.rotation(45)
        assert m2 == m_linear_only

    def test_scale(self):
        # Scale around origin
        m = Matrix.scale(2, 3)
        assert m.transform_point((10, 10)) == pytest.approx((20, 30))

        # Scale around a center point
        m_center = Matrix.scale(2, 3, center=(10, 10))
        assert m_center.transform_point((10, 10)) == pytest.approx((10, 10))
        assert m_center.transform_point((20, 15)) == pytest.approx((30, 25))

    def test_get_scale(self):
        """Tests the modified get_scale() which returns signed scales."""
        # Simple positive scale
        m1 = Matrix.scale(2.5, 5.0)
        assert m1.get_scale() == pytest.approx((2.5, 5.0))

        # Test with rotation
        m2 = Matrix.rotation(30) @ Matrix.scale(2, 3)
        assert m2.get_scale() == pytest.approx((2.0, 3.0))

        # Test with negative scale (reflection)
        m3 = Matrix.scale(4, -5)
        # sx is positive, sy is negative
        assert m3.get_scale() == pytest.approx((4.0, -5.0))

        # Test with negative scale and rotation
        m4 = (
            Matrix.translation(10, 20)
            @ Matrix.rotation(45)
            @ Matrix.scale(-2, 3)
        )
        # The scale factors are (2, -3) because the decomposition finds
        # the rotation of the transformed x-axis (-135 deg) and sx is positive.
        sx, sy = m4.get_scale()
        assert sx == pytest.approx(2.0)
        assert sy == pytest.approx(-3.0)

        # Test with two negative scales (is a rotation, not a flip)
        m5 = Matrix.scale(-2, -3)
        # Note: A scale(-x, -y) is identical to rotate(180) @ scale(x, y).
        # The decomposition will find angle=180, sx=2, sy=3.
        assert m5.get_scale() == pytest.approx((2.0, 3.0))

    def test_get_abs_scale(self):
        """Tests the new get_abs_scale() method."""
        # Positive scale
        m1 = Matrix.scale(2.5, 5.0)
        assert m1.get_abs_scale() == pytest.approx((2.5, 5.0))

        # Negative scale
        m2 = Matrix.scale(-4, 5)
        assert m2.get_abs_scale() == pytest.approx((4.0, 5.0))

        # Complex transform with negative scale
        m3 = (
            Matrix.translation(10, 20)
            @ Matrix.rotation(45)
            @ Matrix.scale(2, -3)
        )
        assert m3.get_abs_scale() == pytest.approx((2.0, 3.0))

    def test_rotation(self):
        # Rotate around origin
        m90 = Matrix.rotation(90)
        assert m90.transform_point((10, 0)) == pytest.approx((0, 10))

        m180 = Matrix.rotation(180)
        assert m180.transform_point((10, 0)) == pytest.approx((-10, 0))

        # Rotate around a center point
        m_center = Matrix.rotation(90, center=(10, 10))
        assert m_center.transform_point((10, 10)) == pytest.approx((10, 10))
        assert m_center.transform_point((20, 10)) == pytest.approx((10, 20))

    def test_get_rotation(self):
        """Tests the modified get_rotation() which is robust to shear."""
        m1 = Matrix.rotation(30)
        assert m1.get_rotation() == pytest.approx(30)

        m2 = Matrix.rotation(-135)
        assert m2.get_rotation() == pytest.approx(-135)

        # Non-uniform scale should not affect rotation extraction
        m3 = Matrix.rotation(60) @ Matrix.scale(2, 5)
        assert m3.get_rotation() == pytest.approx(60)

        # Test robustness against shear
        R = Matrix.rotation(45)
        # Shear matrix: x' = x + 0.5y, y' = y
        Shear = Matrix([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
        M_sheared = R @ Shear
        # get_rotation should extract the original rotation angle
        assert M_sheared.get_rotation() == pytest.approx(45)

    def test_get_determinant_2x2(self):
        """Tests the new get_determinant_2x2() method."""
        # Identity
        assert Matrix.identity().get_determinant_2x2() == pytest.approx(1.0)
        # Simple scale
        assert Matrix.scale(2, 3).get_determinant_2x2() == pytest.approx(6.0)
        # Rotation (cos*cos - sin*(-sin) = cos^2+sin^2 = 1)
        assert Matrix.rotation(30).get_determinant_2x2() == pytest.approx(1.0)
        # Flipped scale
        assert Matrix.scale(-2, 3).get_determinant_2x2() == pytest.approx(-6.0)
        # Complex transform
        M = Matrix.rotation(45) @ Matrix.scale(2, 3)
        assert M.get_determinant_2x2() == pytest.approx(6.0)

    def test_is_flipped(self):
        """Tests the new is_flipped() method."""
        assert not Matrix.identity().is_flipped()
        assert not Matrix.scale(2, 3).is_flipped()
        assert not Matrix.rotation(45).is_flipped()

        # Flipped on one axis
        assert Matrix.scale(-1, 1).is_flipped()
        assert Matrix.scale(1, -1).is_flipped()

        # Flipped on both axes is a rotation, not a flip
        assert not Matrix.scale(-1, -1).is_flipped()

        # Complex transform with a flip
        M = Matrix.rotation(30) @ Matrix.scale(-2, 3)
        assert M.is_flipped()

        # Complex transform without a flip
        M2 = Matrix.rotation(30) @ Matrix.scale(-2, -3)
        assert not M2.is_flipped()

    def test_has_zero_scale(self):
        """Tests the has_zero_scale() method."""
        # Standard matrices should not have zero scale
        assert not Matrix.identity().has_zero_scale()
        assert not Matrix.translation(10, 20).has_zero_scale()
        assert not Matrix.rotation(45).has_zero_scale()
        assert not Matrix.scale(0.1, 0.1).has_zero_scale()

        # Matrices with near-zero scale
        assert Matrix.scale(1, 0).has_zero_scale()
        assert Matrix.scale(0, 1).has_zero_scale()
        assert Matrix.scale(1e-7, 1).has_zero_scale()
        assert not Matrix.scale(1e-5, 1).has_zero_scale()

        # Complex transform with zero scale
        m_complex = (
            Matrix.translation(100, -50)
            @ Matrix.rotation(30)
            @ Matrix.scale(1, 1e-9)
        )
        assert m_complex.has_zero_scale()

        # Test with custom tolerance
        m_borderline = Matrix.scale(0.01, 1)
        assert not m_borderline.has_zero_scale()  # Default tol=1e-6
        assert m_borderline.has_zero_scale(tolerance=0.05)

    def test_matrix_multiplication(self):
        # Order of operations: T @ R means apply R first, then apply T.
        T = Matrix.translation(100, 0)
        R = Matrix.rotation(90)

        # Apply transformations separately: Rotate then Translate
        p = (10, 20)
        p_rotated = R.transform_point(p)  # (-20, 10)
        p_final_separate = T.transform_point(
            p_rotated
        )  # (-20+100, 10+0) -> (80, 10)

        # Apply combined transformation
        M = T @ R
        p_final_combined = M.transform_point(p)

        assert p_final_combined == pytest.approx(p_final_separate)
        assert p_final_combined == pytest.approx((80, 10))

    def test_inversion(self):
        T = Matrix.translation(55, -21)
        R = Matrix.rotation(33)
        S = Matrix.scale(2, 0.5)

        M = T @ R @ S
        M_inv = M.invert()

        # M * M_inv should be the identity matrix
        ident = M @ M_inv
        assert ident == Matrix.identity()

        # Applying a transform and its inverse should return to the original
        # point
        p_start = (12, 34)
        p_transformed = M.transform_point(p_start)
        p_restored = M_inv.transform_point(p_transformed)

        assert p_restored == pytest.approx(p_start)

    def test_inversion_singular_matrix(self):
        # A matrix with 0 scale is not invertible
        M_singular = Matrix.scale(1, 0)
        with pytest.raises(np.linalg.LinAlgError):
            M_singular.invert()

    def test_transform_vector(self):
        # A standard transformation pipeline is Translate -> Rotate -> Scale
        # This corresponds to the matrix multiplication order T @ R @ S.
        m = (
            Matrix.translation(100, 200)
            @ Matrix.rotation(90)
            @ Matrix.scale(2, 2)
        )

        # Test vector transformation (ignores translation)
        # 1. Scale: (10, 0) -> (20, 0)
        # 2. Rotate 90 deg: (20, 0) -> (0, 20)
        # 3. Translation is ignored.
        v = (10, 0)
        transformed_v = m.transform_vector(v)
        assert transformed_v == pytest.approx((0, 20))

        # Compare with point transformation (includes translation)
        # 1. Scale: (10, 0) -> (20, 0)
        # 2. Rotate 90 deg: (20, 0) -> (0, 20)
        # 3. Translate: (0, 20) -> (100, 220)
        p = (10, 0)
        transformed_p = m.transform_point(p)
        assert transformed_p == pytest.approx((100, 220))

    def test_transform_rectangle(self):
        """Tests the transform_rectangle method."""
        rect = (10, 20, 30, 40)  # x, y, w, h

        # Translation
        m_trans = Matrix.translation(50, 60)
        assert m_trans.transform_rectangle(rect) == pytest.approx(
            (60, 80, 30, 40)
        )

        # Rotation by 90 degrees around origin (0,0)
        m_rot90 = Matrix.rotation(90)
        # Original corners: (10,20), (40,20), (40,60), (10,60)
        # Rotated corners: (-20,10), (-20,40), (-60,40), (-60,10)
        # Bbox: min_x=-60, min_y=10, max_x=-20, max_y=40
        # Bbox rect: (-60, 10, 40, 30)
        assert m_rot90.transform_rectangle(rect) == pytest.approx(
            (-60, 10, 40, 30)
        )

        # Scale
        m_scale = Matrix.scale(2, 3)
        # Scaled rect: (20, 60, 60, 120)
        assert m_scale.transform_rectangle(rect) == pytest.approx(
            (20, 60, 60, 120)
        )

    def test_get_axis_angles(self):
        """Tests the get_x_axis_angle and get_y_axis_angle methods."""
        # No rotation
        m_ident = Matrix.identity()
        assert m_ident.get_x_axis_angle() == pytest.approx(0)
        assert m_ident.get_y_axis_angle() == pytest.approx(90)

        # Pure rotation
        m_rot = Matrix.rotation(30)
        assert m_rot.get_x_axis_angle() == pytest.approx(30)
        assert m_rot.get_y_axis_angle() == pytest.approx(30 + 90)

        # Rotation and scale
        m_rs = Matrix.rotation(45) @ Matrix.scale(2, 1)
        assert m_rs.get_x_axis_angle() == pytest.approx(45)
        assert m_rs.get_y_axis_angle() == pytest.approx(45 + 90)

        # With shear
        # x' = x + 0.5y, y' = y
        m_shear_x = Matrix.shear(0.5, 0)
        # X-axis (y=0) is unchanged: (1,0) -> (1,0), angle is 0
        assert m_shear_x.get_x_axis_angle() == pytest.approx(0)
        # Y-axis (x=0, y=1) -> (0.5, 1), angle is atan2(1, 0.5)
        assert m_shear_x.get_y_axis_angle() == pytest.approx(
            math.degrees(math.atan2(1, 0.5))
        )

        # Rotation then shear
        m_rot_shear = Matrix.rotation(30) @ Matrix.shear(0, 0.5)
        # X-axis angle is now affected by shear
        # Y-axis angle is the base rotation
        assert m_rot_shear.get_y_axis_angle() == pytest.approx(30 + 90)
        # Check X-axis angle calculation
        x_vec = m_rot_shear.transform_vector((1, 0))
        expected_angle = math.degrees(math.atan2(x_vec[1], x_vec[0]))
        assert m_rot_shear.get_x_axis_angle() == pytest.approx(expected_angle)

    def test_decompose_simple(self):
        """Test decomposition without shear."""
        T = Matrix.translation(50, -100)
        R = Matrix.rotation(30)
        S = Matrix.scale(2, 3)

        M = T @ R @ S

        tx, ty, angle, sx, sy, skew = M.decompose()

        assert tx == pytest.approx(50)
        assert ty == pytest.approx(-100)
        assert angle == pytest.approx(30)
        assert sx == pytest.approx(2)
        assert sy == pytest.approx(3)
        assert skew == pytest.approx(0)

    def test_decompose_with_shear(self):
        """Test decomposition with a sheared matrix."""
        # Manually create a matrix with shear.
        shear_factor = 0.5
        shear_angle = math.degrees(math.atan(shear_factor))  # Approx 26.565°

        # We will create R * S * K (K is skew/shear matrix)
        # K = [[1, tan(skew_rad)], [0, 1]]
        K = Matrix([[1, shear_factor, 0], [0, 1, 0], [0, 0, 1]])

        T = Matrix.translation(10, 20)
        R = Matrix.rotation(45)
        S = Matrix.scale(2, 3)

        # Full transform: Translate -> Rotate -> Scale -> Skew
        M = T @ R @ S @ K

        tx, ty, angle, sx, sy, skew = M.decompose()

        assert tx == pytest.approx(10)
        assert ty == pytest.approx(20)
        assert angle == pytest.approx(45)
        assert sx == pytest.approx(2)
        # The sy value is not simply 3 anymore because R*S*K is complex.
        # But the skew angle should be correctly extracted.
        assert skew == pytest.approx(shear_angle, abs=1e-5)

    def test_decompose_with_reflection(self):
        """Test decomposition with negative scale (reflection)."""
        T = Matrix.translation(10, 20)
        R = Matrix.rotation(45)
        S = Matrix.scale(-2, 3)  # Flipped on the x-axis

        M = T @ R @ S

        tx, ty, angle, sx, sy, skew = M.decompose()

        # The new decomposition is stable. The transformed x-axis is
        # (-2,0) rotated by 45 deg, which is a vector at -135 deg.
        # The sx is the length of that vector (2), and the flip is
        # represented in sy.
        assert tx == pytest.approx(10)
        assert ty == pytest.approx(20)
        assert angle == pytest.approx(-135)
        assert sx == pytest.approx(2)
        assert sy == pytest.approx(-3)
        assert skew == pytest.approx(0)

    def test_set_translation(self):
        """Tests the set_translation() method."""
        m1 = Matrix.rotation(45) @ Matrix.scale(2, 3)
        m2 = m1.set_translation(100, -200)

        # Check that the new matrix has the correct translation
        assert m2.get_translation() == pytest.approx((100, -200))

        # Check that the linear part (rotation/scale) is unchanged
        # Compare the top-left 2x2 submatrices
        assert np.allclose(m1.m[0:2, 0:2], m2.m[0:2, 0:2])

        # Setting translation on an identity matrix should be like creating one
        m3 = Matrix.identity().set_translation(50, 60)
        m4 = Matrix.translation(50, 60)
        assert m3 == m4

    def test_post_translate(self):
        m_base = Matrix.rotation(90)
        m_fluent = m_base.post_translate(100, 50)
        m_manual = m_base @ Matrix.translation(100, 50)
        assert m_fluent == m_manual

        # Order of operations: translate, then rotate
        p = (10, 20)
        p_translated = (10 + 100, 20 + 50)  # (110, 70)
        p_final = m_base.transform_point(p_translated)  # (-70, 110)
        assert m_fluent.transform_point(p) == pytest.approx(p_final)

    def test_pre_translate(self):
        m_base = Matrix.rotation(90)
        m_fluent = m_base.pre_translate(100, 50)
        m_manual = Matrix.translation(100, 50) @ m_base
        assert m_fluent == m_manual

        # Order of operations: rotate, then translate
        p = (10, 20)
        p_rotated = m_base.transform_point(p)  # (-20, 10)
        p_final = (p_rotated[0] + 100, p_rotated[1] + 50)  # (80, 60)
        assert m_fluent.transform_point(p) == pytest.approx(p_final)

    def test_post_rotate(self):
        m_base = Matrix.rotation(30)
        m_fluent = m_base.post_rotate(60)
        m_manual = m_base @ Matrix.rotation(60)
        assert m_fluent == m_manual
        # Total rotation should be 30 + 60 = 90
        assert m_fluent.get_rotation() == pytest.approx(90)

    def test_pre_rotate(self):
        m_base = Matrix.rotation(30)
        m_fluent = m_base.pre_rotate(60)
        m_manual = Matrix.rotation(60) @ m_base
        assert m_fluent == m_manual
        # Total rotation should be 60 + 30 = 90
        assert m_fluent.get_rotation() == pytest.approx(90)

    def test_post_scale(self):
        m_base = Matrix.rotation(90)
        m_fluent = m_base.post_scale(2, 3)
        m_manual = m_base @ Matrix.scale(2, 3)
        assert m_fluent == m_manual

        # Order of operations: scale, then rotate
        p = (10, 20)
        p_scaled = (10 * 2, 20 * 3)  # (20, 60)
        p_final = m_base.transform_point(p_scaled)  # (-60, 20)
        assert m_fluent.transform_point(p) == pytest.approx(p_final)

    def test_pre_scale(self):
        m_base = Matrix.rotation(90)
        m_fluent = m_base.pre_scale(2, 3)
        m_manual = Matrix.scale(2, 3) @ m_base
        assert m_fluent == m_manual

        # Order of operations: rotate, then scale
        p = (10, 20)
        p_rotated = m_base.transform_point(p)  # (-20, 10)
        p_final = (p_rotated[0] * 2, p_rotated[1] * 3)  # (-40, 30)
        assert m_fluent.transform_point(p) == pytest.approx(p_final)

    def test_fluent_chaining(self):
        # Standard T @ R @ S order applies S, then R, then T.
        # This is built by pre-pending transforms to an identity matrix.
        m_chain = (
            Matrix.identity()
            .pre_scale(2, 2)
            .pre_rotate(90)
            .pre_translate(100, 0)
        )
        m_manual = (
            Matrix.translation(100, 0)
            @ Matrix.rotation(90)
            @ Matrix.scale(2, 2)
        )
        assert m_chain == m_manual

        # Verify the transformation on a point
        p_start = (10, 20)
        p_translated = (-40 + 100, 20)  # (60, 20)
        assert m_chain.transform_point(p_start) == pytest.approx(p_translated)

    def test_shear(self):
        """Tests the new shear() method."""
        # Shear along x-axis from origin
        m_x = Matrix.shear(0.5, 0)
        # x' = x + 0.5 * y, y' = y
        assert m_x.transform_point((10, 20)) == pytest.approx(
            (10 + 0.5 * 20, 20)
        )  # (20, 20)

        # Shear along y-axis from origin
        m_y = Matrix.shear(0, 0.2)
        # x' = x, y' = y + 0.2 * x
        assert m_y.transform_point((10, 20)) == pytest.approx(
            (10, 20 + 0.2 * 10)
        )  # (10, 22)

        # Shear around a center point
        center = (10, 20)
        m_center = Matrix.shear(0.5, 0, center=center)
        # Center point should be invariant
        assert m_center.transform_point(center) == pytest.approx(center)
        # Test another point
        # Point relative to center: (12, 24) - (10, 20) = (2, 4)
        # Shear relative point: x' = 2 + 0.5*4 = 4, y' = 4
        # New relative point: (4, 4)
        # Translate back: (4, 4) + (10, 20) = (14, 24)
        assert m_center.transform_point((12, 24)) == pytest.approx((14, 24))

    def test_post_shear(self):
        m_base = Matrix.rotation(90)
        m_fluent = m_base.post_shear(0.5, 0)
        m_manual = m_base @ Matrix.shear(0.5, 0)
        assert m_fluent == m_manual

        # Order of operations: shear, then rotate
        p = (10, 20)
        p_sheared = Matrix.shear(0.5, 0).transform_point(p)  # (20, 20)
        p_final = m_base.transform_point(p_sheared)  # (-20, 20)
        assert m_fluent.transform_point(p) == pytest.approx(p_final)

    def test_pre_shear(self):
        m_base = Matrix.rotation(90)
        m_fluent = m_base.pre_shear(0.5, 0)
        m_manual = Matrix.shear(0.5, 0) @ m_base
        assert m_fluent == m_manual

        # Order of operations: rotate, then shear
        p = (10, 20)
        p_rotated = m_base.transform_point(p)  # (-20, 10)
        p_final = Matrix.shear(0.5, 0).transform_point(p_rotated)  # (-15, 10)
        assert m_fluent.transform_point(p) == pytest.approx(p_final)

    def test_get_cairo(self):
        """Tests the get_cairo() method."""
        # Cairo matrix format is: (xx, yx, xy, yy, x0, y0)
        # Our matrix is: [[xx, xy, x0], [yx, yy, y0], [0, 0, 1]]
        m = Matrix([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        assert m.for_cairo() == (1, 4, 2, 5, 3, 6)

        # Test with a transform
        m_trans = Matrix.translation(50, -60)
        # [[1, 0, 50], [0, 1, -60], [0, 0, 1]]
        assert m_trans.for_cairo() == (1, 0, 0, 1, 50, -60)
