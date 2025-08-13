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

        # Composition: T @ R @ S applies T -> R -> S
        m2 = (
            Matrix.translation(10, 20)
            @ Matrix.rotation(45)
            @ Matrix.scale(2, 2)
        )
        tx, ty = m2.get_translation()
        assert tx == pytest.approx(10)
        assert ty == pytest.approx(20)

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
        # Note: numpy's pre-multiplication order means M @ M_inv is correct.
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
