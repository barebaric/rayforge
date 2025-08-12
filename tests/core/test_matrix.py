import pytest
import numpy as np
import copy
from rayforge.core.matrix import Matrix


class TestMatrix:
    def test_initialization(self):
        # Default initialization should be identity
        m1 = Matrix()
        assert m1 == Matrix(np.identity(3))

        # Initialization from list
        list_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
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

    def test_translation(self):
        m = Matrix.translation(50, -30)
        assert m.transform_point((0, 0)) == pytest.approx((50, -30))
        assert m.transform_point((10, 10)) == pytest.approx((60, -20))

    def test_get_translation(self):
        m1 = Matrix.translation(123, -456)
        assert m1.get_translation() == pytest.approx((123, -456))

        # The last matrix in the chain determines the final translation.
        # S @ R @ T applies S -> R -> T.
        m2 = (
            Matrix.scale(2, 2)
            @ Matrix.rotation(45)
            @ Matrix.translation(10, 20)
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
        m1 = Matrix.scale(2.5, 5.0)
        assert m1.get_scale() == pytest.approx((2.5, 5.0))

        # Test with rotation. For get_scale to extract the original factors
        # cleanly, the scale must be applied BEFORE the rotation.
        # This corresponds to the composition: scale first, then rotate.
        m2 = Matrix.scale(2, 3) @ Matrix.rotation(30)
        assert m2.get_scale() == pytest.approx((2.0, 3.0))

        # A more complex transform: scale -> rotate -> translate.
        # The translation should not affect the scale components.
        m3 = (
            Matrix.scale(4, 5)
            @ Matrix.rotation(45)
            @ Matrix.translation(10, 20)
        )
        assert m3.get_scale() == pytest.approx((4.0, 5.0))

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
        m1 = Matrix.rotation(30)
        assert m1.get_rotation() == pytest.approx(30)

        m2 = Matrix.rotation(-135)
        assert m2.get_rotation() == pytest.approx(-135)

        # Uniform scale does not affect rotation extraction
        m3 = Matrix.scale(2, 2) @ Matrix.rotation(60)
        assert m3.get_rotation() == pytest.approx(60)

        # Non-uniform scale applied before rotation will also not affect
        # get_rotation, which correctly finds the angle of the new x-axis.
        m4 = Matrix.scale(2, 5) @ Matrix.rotation(45)
        assert m4.get_rotation() == pytest.approx(45)

    def test_matrix_multiplication(self):
        # Order of operations: R @ T means apply R first, then apply T.
        T = Matrix.translation(100, 0)
        R = Matrix.rotation(90)

        # Apply transformations separately: Rotate then Translate
        p = (10, 20)
        p_rotated = R.transform_point(p)  # (-20, 10)
        p_final_separate = T.transform_point(
            p_rotated
        )  # (-20+100, 10+0) -> (80, 10)

        # Apply combined transformation
        M = R @ T
        p_final_combined = M.transform_point(p)

        assert p_final_combined == pytest.approx(p_final_separate)
        assert p_final_combined == pytest.approx((80, 10))

    def test_inversion(self):
        T = Matrix.translation(55, -21)
        R = Matrix.rotation(33)
        S = Matrix.scale(2, 0.5)

        M = S @ R @ T
        M_inv = M.invert()

        # M * M_inv should be the identity matrix
        ident = M_inv @ M
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
        # A standard transformation pipeline is Scale -> Rotate -> Translate.
        # This corresponds to the matrix multiplication order S @ R @ T.
        m = (
            Matrix.scale(2, 2)
            @ Matrix.rotation(90)
            @ Matrix.translation(100, 200)
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
