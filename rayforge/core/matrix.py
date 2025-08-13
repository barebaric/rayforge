import math
from typing import Tuple, Any, Optional
import numpy as np


class Matrix:
    """
    A 3x3 affine transformation matrix for 2D graphics.

    Provides an object-oriented interface for matrix operations, including
    translations, rotations, and scaling. Uses numpy for the underlying
    calculations.
    """

    def __init__(self, data: Any = None):
        """
        Initializes a 3x3 matrix.

        Args:
            data: Can be another Matrix, a 3x3 list/tuple, a 3x3 numpy
                  array, or None to create an identity matrix.
        """
        if data is None:
            self.m: np.ndarray = np.identity(3, dtype=float)
        elif isinstance(data, Matrix):
            self.m = data.m.copy()
        else:
            try:
                self.m = np.array(data, dtype=float)
                if self.m.shape != (3, 3):
                    raise ValueError("Input data must be a 3x3 matrix.")
            except Exception as e:
                raise ValueError(f"Could not create Matrix from data: {e}")

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """
        Performs matrix multiplication: self @ other.

        This implements standard pre-multiplication, where (A @ B) @ p is
        equivalent to applying transform A, then transform B.
        M_combined = M_second @ M_first.

        Args:
            other: The matrix to multiply with on the right.

        Returns:
            The resulting new Matrix.
        """
        if not isinstance(other, Matrix):
            return NotImplemented
        # Standard pre-multiplication: self.m is the first transform,
        # other.m is the second.
        return Matrix(np.dot(self.m, other.m))

    def __eq__(self, other: Any) -> bool:
        """
        Checks for equality between two matrices.

        Uses np.allclose for floating-point comparisons.
        """
        if not isinstance(other, Matrix):
            return False
        return np.allclose(self.m, other.m)

    def __repr__(self) -> str:
        """Returns a developer-friendly, evaluatable string representation."""
        return f"Matrix({self.m.tolist()})"

    def __str__(self) -> str:
        """Returns a human-readable string representation of the matrix."""
        return str(self.m)

    def __copy__(self) -> "Matrix":
        """Creates a shallow copy of the matrix."""
        return Matrix(self)

    def copy(self) -> "Matrix":
        """
        Creates a new Matrix instance with a copy of the internal data.
        This is a convenience method for `copy.copy(self)`.
        """
        return Matrix(self)

    def __deepcopy__(self, memo: dict) -> "Matrix":
        """Creates a deep copy of the matrix."""
        # Since self.m is a numpy array of simple types, a regular
        # copy is sufficient.
        return Matrix(self)

    @staticmethod
    def identity() -> "Matrix":
        """Returns a new identity matrix."""
        return Matrix()

    def is_identity(self) -> bool:
        """
        Checks if the matrix is an identity matrix.

        Uses np.allclose for robust floating-point comparisons.

        Returns:
            True if the matrix is close to an identity matrix, False otherwise.
        """
        return np.allclose(self.m, np.identity(3))

    def get_translation(self) -> Tuple[float, float]:
        """
        Extracts the translation component (tx, ty) from the matrix.
        """
        return (self.m[0, 2], self.m[1, 2])

    @staticmethod
    def translation(tx: float, ty: float) -> "Matrix":
        """Creates a translation matrix."""
        return Matrix(
            [
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1],
            ]
        )

    def set_translation(self, tx: float, ty: float) -> "Matrix":
        """
        Returns a new matrix with the same rotation, scale, and shear,
        but with a new translation component.
        """
        new_matrix = self.copy()
        new_matrix.m[0, 2] = tx
        new_matrix.m[1, 2] = ty
        return new_matrix

    def get_scale(self) -> Tuple[float, float]:
        """
        Extracts the scale components (sx, sy) from the matrix.

        This computes the magnitude of the new basis vectors (the first
        two columns). It doesn't handle negative scaling correctly, as it
        will always return positive magnitudes.
        """
        # sx is the length of the first column vector (the new x-axis)
        sx = np.linalg.norm(self.m[0:2, 0])
        # sy is the length of the second column vector (the new y-axis)
        sy = np.linalg.norm(self.m[0:2, 1])
        return float(sx), float(sy)

    @staticmethod
    def scale(
        sx: float, sy: float, center: Optional[Tuple[float, float]] = None
    ) -> "Matrix":
        """
        Creates a scaling matrix.

        Args:
            sx: Scale factor for the x-axis.
            sy: Scale factor for the y-axis.
            center: Optional (x, y) point to scale around. If None,
                    scales around the origin (0, 0).
        """
        m = Matrix(
            [
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1],
            ]
        )
        if center:
            cx, cy = center
            t_to_origin = Matrix.translation(-cx, -cy)
            t_back = Matrix.translation(cx, cy)
            # Translate to origin, scale, then translate back
            return t_back @ m @ t_to_origin

        return m

    def get_rotation(self) -> float:
        """
        Extracts the rotation angle in degrees from the matrix.

        This computes the angle of the transformed x-axis. It assumes that
        there is no negative scaling (flipping) on the y-axis.
        """
        # The rotation is encoded in the top-left 2x2 submatrix.
        # atan2(m10, m00) gives the angle of the new x-axis vector.
        return math.degrees(math.atan2(self.m[1, 0], self.m[0, 0]))

    @staticmethod
    def rotation(
        angle_deg: float, center: Optional[Tuple[float, float]] = None
    ) -> "Matrix":
        """
        Creates a rotation matrix.

        Args:
            angle_deg: The rotation angle in degrees.
            center: Optional (x, y) point to rotate around. If None,
                    rotates around the origin (0, 0).
        """
        angle_rad = math.radians(angle_deg)
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        m = Matrix(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ]
        )
        if center:
            cx, cy = center
            t_to_origin = Matrix.translation(-cx, -cy)
            t_back = Matrix.translation(cx, cy)
            # Translate to origin, rotate, then translate back
            return t_back @ m @ t_to_origin

        return m

    def invert(self) -> "Matrix":
        """
        Computes the inverse of the matrix.

        The inverse matrix can be used to reverse a transformation.
        Will raise a `numpy.linalg.LinAlgError` if the matrix is singular
        (i.e., not invertible), for example, a scale of zero.

        Returns:
            A new Matrix that is the inverse of this one.
        """
        return Matrix(np.linalg.inv(self.m))

    def transform_point(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Applies the full affine transformation to a 2D point.

        Args:
            point: An (x, y) tuple representing the point to transform.

        Returns:
            A new (x, y) tuple of the transformed point.
        """
        vec = np.array([point[0], point[1], 1])
        res_vec = np.dot(self.m, vec)
        return (res_vec[0], res_vec[1])

    def transform_vector(
        self, vector: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Applies the transformation to a 2D vector, ignoring translation.
        Useful for transforming direction or delta values.
        """
        # Use 0 for the homogeneous coordinate to ignore translation
        vec = np.array([vector[0], vector[1], 0])
        res_vec = np.dot(self.m, vec)
        return (res_vec[0], res_vec[1])

    def decompose(self) -> Tuple[float, float, float, float, float, float]:
        """
        Decomposes the matrix into translation, rotation, scale, and skew.
        This implementation is robust against shear and reflection. It assumes
        a composition order of: Rotate, then Scale, then Shear.

        Returns:
            A tuple (tx, ty, angle_deg, sx, sy, skew_angle_deg).
        """
        # Translation is always the last column
        tx = self.m[0, 2]
        ty = self.m[1, 2]

        # Extract the 2x2 linear transformation part
        a, b = self.m[0, 0], self.m[1, 0]  # First column
        c, d = self.m[0, 1], self.m[1, 1]  # Second column

        # The X scale is the length of the first column vector
        sx = math.hypot(a, b)

        # The rotation is the angle of the first column vector
        angle_rad = math.atan2(b, a)

        # Shear and Y Scale
        # Compute the determinant to detect reflections
        det = a * d - b * c
        if sx != 0:
            sy = det / sx
        else:
            sy = math.hypot(c, d)  # Degenerate case, sx=0

        # Check for reflection (negative determinant indicates a flip)
        if det < 0:
            # If there's a reflection, adjust the rotation angle
            # A reflection flips the coordinate system, adding 180 degrees to
            # the angle
            angle_rad = (
                angle_rad + math.pi if angle_rad <= 0 else angle_rad - math.pi
            )
            sx = -sx  # Correct the x-scale to reflect the negative scaling
            sy = -sy  # Correct the y-scale if necessary

        angle_deg = math.degrees(angle_rad)

        # Solve for the shear factor 'm' in a shear matrix K = [[1, m], [0, 1]]
        # We know L = R * S * K. So R_inv * L = S * K
        # The top-right element of S*K is sx * m
        # The top-right element of R_inv * L is (cos_r * c + sin_r * d)
        cos_r = math.cos(angle_rad)
        sin_r = math.sin(angle_rad)
        if sx != 0:
            shear_factor = (cos_r * c + sin_r * d) / sx
            skew_rad = math.atan(shear_factor)
            skew_angle_deg = math.degrees(skew_rad)
        else:
            skew_angle_deg = 0.0

        return (
            float(tx),
            float(ty),
            float(angle_deg),
            float(sx),
            float(sy),
            float(skew_angle_deg),
        )
