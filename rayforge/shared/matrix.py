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
        if not isinstance(other, Matrix):
            return NotImplemented
        return Matrix(np.dot(other.m, self.m))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Matrix):
            return False
        return np.allclose(self.m, other.m)

    def __repr__(self) -> str:
        return f"Matrix({self.m.tolist()})"

    def __str__(self) -> str:
        return str(self.m)

    def __copy__(self) -> "Matrix":
        return Matrix(self)

    def __deepcopy__(self, memo: dict) -> "Matrix":
        return Matrix(self)

    @staticmethod
    def identity() -> "Matrix":
        """Returns a new identity matrix."""
        return Matrix()

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
            return t_to_origin @ m @ t_back

        # If no center is provided, return the matrix that scales around the
        # origin.
        return m

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
            return t_to_origin @ m @ t_back

        # If no center is provided, return the matrix that rotates around the
        # origin.
        return m

    def invert(self) -> "Matrix":
        return Matrix(np.linalg.inv(self.m))

    def transform_point(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        vec = np.array([point[0], point[1], 1])
        res_vec = np.dot(self.m, vec)
        return (res_vec[0], res_vec[1])
