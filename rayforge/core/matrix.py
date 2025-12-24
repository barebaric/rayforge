import math
from typing import Tuple, Any, Optional, Union, Sequence, List, cast
import numpy as np

# A type alias for data that can be converted into a 3x3 matrix.
# This includes another Matrix, a numpy array, or a 3x3 nested sequence.
MatrixLike = Union[
    "Matrix",
    np.ndarray,
    Sequence[Sequence[float]],
]


class Matrix:
    """
    A 3x3 affine transformation matrix for 2D graphics.

    Provides an object-oriented interface for matrix operations, including
    translations, rotations, and scaling. Uses numpy for the underlying
    calculations.
    """

    def __init__(self, data: Optional[MatrixLike] = None):
        """
        Initializes a 3x3 matrix.

        Args:
            data: Initialization data. Can be:
                  - Another `Matrix` instance (a copy is made).
                  - A 3x3 or 4x4 numpy array.
                  - A 3x3 or 4x4 nested sequence (e.g., list of lists).
                  - `None` (default) to create an identity matrix.
        """
        if data is None:
            self.m: np.ndarray = np.identity(3, dtype=float)
        elif isinstance(data, Matrix):
            self.m = data.m.copy()
        else:
            try:
                array_data = np.array(data, dtype=float)
                if array_data.shape == (3, 3):
                    self.m = array_data
                elif array_data.shape == (4, 4):
                    # Allow initialization from a 4x4 matrix by extracting
                    # the 2D affine components.
                    self.m = np.identity(3, dtype=float)
                    # Copy the 2x2 rotation/scale/shear part
                    self.m[0:2, 0:2] = array_data[0:2, 0:2]
                    # Copy the 2D translation part
                    self.m[0:2, 2] = array_data[0:2, 3]
                else:
                    raise ValueError(
                        "Input data must resolve to a 3x3 or 4x4 shape."
                    )
            except (ValueError, TypeError) as e:
                raise ValueError(f"Could not create Matrix from data: {e}")

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """
        Performs matrix multiplication: self @ other.

        This implements standard pre-multiplication, where (A @ B) @ p is
        equivalent to applying transform B first, then transform A.
        M_combined = M_second @ M_first.

        Args:
            other: The matrix to multiply with on the right.

        Returns:
            The resulting new Matrix.
        """
        if not isinstance(other, Matrix):
            return NotImplemented
        # Standard pre-multiplication: self is the second transform,
        # other is the first.
        return Matrix(np.dot(self.m, other.m))

    def __eq__(self, other: Any) -> bool:
        """
        Checks for equality between two matrices.

        Uses np.allclose for floating-point comparisons.
        """
        if not isinstance(other, Matrix):
            return False
        return np.allclose(self.m, other.m)

    def is_close(self, other: "Matrix", tol: float = 1e-6) -> bool:
        """
        Checks if two matrices are effectively equal within a specific
        tolerance.

        Args:
            other: The matrix to compare against.
            tol: The absolute tolerance parameter.
        """
        if not isinstance(other, Matrix):
            raise TypeError("Can only compare with another Matrix.")
        return np.allclose(self.m, other.m, atol=tol)

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

    def to_numpy(self) -> np.ndarray:
        """Returns a copy of the underlying 3x3 NumPy array."""
        return self.m.copy()

    def to_4x4_numpy(self) -> np.ndarray:
        """
        Converts the 3x3 affine matrix to a 4x4 numpy array suitable for
        3D transformations where Z is preserved.
        """
        m44 = np.identity(4, dtype=float)
        m44[0:2, 0:2] = self.m[0:2, 0:2]  # Copy rotation/scale/shear part
        m44[0:2, 3] = self.m[0:2, 2]  # Copy translation part
        return m44

    def to_list(self) -> List[List[float]]:
        """
        Converts the matrix to a nested list, suitable for serialization.
        """
        # The numpy stubs can sometimes incorrectly infer the return type of
        # tolist(). We use cast to assure the type checker of the correct type.
        return cast(List[List[float]], self.m.tolist())

    @classmethod
    def from_list(cls, data: List[List[float]]) -> "Matrix":
        """
        Creates a Matrix instance from a nested list.
        """
        return cls(data)

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

    def get_determinant_2x2(self) -> float:
        """
        Calculates the determinant of the top-left 2x2 sub-matrix.

        This represents the determinant of the linear transformation part
        (scaling, rotation, shear), ignoring translation. A negative
        determinant indicates a reflection (a "flip").

        Returns:
            The determinant value.
        """
        # ad - bc for M = [[a, c], [b, d]]
        return self.m[0, 0] * self.m[1, 1] - self.m[0, 1] * self.m[1, 0]

    def is_flipped(self) -> bool:
        """
        Checks if the matrix includes a reflection (flip).

        This is determined by checking if the determinant of the linear
        transformation part is negative.

        Returns:
            True if the coordinate system is flipped, False otherwise.
        """
        return self.get_determinant_2x2() < 0

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

    def pre_translate(self, tx: float, ty: float) -> "Matrix":
        """
        Applies a translation before this matrix's transformation.
        Equivalent to `Matrix.translation(tx, ty) @ self`.
        The order of operations is: apply original transform, then translation.
        """
        t = Matrix.translation(tx, ty)
        return t @ self

    def post_translate(self, tx: float, ty: float) -> "Matrix":
        """
        Applies a translation after this matrix's transformation.
        Equivalent to `self @ Matrix.translation(tx, ty)`.
        The order of operations is: apply translation, then original transform.
        """
        t = Matrix.translation(tx, ty)
        return self @ t

    def set_translation(self, tx: float, ty: float) -> "Matrix":
        """
        Returns a new matrix with the same rotation, scale, and shear,
        but with a new translation component.
        """
        new_matrix = self.copy()
        new_matrix.m[0, 2] = tx
        new_matrix.m[1, 2] = ty
        return new_matrix

    def without_translation(self) -> "Matrix":
        """
        Returns a new matrix with the same linear transformation components
        (rotation, scale, shear) but with the translation set to zero.
        """
        new_matrix = self.copy()
        new_matrix.m[0, 2] = 0.0
        new_matrix.m[1, 2] = 0.0
        return new_matrix

    def get_scale(self) -> Tuple[float, float]:
        """
        Extracts the signed scale components (sx, sy) from the matrix.

        This method is robust against rotation and shear. A negative value
        for sx or sy indicates a reflection (flip) along that axis.
        Note: Per the decomposition algorithm, `sx` will always be positive,
        and `sy` will be negative in the case of a reflection.

        Returns:
            A tuple (sx, sy) representing the scale factors.
        """
        # This now uses decompose for a more robust result that includes sign.
        _, _, _, sx, sy, _ = self.decompose()
        return (sx, sy)

    def get_abs_scale(self) -> Tuple[float, float]:
        """
        Extracts the absolute scale components (sx, sy) from the matrix.

        This is useful for calculations where the direction of scaling
        (reflection) does not matter, such as checking for zero scale.

        Returns:
            A tuple (sx, sy) of the absolute scale factors.
        """
        sx, sy = self.get_scale()
        return (abs(sx), abs(sy))

    def has_zero_scale(self, tolerance: float = 1e-6) -> bool:
        """
        Checks if the transformation collapses space onto a line or point.

        This is true if either of the absolute scale factors is smaller than
        the given tolerance. Such a matrix is singular and cannot be
        inverted reliably.

        Args:
            tolerance: The threshold below which a scale is considered zero.

        Returns:
            True if the matrix has effectively zero scale on any axis.
        """
        sx_abs, sy_abs = self.get_abs_scale()
        return sx_abs < tolerance or sy_abs < tolerance

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

    def pre_scale(
        self,
        sx: float,
        sy: float,
        center: Optional[Tuple[float, float]] = None,
    ) -> "Matrix":
        """
        Applies a scale before this matrix's transformation.
        Equivalent to `Matrix.scale(sx, sy, center) @ self`.
        """
        s = Matrix.scale(sx, sy, center)
        return s @ self

    def post_scale(
        self,
        sx: float,
        sy: float,
        center: Optional[Tuple[float, float]] = None,
    ) -> "Matrix":
        """
        Applies a scale after this matrix's transformation.
        Equivalent to `self @ Matrix.scale(sx, sy, center)`.
        """
        s = Matrix.scale(sx, sy, center)
        return self @ s

    def get_rotation(self) -> float:
        """
        Extracts the rotation angle in degrees from the matrix.

        This method uses a full decomposition, making it robust against
        shear and non-uniform scaling.

        Returns:
            The rotation angle in degrees.
        """
        # This now uses decompose for a more robust result.
        _, _, angle_deg, _, _, _ = self.decompose()
        return angle_deg

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

    def pre_rotate(
        self, angle_deg: float, center: Optional[Tuple[float, float]] = None
    ) -> "Matrix":
        """
        Applies a rotation before this matrix's transformation.
        Equivalent to `Matrix.rotation(angle_deg, center) @ self`.
        """
        r = Matrix.rotation(angle_deg, center)
        return r @ self

    def post_rotate(
        self, angle_deg: float, center: Optional[Tuple[float, float]] = None
    ) -> "Matrix":
        """
        Applies a rotation after this matrix's transformation.
        Equivalent to `self @ Matrix.rotation(angle_deg, center)`.
        """
        r = Matrix.rotation(angle_deg, center)
        return self @ r

    @staticmethod
    def shear(
        sh_x: float, sh_y: float, center: Optional[Tuple[float, float]] = None
    ) -> "Matrix":
        """
        Creates a shearing matrix.

        Args:
            sh_x: Shear factor for the x-axis (x' = x + sh_x * y).
            sh_y: Shear factor for the y-axis (y' = y + sh_y * x).
            center: Optional (x, y) point to shear around. If None,
                    shears around the origin (0, 0).
        """
        m = Matrix(
            [
                [1, sh_x, 0],
                [sh_y, 1, 0],
                [0, 0, 1],
            ]
        )
        if center:
            cx, cy = center
            t_to_origin = Matrix.translation(-cx, -cy)
            t_back = Matrix.translation(cx, cy)
            # Translate to origin, shear, then translate back
            return t_back @ m @ t_to_origin
        return m

    def pre_shear(
        self,
        sh_x: float,
        sh_y: float,
        center: Optional[Tuple[float, float]] = None,
    ) -> "Matrix":
        """
        Applies a shear before this matrix's transformation.
        Equivalent to `Matrix.shear(sh_x, sh_y, center) @ self`.
        """
        s = Matrix.shear(sh_x, sh_y, center)
        return s @ self

    def post_shear(
        self,
        sh_x: float,
        sh_y: float,
        center: Optional[Tuple[float, float]] = None,
    ) -> "Matrix":
        """
        Applies a shear after this matrix's transformation.
        Equivalent to `self @ Matrix.shear(sh_x, sh_y, center)`.
        """
        s = Matrix.shear(sh_x, sh_y, center)
        return self @ s

    @staticmethod
    def flip_horizontal(
        center: Optional[Tuple[float, float]] = None,
    ) -> "Matrix":
        """
        Creates a horizontal flip (mirror along the Y-axis) matrix.

        Args:
            center: Optional (x, y) point to flip around. If None,
                    flips around the origin (0, 0).
        """
        return Matrix.scale(-1.0, 1.0, center=center)

    @staticmethod
    def flip_vertical(
        center: Optional[Tuple[float, float]] = None,
    ) -> "Matrix":
        """
        Creates a vertical flip (mirror along the X-axis) matrix.

        Args:
            center: Optional (x, y) point to flip around. If None,
                    flips around the origin (0, 0).
        """
        return Matrix.scale(1.0, -1.0, center=center)

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

    def transform_rectangle(
        self, rect: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Transforms a rectangle and computes its new axis-aligned bounding box.

        Args:
            rect: A tuple (x, y, width, height) of the rectangle to transform.

        Returns:
            A tuple (x, y, width, height) of the resulting bounding box.
        """
        x, y, w, h = rect
        corners = [
            self.transform_point((x, y)),
            self.transform_point((x + w, y)),
            self.transform_point((x + w, y + h)),
            self.transform_point((x, y + h)),
        ]
        min_x = min(p[0] for p in corners)
        min_y = min(p[1] for p in corners)
        max_x = max(p[0] for p in corners)
        max_y = max(p[1] for p in corners)
        return min_x, min_y, max_x - min_x, max_y - min_y

    def get_x_axis_angle(self) -> float:
        """
        Calculates the angle of the transformed X-axis in degrees.
        This represents the visual angle of horizontal lines after
        transformation.
        """
        # The transformed x-axis is represented by the first column of the
        # matrix
        xx, yx = self.m[0, 0], self.m[1, 0]
        return math.degrees(math.atan2(yx, xx))

    def get_y_axis_angle(self) -> float:
        """
        Calculates the angle of the transformed Y-axis in degrees.
        This represents the visual angle of vertical lines after
        transformation.
        """
        # The transformed y-axis is represented by the second column of the
        # matrix
        xy, yy = self.m[0, 1], self.m[1, 1]
        return math.degrees(math.atan2(yy, xy))

    @staticmethod
    def compose(
        tx: float,
        ty: float,
        angle_deg: float,
        sx: float,
        sy: float,
        skew_angle_deg: float,
    ) -> "Matrix":
        """
        Composes a matrix from translation, rotation, scale, and skew.
        This is the inverse of the `decompose` method, assuming a
        composition order of: Rotate, then Scale, then Shear.
        """
        R = Matrix.rotation(angle_deg)
        S = Matrix.scale(sx, sy)

        skew_rad = math.radians(skew_angle_deg)
        shear_factor = math.tan(skew_rad)
        K = Matrix.shear(shear_factor, 0)

        # Combine linear transformations
        linear_matrix = R @ S @ K

        # Set the translation component
        composed_matrix = linear_matrix.set_translation(tx, ty)

        return composed_matrix

    def decompose(self) -> Tuple[float, float, float, float, float, float]:
        """
        Decomposes the matrix into translation, rotation, scale, and skew.
        This implementation is robust against shear and reflection. It assumes
        a composition order of: Rotate, then Scale, then Shear.

        The decomposition is stable: sx will always be positive, and any
        reflection is represented by a negative sy.

        Returns:
            A tuple (tx, ty, angle_deg, sx, sy, skew_angle_deg).
        """
        # Translation is always the last column
        tx = self.m[0, 2]
        ty = self.m[1, 2]

        # Extract the 2x2 linear transformation part
        a, b = self.m[0, 0], self.m[1, 0]  # First column
        c, d = self.m[0, 1], self.m[1, 1]  # Second column

        # The X scale is the length of the first column vector (always > 0)
        sx = math.hypot(a, b)

        # The rotation is the angle of the first column vector
        angle_rad = math.atan2(b, a)
        angle_deg = math.degrees(angle_rad)

        # Shear and Y Scale
        # We find sy by `det(R*S*K) = det(R)*det(S)*det(K) = 1 * (sx*sy) * 1`
        # So, sy = det(M) / sx. This carries the sign of the reflection.
        det = a * d - b * c
        if sx != 0:
            sy = det / sx
        else:
            # Degenerate case, sx=0. sy is the length of the second column.
            sy = math.hypot(c, d)

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

    def for_cairo(self) -> Tuple[float, float, float, float, float, float]:
        """
        Returns the matrix components in the order expected by cairo.Matrix.
        The order is (xx, yx, xy, yy, x0, y0).
        """
        # Our matrix: [[xx, xy, x0], [yx, yy, y0], [0, 0, 1]]
        xx = self.m[0, 0]
        xy = self.m[0, 1]
        x0 = self.m[0, 2]
        yx = self.m[1, 0]
        yy = self.m[1, 1]
        y0 = self.m[1, 2]
        return (xx, yx, xy, yy, x0, y0)
