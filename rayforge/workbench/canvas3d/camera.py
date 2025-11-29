"""
Defines the Camera class for managing 3D perspective and navigation.
"""

import math
import numpy as np


def rotation_matrix_from_axis_angle(
    axis: np.ndarray, angle: float
) -> np.ndarray:
    """Creates a rotation matrix from an axis and an angle (Rodrigues)."""
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return np.identity(3, dtype=np.float64)
    axis = axis / norm

    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


class Camera:
    """
    Manages the camera's position, orientation, and projection.
    """

    def __init__(
        self,
        position: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
        width: int,
        height: int,
    ):
        """
        Initializes the Camera.

        Args:
            position: The 3D position of the camera.
            target: The 3D point the camera is looking at.
            up: The "up" direction vector for the camera.
            width: The width of the viewport in pixels.
            height: The height of the viewport in pixels.
        """
        self.position = np.array(position, dtype=np.float64)
        self.target = np.array(target, dtype=np.float64)
        self.up = np.array(up, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)
        self.is_perspective = True

    def get_view_matrix(self) -> np.ndarray:
        """
        Calculates the view matrix (look-at matrix).

        Returns:
            A 4x4 numpy array representing the view transformation.
        """
        forward = self.target - self.position
        forward /= np.linalg.norm(forward)

        side = np.cross(forward, self.up)
        side /= np.linalg.norm(side)

        up_vec = np.cross(side, forward)

        view_matrix = np.identity(4, dtype=np.float32)
        view_matrix[0, 0], view_matrix[1, 0], view_matrix[2, 0] = side
        view_matrix[0, 1], view_matrix[1, 1], view_matrix[2, 1] = up_vec
        view_matrix[0, 2], view_matrix[1, 2], view_matrix[2, 2] = -forward
        view_matrix[3, 0] = -np.dot(side, self.position)
        view_matrix[3, 1] = -np.dot(up_vec, self.position)
        view_matrix[3, 2] = np.dot(forward, self.position)
        return view_matrix.T

    def get_projection_matrix(self) -> np.ndarray:
        """
        Calculates the projection matrix (perspective or orthographic).

        Returns:
            A 4x4 numpy array for the projection transformation.
        """
        aspect_ratio = self.width / self.height if self.height > 0 else 1.0
        near_clip, far_clip = 0.1, 10000.0

        if not self.is_perspective:
            return self._get_ortho_matrix(aspect_ratio, near_clip, far_clip)
        return self._get_perspective_matrix(aspect_ratio, near_clip, far_clip)

    def _get_perspective_matrix(
        self, aspect_ratio: float, near: float, far: float
    ) -> np.ndarray:
        """Builds a perspective projection matrix."""
        fovy_rad = math.radians(45.0)
        f = 1.0 / math.tan(fovy_rad / 2.0)
        return np.array(
            [
                [f / aspect_ratio, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    (far + near) / (near - far),
                    (2 * far * near) / (near - far),
                ],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )

    def _get_ortho_matrix(
        self, aspect_ratio: float, near: float, far: float
    ) -> np.ndarray:
        """Builds an orthographic projection matrix."""
        distance = np.linalg.norm(self.target - self.position)
        fov_y_rad = math.radians(45.0)
        ortho_height = distance * math.tan(fov_y_rad / 2.0) * 2.0
        ortho_width = ortho_height * aspect_ratio
        right, top = ortho_width / 2.0, ortho_height / 2.0

        return np.array(
            [
                [1.0 / right, 0.0, 0.0, 0.0],
                [0.0, 1.0 / top, 0.0, 0.0],
                [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def pan(self, delta_x: float, delta_y: float):
        """
        Moves the camera and its target sideways and up/down.

        Args:
            delta_x: The horizontal change in screen coordinates.
            delta_y: The vertical change in screen coordinates.
        """
        distance = np.linalg.norm(self.target - self.position)
        pan_speed = 0.001 * distance

        forward = self.target - self.position
        forward /= distance + 1e-9

        side = np.cross(forward, self.up)
        side /= np.linalg.norm(side) + 1e-9

        up_vec = np.cross(side, forward)

        pan_vector = (side * delta_x - up_vec * delta_y) * pan_speed
        self.position += pan_vector
        self.target += pan_vector

    def dolly(self, delta_z: float):
        """
        Moves the camera forward or backward along its line of sight.

        Args:
            delta_z: The amount to dolly (typically from a scroll wheel).
        """
        forward = self.target - self.position
        distance = np.linalg.norm(forward)

        # Prevent zooming in too close
        if distance < 0.2 and delta_z < 0:
            return

        zoom_amount = -delta_z * 0.1 * distance
        self.position += (forward / distance) * zoom_amount

    def orbit(self, pivot: np.ndarray, axis: np.ndarray, angle: float):
        """
        Orbits the camera around a pivot point.

        Args:
            pivot: The 3D point to orbit around.
            axis: The axis of rotation.
            angle: The angle of rotation in radians.
        """
        if abs(angle) < 1e-6:
            return

        rot_matrix = rotation_matrix_from_axis_angle(axis, angle)

        self.position = pivot + rot_matrix @ (self.position - pivot)
        self.target = pivot + rot_matrix @ (self.target - pivot)
        self.up = rot_matrix @ self.up

    def set_top_view(self, world_width: float, world_depth: float):
        """Configures the camera for a top-down, orthographic view (Z-up)."""
        center_x, center_y = world_width / 2.0, world_depth / 2.0
        max_dim = max(world_width, world_depth)

        # Look from above (positive Z) down to the XY plane.
        self.position = np.array(
            [center_x, center_y, max_dim * 1.5], dtype=np.float64
        )
        self.target = np.array([center_x, center_y, 0.0], dtype=np.float64)

        # Standard orientation: Up vector points along positive Y.
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        # A top-down view should be orthographic, not perspective.

    def set_front_view(self, world_width: float, world_depth: float):
        """Configures the camera for a 45-degree front view."""
        center_x, center_y = world_width / 2.0, world_depth / 2.0
        max_dim = max(world_width, world_depth)

        # Target the center of the XY "floor" plane.
        self.target = np.array([center_x, center_y, 0.0], dtype=np.float64)

        # Position the camera in front of the bed (negative Y) and above it
        # (positive Z) to get a 45-degree downward angle.
        direction = np.array([0.0, -1.0, 1.0])
        direction = direction / np.linalg.norm(direction)

        distance = max_dim * 1.7
        self.position = self.target + direction * distance

        # World Z-axis is the conventional "up" for this view.
        self.up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def set_iso_view(self, world_width: float, world_depth: float):
        """Configures the camera for a standard isometric view (Z-up)."""
        center_x, center_y = world_width / 2.0, world_depth / 2.0
        max_dim = max(world_width, world_depth)

        # Target the center of the XY "floor" plane.
        self.target = np.array([center_x, center_y, 0.0], dtype=np.float64)

        # Position the camera for a view from the top-front-left.
        direction = np.array([-1.0, -1.0, 1.0])
        direction = direction / np.linalg.norm(direction)

        distance = max_dim * 1.7  # A bit more distance for perspective
        self.position = self.target + direction * distance

        # In a Z-up system, the world Z-axis is the conventional "up"
        # vector for an isometric view, producing the correct orientation.
        self.up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
