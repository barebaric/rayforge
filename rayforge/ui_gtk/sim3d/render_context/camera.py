"""Camera/frame-level render context section."""

from typing import TYPE_CHECKING, Optional

import numpy as np

from ....core.color import ColorSet

if TYPE_CHECKING:
    from ....machine.models.machine import Machine
    from ..camera import Camera
    from .base import FrameInputs


class CameraContext:
    """Frame-level camera matrices, colors, line width and display toggles.

    The plain constructor leaves the section empty; call :meth:`update`
    each frame to populate it from the camera, machine and viewport.
    """

    def __init__(
        self,
        *,
        proj_matrix: np.ndarray | None = None,
        view_matrix: np.ndarray | None = None,
        mvp_ui: np.ndarray | None = None,
        viewport_height: int = 0,
        camera_position: np.ndarray | None = None,
        color_set: Optional["ColorSet"] = None,
        line_width: float = 2.0,
        show_travel_moves: bool = False,
        show_grid: bool = True,
        show_nogo_zones: bool = True,
        show_models: bool = True,
        show_ops_underlay: bool = True,
    ):
        identity = np.eye(4, dtype=np.float32)
        self.proj_matrix = identity if proj_matrix is None else proj_matrix
        self.view_matrix = identity if view_matrix is None else view_matrix
        self.mvp_ui = identity if mvp_ui is None else mvp_ui
        self.viewport_height = viewport_height
        self.camera_position = (
            np.zeros(3) if camera_position is None else camera_position
        )
        self.color_set = color_set if color_set is not None else ColorSet()
        self.line_width = line_width
        self.show_travel_moves = show_travel_moves
        self.show_grid = show_grid
        self.show_nogo_zones = show_nogo_zones
        self.show_models = show_models
        self.show_ops_underlay = show_ops_underlay

    def update(self, frame: "FrameInputs") -> None:
        """Recomputes the camera section from the current frame inputs."""
        camera = frame.camera
        proj_matrix = camera.get_projection_matrix()
        view_matrix = camera.get_view_matrix()
        mvp_ui = proj_matrix @ view_matrix
        self.proj_matrix = proj_matrix
        self.view_matrix = view_matrix
        self.mvp_ui = mvp_ui
        self.viewport_height = camera.height
        self.camera_position = camera.position
        self.color_set = frame.color_set
        self.line_width = self._compute_spot_line_width(
            frame.machine, camera, mvp_ui
        )
        self.show_travel_moves = frame.show_travel_moves
        self.show_grid = frame.show_grid
        self.show_nogo_zones = frame.show_nogo_zones
        self.show_models = frame.show_models
        self.show_ops_underlay = frame.show_ops_underlay

    @staticmethod
    def _world_size_to_pixels(
        mvp: np.ndarray,
        world_mm: float,
        viewport_w: int,
        viewport_h: int,
    ) -> float:
        p0 = mvp @ np.array([0, 0, 0, 1], dtype=np.float32)
        p1 = mvp @ np.array([world_mm, 0, 0, 1], dtype=np.float32)
        if abs(p0[3]) < 1e-9 or abs(p1[3]) < 1e-9:
            return 1.0
        ndc_dx = (p1[0] / p1[3]) - (p0[0] / p0[3])
        return abs(ndc_dx) * viewport_w * 0.5

    @classmethod
    def _compute_spot_line_width(
        cls,
        machine: Optional["Machine"],
        camera: "Camera",
        mvp: np.ndarray,
    ) -> float:
        spot_mm = 0.1
        laser_head = machine.get_default_laser_head() if machine else None
        if laser_head is not None:
            spot_mm = laser_head.spot_size_mm[0]
        if not camera:
            return 2.0
        px = cls._world_size_to_pixels(
            mvp,
            spot_mm,
            camera.width,
            camera.height,
        )
        return max(2.0, px)
