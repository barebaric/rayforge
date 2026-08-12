from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from raygeo.geo.types import Point3D, Rect

if TYPE_CHECKING:
    from ...machine.models.machine import Machine


@dataclass
class ViewportConfig:
    width_mm: float
    depth_mm: float
    model_matrix: np.ndarray
    world_to_panel: np.ndarray
    wcs_offset_mm: Point3D
    margin_shift: np.ndarray
    extent_frame: Rect | None
    x_right: bool
    y_down: bool
    x_negative: bool
    y_negative: bool

    @classmethod
    def default(
        cls, width_mm: float = 100.0, depth_mm: float = 100.0
    ) -> "ViewportConfig":
        identity = np.identity(4, dtype=np.float32)
        return cls(
            width_mm=width_mm,
            depth_mm=depth_mm,
            model_matrix=identity,
            world_to_panel=identity,
            wcs_offset_mm=(0.0, 0.0, 0.0),
            margin_shift=identity,
            extent_frame=None,
            x_right=False,
            y_down=False,
            x_negative=False,
            y_negative=False,
        )

    @classmethod
    def from_machine(cls, machine: "Machine") -> "ViewportConfig":
        return cls.from_machine_with_wcs(
            machine, machine.get_active_wcs_offset()
        )

    @classmethod
    def from_machine_with_wcs(
        cls, machine: "Machine", wcs_offset: tuple
    ) -> "ViewportConfig":
        panel = machine.panel
        width_mm = float(panel.workarea_size[0])
        depth_mm = float(panel.workarea_size[1])

        world_to_panel = panel.world_to_panel.astype(np.float32)

        translate_mat = np.identity(4, dtype=np.float32)
        scale_mat = np.identity(4, dtype=np.float32)
        if panel.y_axis_down:
            translate_mat[1, 3] = depth_mm
            scale_mat[1, 1] = -1.0
        if panel.x_axis_right:
            translate_mat[0, 3] = width_mm
            scale_mat[0, 0] = -1.0
        model_matrix = translate_mat @ scale_mat

        if machine.wcs_origin_is_workarea_origin:
            wcs_offset_mm: Point3D = (0.0, 0.0, 0.0)
        else:
            wcs_x, wcs_y, wcs_z = wcs_offset
            ml, mt, mr, mb = panel.margins
            machine_x = -mr if panel.x_axis_right else -ml
            machine_y = -mt if panel.y_axis_down else -mb
            local_x = (
                machine_x - wcs_x
                if panel.x_axis_negative
                else machine_x + wcs_x
            )
            local_y = (
                machine_y - wcs_y
                if panel.y_axis_negative
                else machine_y + wcs_y
            )
            wcs_offset_mm = (local_x, local_y, wcs_z)

        margin_shift = np.identity(4, dtype=np.float32)
        ml, _, _, mb = panel.margins
        margin_shift[0, 3] = -ml
        margin_shift[1, 3] = -mb

        extent_frame: Rect | None = None
        if panel.has_custom_work_area:
            extent_frame = panel.extent_frame

        return cls(
            width_mm=width_mm,
            depth_mm=depth_mm,
            model_matrix=model_matrix,
            world_to_panel=world_to_panel,
            wcs_offset_mm=wcs_offset_mm,
            margin_shift=margin_shift,
            extent_frame=extent_frame,
            x_right=panel.x_axis_right,
            y_down=panel.y_axis_down,
            x_negative=panel.x_axis_negative,
            y_negative=panel.y_axis_negative,
        )
