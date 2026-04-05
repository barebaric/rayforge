from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from ...core.geo import Point3D, Rect

if TYPE_CHECKING:
    from ...machine.models.machine import Machine


@dataclass
class ViewportConfig:
    width_mm: float
    depth_mm: float
    model_matrix: np.ndarray
    wcs_offset_mm: Point3D
    margin_shift: np.ndarray
    extent_frame: Optional[Rect]
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
        area = machine.work_area
        width_mm = float(area[2])
        depth_mm = float(area[3])

        translate_mat = np.identity(4, dtype=np.float32)
        scale_mat = np.identity(4, dtype=np.float32)
        if machine.y_axis_down:
            translate_mat[1, 3] = depth_mm
            scale_mat[1, 1] = -1.0
        if machine.x_axis_right:
            translate_mat[0, 3] = width_mm
            scale_mat[0, 0] = -1.0
        model_matrix = translate_mat @ scale_mat

        if machine.wcs_origin_is_workarea_origin:
            wcs_offset_mm: Point3D = (0.0, 0.0, 0.0)
        else:
            wcs_x, wcs_y, wcs_z = machine.get_active_wcs_offset()
            ml, mt, mr, mb = machine.work_margins
            machine_x = -mr if machine.x_axis_right else -ml
            machine_y = -mt if machine.y_axis_down else -mb
            local_x = (
                machine_x - wcs_x
                if machine.reverse_x_axis
                else machine_x + wcs_x
            )
            local_y = (
                machine_y - wcs_y
                if machine.reverse_y_axis
                else machine_y + wcs_y
            )
            wcs_offset_mm = (local_x, local_y, wcs_z)

        margin_shift = np.identity(4, dtype=np.float32)
        ml, _, _, mb = machine.work_margins
        margin_shift[0, 3] = -ml
        margin_shift[1, 3] = -mb

        extent_frame: Optional[Rect] = None
        if machine.has_custom_work_area():
            extent_frame = machine.get_visual_extent_frame()

        return cls(
            width_mm=width_mm,
            depth_mm=depth_mm,
            model_matrix=model_matrix,
            wcs_offset_mm=wcs_offset_mm,
            margin_shift=margin_shift,
            extent_frame=extent_frame,
            x_right=machine.x_axis_right,
            y_down=machine.y_axis_down,
            x_negative=machine.reverse_x_axis,
            y_negative=machine.reverse_y_axis,
        )
