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
    native_to_workspace: np.ndarray
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
            native_to_workspace=identity,
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
        area = machine.workspace_work_area
        width_mm = float(area[2])
        depth_mm = float(area[3])
        space = machine.get_coordinate_space()
        native_to_workspace = space.get_native_to_workspace_matrix().astype(
            np.float32
        )
        origin = space.workspace_origin
        x_right = origin.value.endswith("right")
        y_down = origin.value.startswith("top")

        translate_mat = np.identity(4, dtype=np.float32)
        scale_mat = np.identity(4, dtype=np.float32)
        if y_down:
            translate_mat[1, 3] = depth_mm
            scale_mat[1, 1] = -1.0
        if x_right:
            translate_mat[0, 3] = width_mm
            scale_mat[0, 0] = -1.0
        model_matrix = translate_mat @ scale_mat

        if machine.wcs_origin_is_workarea_origin:
            wcs_offset_mm: Point3D = (0.0, 0.0, 0.0)
        else:
            wcs_x, wcs_y, wcs_z = wcs_offset
            ml, mt, mr, mb = machine.workspace_margins
            machine_x = -mr if x_right else -ml
            machine_y = -mt if y_down else -mb
            workspace_wcs = space.get_axis_label_origin(wcs_offset)
            wcs_x, wcs_y = workspace_wcs[0], workspace_wcs[1]
            local_x = (
                machine_x - wcs_x
                if space.workspace_x_negative
                else machine_x + wcs_x
            )
            local_y = (
                machine_y - wcs_y
                if space.workspace_y_negative
                else machine_y + wcs_y
            )
            wcs_offset_mm = (local_x, local_y, wcs_z)

        margin_shift = np.identity(4, dtype=np.float32)
        ml, _, _, mb = machine.workspace_margins
        margin_shift[0, 3] = -ml
        margin_shift[1, 3] = -mb

        extent_frame: Rect | None = None
        if machine.has_custom_work_area():
            extent_frame = machine.get_visual_extent_frame()

        return cls(
            width_mm=width_mm,
            depth_mm=depth_mm,
            model_matrix=model_matrix,
            native_to_workspace=native_to_workspace,
            wcs_offset_mm=wcs_offset_mm,
            margin_shift=margin_shift,
            extent_frame=extent_frame,
            x_right=x_right,
            y_down=y_down,
            x_negative=space.workspace_x_negative,
            y_negative=space.workspace_y_negative,
        )
