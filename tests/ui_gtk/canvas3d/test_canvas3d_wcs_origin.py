import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from rayforge.machine.models.machine import Origin
from rayforge.ui_gtk.canvas3d.axis_renderer_3d import AxisRenderer3D

WCS_SCENARIOS = [
    # (Origin, Dimensions, WCS Offset, NegativeFlags(x,y), Expected World Pos)
    # WCS Offset is (MachineX, MachineY)
    # 1. Bottom-Left (Std). WCS(10, 20). World(10, 20).
    (Origin.BOTTOM_LEFT, (100, 100), (10, 20), (False, False), (10, 20)),
    # 2. Top-Left. WCS(10, 20).
    # Origin Top-Left.
    # X=10 -> World X=10.
    # Y=20 (Down). World Y = 100 - 20 = 80.
    (Origin.TOP_LEFT, (100, 100), (10, 20), (False, False), (10, 80)),
    # 3. Top-Right. WCS(10, 20).
    # Origin Top-Right.
    # X=10 (Left). World X = 100 - 10 = 90.
    # Y=20 (Down). World Y = 100 - 20 = 80.
    (Origin.TOP_RIGHT, (100, 100), (10, 20), (False, False), (90, 80)),
    # 4. Bottom-Right. WCS(10, 20).
    # Origin Bottom-Right.
    # X=10 (Left). World X = 100 - 10 = 90.
    # Y=20 (Up). World Y = 20.
    (Origin.BOTTOM_RIGHT, (100, 100), (10, 20), (False, False), (90, 20)),
    # --- Negative Axis ---
    # 5. Bottom-Left, Neg X. WCS(-10, 20).
    # Origin BL.
    # X is Negative. WCS X = -10. Mag = 10.
    # Pos: 10mm Right of Origin. World X = 10.
    (Origin.BOTTOM_LEFT, (100, 100), (-10, 20), (True, False), (10, 20)),
    # 6. Top-Right, Neg X, Neg Y. WCS(-10, -20).
    # Origin TR.
    # X Neg. WCS -10 -> Mag 10. Pos: 10mm Left of Origin.
    # World X = 100 - 10 = 90.
    # Y Neg. WCS -20 -> Mag 20. Pos: 20mm Down from Origin.
    # World Y = 100 - 20 = 80.
    (Origin.TOP_RIGHT, (100, 100), (-10, -20), (True, True), (90, 80)),
]


@pytest.mark.ui
@pytest.mark.parametrize(
    "origin, dims, wcs, negative_flags, expected_world", WCS_SCENARIOS
)
def test_wcs_marker_position(
    origin, dims, wcs, negative_flags, expected_world
):
    width, height = dims
    wcs_x, wcs_y = wcs
    x_neg, y_neg = negative_flags
    exp_x, exp_y = expected_world

    x_right = origin in (Origin.TOP_RIGHT, Origin.BOTTOM_RIGHT)
    y_down = origin in (Origin.TOP_LEFT, Origin.TOP_RIGHT)

    translate_mat = np.identity(4, dtype=np.float32)
    scale_mat = np.identity(4, dtype=np.float32)

    if y_down:
        translate_mat[1, 3] = height
        scale_mat[1, 1] = -1.0
    if x_right:
        translate_mat[0, 3] = width
        scale_mat[0, 0] = -1.0

    model_matrix = translate_mat @ scale_mat

    # Patch GL in axis_renderer_3d to prevent "invalid operation"
    # (missing context) on Windows.
    with patch("rayforge.ui_gtk.canvas3d.axis_renderer_3d.GL"):
        renderer = AxisRenderer3D(width, height)
        renderer.grid_vao = 1
        renderer.axes_vao = 2
        renderer.wcs_marker_vao = 3

        # Mock background renderer to prevent it calling real GL from its
        # module. This prevents execution of background_renderer.render(),
        # reducing the number of uMVP calls by 1.
        renderer.background_renderer = MagicMock()
        renderer.background_renderer.vao = 99

        renderer.text_renderer = MagicMock()

        mock_line_shader = MagicMock()
        mock_text_shader = MagicMock()

        text_mvp = np.identity(4, dtype=np.float32)
        scene_mvp = np.identity(4, dtype=np.float32)
        view_matrix = np.identity(4, dtype=np.float32)

        renderer.render(
            line_shader=mock_line_shader,
            text_shader=mock_text_shader,
            scene_mvp=scene_mvp,
            text_mvp=text_mvp,
            view_matrix=view_matrix,
            model_matrix=model_matrix,
            origin_offset_mm=(wcs_x, wcs_y, 0.0),
            x_right=x_right,
            y_down=y_down,
            x_negative=x_neg,
            y_negative=y_neg,
        )

    mvp_calls = [
        args[1]
        for args, _ in mock_line_shader.set_mat4.call_args_list
        if args[0] == "uMVP"
    ]

    # We expect at least 2 calls: one for Grid/Axes, one for WCS Marker.
    # (The background plane call is suppressed by the mock).
    assert len(mvp_calls) >= 2
    wcs_mvp = mvp_calls[-1]

    actual_x = wcs_mvp[3, 0]
    actual_y = wcs_mvp[3, 1]

    np.testing.assert_allclose(
        (actual_x, actual_y),
        (exp_x, exp_y),
        atol=0.001,
        err_msg=f"WCS Marker pos mismatch for {origin}",
    )
