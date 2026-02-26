import pytest
import numpy as np
from unittest.mock import MagicMock
from rayforge.machine.models.machine import Origin
from rayforge.ui_gtk.canvas3d.axis_renderer_3d import AxisRenderer3D

# Test scenarios for axis label expectations, mirrored from the 2D canvas tests
# (
#  origin,
#  negative,
#  machine_dimensions,
#  wcs_offset,
#  expected_axis_labels,
# )
AXIS_EXPECTATIONS = [
    # ---------- Negative Axis Scenarios ----------
    # 1. Bottom-Left, Negative. WCS (-20, -20).
    (
        Origin.BOTTOM_LEFT,
        True,
        (120, 120),
        (-20.0, -20.0, 0.0),
        (-140, -20),
    ),
    # 2. Bottom-Left, Negative. WCS (20, 20).
    (
        Origin.BOTTOM_LEFT,
        True,
        (120, 120),
        (20.0, 20.0, 0.0),
        (-100, 20),
    ),
    # 3. Top-Right, Negative. WCS (-20, -20).
    (
        Origin.TOP_RIGHT,
        True,
        (120, 120),
        (-20.0, -20.0, 0.0),
        (-140, -20),
    ),
    # 4. Top-Right, Negative. WCS (20, 20).
    (
        Origin.TOP_RIGHT,
        True,
        (120, 120),
        (20.0, 20.0, 0.0),
        (-100, 20),
    ),
    # ---------- Positive Axis Scenarios ----------
    # 5. Bottom-Left, Positive. WCS (-20, -20).
    (
        Origin.BOTTOM_LEFT,
        False,
        (120, 120),
        (-20.0, -20.0, 0.0),
        (20, 140),
    ),
    # 6. Bottom-Left, Positive. WCS (20, 20).
    (
        Origin.BOTTOM_LEFT,
        False,
        (120, 120),
        (20.0, 20.0, 0.0),
        (-20, 100),
    ),
    # 7. Top-Right, Positive. WCS (-20, -20).
    (
        Origin.TOP_RIGHT,
        False,
        (120, 120),
        (-20.0, -20.0, 0.0),
        (20, 140),
    ),
    # 8. Top-Right, Positive. WCS (20, 20).
    (
        Origin.TOP_RIGHT,
        False,
        (120, 120),
        (20.0, 20.0, 0.0),
        (-20, 100),
    ),
]


@pytest.mark.ui
@pytest.mark.parametrize(
    "origin, negative, machine_dimensions, wcs_offset, expected_range",
    AXIS_EXPECTATIONS,
)
def test_axis_label_rendering(
    origin, negative, machine_dimensions, wcs_offset, expected_range
):
    """
    Validates that the AxisRenderer3D produces the correct axis labels
    under various machine configurations (origin, coordinate direction, WCS).

    This test mocks the TextRenderer3D to capture the labels that the
    AxisRenderer attempts to draw, and then compares them against the
    pre-defined expectations. This approach isolates the label generation logic
    from the actual rendering process.
    """
    # 1. Setup machine configuration from test parameters
    width_mm, height_mm = machine_dimensions
    x_right = origin in (Origin.TOP_RIGHT, Origin.BOTTOM_RIGHT)
    y_down = origin in (Origin.TOP_LEFT, Origin.TOP_RIGHT)
    # The 'negative' parameter is assumed to apply to both axes for these tests
    x_negative = negative
    y_negative = negative

    # 2. Instantiate the renderer and mock its text rendering dependency
    axis_renderer = AxisRenderer3D(width_mm=width_mm, height_mm=height_mm)
    mock_text_renderer = MagicMock()
    axis_renderer.text_renderer = mock_text_renderer

    # 3. Replicate the matrix calculations from Canvas3D to pass the correct
    #    arguments to the method under test.
    translate_mat = np.identity(4, dtype=np.float32)
    scale_mat = np.identity(4, dtype=np.float32)
    if y_down:
        translate_mat[1, 3] = height_mm
        scale_mat[1, 1] = -1.0
    if x_right:
        translate_mat[0, 3] = width_mm
        scale_mat[0, 0] = -1.0
    model_matrix = translate_mat @ scale_mat

    # Dummy matrices are sufficient as we only test label text, not position
    dummy_mvp = np.identity(4, dtype=np.float32)
    dummy_view = np.identity(4, dtype=np.float32)

    # 4. Call the private method containing the label generation logic
    axis_renderer._render_axis_labels(
        text_shader=MagicMock(),
        text_mvp_matrix=dummy_mvp,
        view_matrix=dummy_view,
        model_matrix=model_matrix,
        origin_offset_mm=wcs_offset,
        x_right=x_right,
        y_down=y_down,
        x_negative=x_negative,
        y_negative=y_negative,
    )

    # 5. Collect and verify the results against expectations
    rendered_labels = []
    if mock_text_renderer.render_text.called:
        for call_args in mock_text_renderer.render_text.call_args_list:
            # The label text is the second positional argument (index 1)
            label_text = call_args.args[1]
            if label_text:
                rendered_labels.append(int(label_text))

    # Generate the set of expected labels based on the provided range
    expected_single_axis_labels = []
    min_val, max_val = sorted(expected_range)
    step = int(axis_renderer.grid_size_mm)

    # Generate labels including endpoints
    expected_single_axis_labels.extend(range(min_val, max_val + 1, step))

    # The renderer should produce labels for both the X and Y axes
    full_expected_list = sorted(expected_single_axis_labels * 2)

    assert sorted(rendered_labels) == full_expected_list, (
        f"Failed for origin={origin.name}, negative={negative}, "
        f"wcs_offset={wcs_offset}"
    )
