import pytest
import cairo
import numpy as np
from unittest.mock import MagicMock
from rayforge.ui_gtk.canvas.axis import AxisRenderer
from rayforge.core.matrix import Matrix

# A known widget size for consistent testing
WIDGET_W, WIDGET_H = 800, 600
# A known world size for consistent testing
WORLD_W, WORLD_H = 400.0, 300.0

# Test scenarios for axis expectations
# (origin, axis_negative, wcs_offset, expected_labels, expected_marker_pos)
AXIS_EXPECTATIONS = [
    # ---------- Negative Axis Scenarios ----------
    # TEST CASE 1: Bottom-Left origin, negative axis, WCS -20,-20
    (
        "bottom_left",
        True,
        (-20.0, -20.0),
        (-100.0, 20.0),
        (20.0, 20.0),
    ),
    # TEST CASE 2: Bottom-Left origin, negative axis, WCS 20,20
    (
        "bottom_left",
        True,
        (20.0, 20.0),
        (-140.0, -20.0),
        (-20.0, -20.0),
    ),
    # TEST CASE 3: Top-Right origin, negative axis, WCS -20,-20
    (
        "top_right",
        True,
        (-20.0, -20.0),
        (-100.0, 20.0),
        (100.0, 100.0),
    ),
    # TEST CASE 4: Top-Right origin, negative axis, WCS 20,20
    (
        "top_right",
        True,
        (20.0, 20.0),
        (-140.0, -20.0),
        (140.0, 140.0),
    ),
    # ---------- Positive Axis Scenarios ----------
    # TEST CASE 5: Bottom-Left origin, positive axis, WCS -20,-20
    (
        "bottom_left",
        False,
        (-20.0, -20.0),
        (20.0, 140.0),
        (-20.0, -20.0),
    ),
    # TEST CASE 6: Bottom-Left origin, positive axis, WCS 20,20
    (
        "bottom_left",
        False,
        (20.0, 20.0),
        (-20.0, 100.0),
        (20.0, 20.0),
    ),
    # TEST CASE 7: Top-Right origin, positive axis, WCS -20,-20
    (
        "top_right",
        False,
        (-20.0, -20.0),
        (20.0, 140.0),
        (140.0, 140.0),
    ),
    # TEST CASE 8: Top-Right origin, positive axis, WCS 20,20
    (
        "top_right",
        False,
        (20.0, 20.0),
        (-20.0, 100.0),
        (100.0, 100.0),
    ),
]


@pytest.fixture
def renderer() -> AxisRenderer:
    """Provides a default AxisRenderer instance for testing."""
    return AxisRenderer(width_mm=WORLD_W, height_mm=WORLD_H, grid_size_mm=10.0)


@pytest.fixture
def mock_context(mocker) -> MagicMock:
    """Provides a mocked Cairo context to spy on drawing calls."""
    mock = mocker.MagicMock(spec=cairo.Context)
    # Mock text_extents to return a predictable size
    mock.text_extents.return_value = MagicMock(width=10, height=12)
    return mock


@pytest.fixture
def mock_context_with_font_size(mocker) -> MagicMock:
    """Provides a mocked Cairo context with font size tracking."""
    mock = mocker.MagicMock(spec=cairo.Context)
    mock.text_extents.return_value = MagicMock(width=10, height=12)
    mock.font_size = None

    def capture_font_size(size):
        mock.font_size = size

    mock.set_font_size.side_effect = capture_font_size
    return mock


@pytest.mark.ui
class TestAxisRendererLayout:
    """Tests for layout calculations and adaptive sizing."""

    def test_initialization(self):
        """Test that the renderer initializes with correct properties."""
        r = AxisRenderer(
            width_mm=500.0, y_axis_down=True, x_axis_negative=True
        )
        assert r.width_mm == 500.0
        assert r.y_axis_down is True
        assert r.x_axis_negative is True
        assert r.show_grid is True
        assert r.label_font_size == 12.0

    def test_initialization_with_custom_font_size(self):
        """Test that the renderer initializes with custom font size."""
        r = AxisRenderer(width_mm=500.0, label_font_size=16.0)
        assert r.label_font_size == 16.0

    def test_set_label_font_size(self):
        """Test setting the label font size."""
        r = AxisRenderer(width_mm=500.0)
        assert r.label_font_size == 12.0
        r.set_label_font_size(18.0)
        assert r.label_font_size == 18.0

    def test_get_content_layout(self, renderer):
        """Test that the content area is calculated correctly."""
        # For a 400x300 world in an 800x600 widget, the aspect ratios match.
        # We expect the content area to fill the available space minus padding.
        x, y, w, h = renderer.get_content_layout(WIDGET_W, WIDGET_H)
        assert w > 0 and h > 0
        # Aspect ratio should be preserved
        assert np.isclose(w / h, WORLD_W / WORLD_H)

    def test_font_size_affects_layout(self, renderer):
        """Test that changing font size affects the layout calculations."""
        x1, y1, w1, h1 = renderer.get_content_layout(WIDGET_W, WIDGET_H)
        renderer.set_label_font_size(20.0)
        x2, y2, w2, h2 = renderer.get_content_layout(WIDGET_W, WIDGET_H)
        # Larger font should result in smaller content area
        assert w2 < w1
        assert h2 < h1

    def test_get_base_pixels_per_mm(self, renderer):
        """Test the calculation of the base pixels-per-millimeter scale."""
        _, _, content_w, _ = renderer.get_content_layout(WIDGET_W, WIDGET_H)
        expected_ppm = content_w / WORLD_W
        assert np.isclose(
            renderer.get_base_pixels_per_mm(WIDGET_W, WIDGET_H), expected_ppm
        )

    @pytest.mark.parametrize(
        "pixels_per_mm, expected_grid_size",
        [
            (0.1, 500.0),  # Zoomed way out, target is 500mm, nearest 5*p10=500
            (1.0, 50.0),  # Zoomed out, target is 50mm, nearest 5*p10=50
            (5.0, 10.0),  # Normal view, target is 10mm, nearest 1*p10=10
            (12.0, 5.0),  # Zoomed in, target is 4.16mm, nearest 5*p10=5
            (30.0, 2.0),  # Zoomed way in, target is 1.66mm, nearest 2*p10=2
            (
                80.0,
                1.0,
            ),  # Zoomed way, way in, target is 0.625mm, nearest 1*p10=1.0
        ],
    )
    def test_get_adaptive_grid_size(
        self, renderer, pixels_per_mm, expected_grid_size
    ):
        """Test that the grid size adapts correctly to the zoom level."""
        renderer.min_grid_spacing_px = 50.0
        adaptive_size = renderer._get_adaptive_grid_size(pixels_per_mm)
        assert adaptive_size == expected_grid_size


@pytest.mark.ui
class TestAxisRendererDrawing:
    """
    Tests the actual drawing logic by inspecting calls to a mocked context.
    """

    def test_draw_grid_no_offset(self, renderer, mock_context):
        """
        Verify grid lines are drawn at correct multiples without an offset.
        """
        # Use a simple 1:1 transform for easy coordinate verification
        transform = Matrix.identity()
        renderer._draw_grid(
            mock_context, transform, 10.0, 0, 100, 0, 100, (0, 0, 0)
        )

        # Check that a vertical line was drawn at x=10mm
        # view_transform.transform_point((10, 0)) -> (10, 0)
        # view_transform.transform_point((10, 100)) -> (10, 100)
        mock_context.move_to.assert_any_call(10.0, 0.0)
        mock_context.line_to.assert_any_call(10.0, 100.0)

        # Check that a horizontal line was drawn at y=20mm
        mock_context.move_to.assert_any_call(0.0, 20.0)
        mock_context.line_to.assert_any_call(100.0, 20.0)

    def test_draw_grid_with_offset(self, renderer, mock_context):
        """
        Verify grid lines are shifted correctly when a WCS offset is applied.
        """
        transform = Matrix.identity()
        offset = (50.0, 25.0, 0.0)
        renderer._draw_grid(
            mock_context, transform, 10.0, 0, 100, 0, 100, offset
        )

        # The 'zero' line should now be at the offset position
        mock_context.move_to.assert_any_call(50.0, 0.0)  # Vertical at x=50
        mock_context.line_to.assert_any_call(50.0, 100.0)

        mock_context.move_to.assert_any_call(0.0, 25.0)  # Horizontal at y=25
        mock_context.line_to.assert_any_call(100.0, 25.0)

        # The next line should be at offset + grid_size
        mock_context.move_to.assert_any_call(60.0, 0.0)  # Vertical at x=60
        mock_context.line_to.assert_any_call(60.0, 100.0)

    def test_draw_labels_no_offset(
        self, renderer, mock_context_with_font_size
    ):
        """Verify labels are drawn correctly with no offset."""
        transform = Matrix.identity()
        renderer._draw_axis_and_labels(
            mock_context_with_font_size, transform, 10.0, (0, 0, 0)
        )
        # Verify font size is set
        assert mock_context_with_font_size.font_size == 12.0

        # Check if the text "20" was drawn for the X axis.
        # The preceding move_to call should be at world_x = 20
        calls = mock_context_with_font_size.method_calls
        found_label = False
        for i, current_call in enumerate(calls):
            if (
                current_call[0] == "show_text"
                and current_call[1][0] == "20"
                and i > 0
            ):
                prev_call = calls[i - 1]
                if prev_call[0] == "move_to":
                    # prev_call is ('move_to', (x, y))
                    if np.isclose(
                        prev_call[1][0], 20.0 - 5.0
                    ):  # 20.0 - text_width/2
                        found_label = True
                        break
        assert found_label, 'Label "20" not found at the correct X position'

    def test_draw_labels_with_offset(
        self, renderer, mock_context_with_font_size
    ):
        """Verify labels are drawn correctly with a WCS offset."""
        transform = Matrix.identity()
        offset = (50.0, 25.0, 0.0)
        renderer._draw_axis_and_labels(
            mock_context_with_font_size, transform, 10.0, offset
        )
        # Verify font size is set
        assert mock_context_with_font_size.font_size == 12.0

        calls = mock_context_with_font_size.method_calls
        # The label "10" should be physically located at world_x = 50 + 10 = 60
        found_label_10 = False
        for i, c in enumerate(calls):
            if c[0] == "show_text" and c[1][0] == "10" and i > 0:
                prev_call = calls[i - 1]
                if prev_call[0] == "move_to" and np.isclose(
                    prev_call[1][0], 60.0 - 5.0
                ):
                    found_label_10 = True
                    break
        assert found_label_10, 'Label "10" not found at physical position x=60'

        # The label "-50" should be physically located at world_x = 50 - 50 = 0
        found_label_neg_50 = False
        for i, c in enumerate(calls):
            if c[0] == "show_text" and c[1][0] == "-50" and i > 0:
                prev_call = calls[i - 1]
                if prev_call[0] == "move_to" and np.isclose(
                    prev_call[1][0], 0.0 - 5.0
                ):
                    found_label_neg_50 = True
                    break
        assert found_label_neg_50, (
            'Label "-50" not found at physical position x=0'
        )

    def test_draw_labels_no_duplicate_zero(
        self, renderer, mock_context_with_font_size
    ):
        """Verify the '0' label is only drawn once at the origin."""
        transform = Matrix.identity()
        renderer._draw_axis_and_labels(
            mock_context_with_font_size, transform, 10.0, (0, 0, 0)
        )

        # Count how many times show_text was called with "0" or "0.0"
        zero_label_count = 0
        for c in mock_context_with_font_size.method_calls:
            # Correctly check the mock call tuple
            if c[0] == "show_text" and c[1][0] in (
                "0",
                "0.0",
                "0.00",
            ):  # Allow for precision
                zero_label_count += 1

        assert zero_label_count == 1, (
            f"Expected '0' label to be drawn once, but it was drawn "
            f"{zero_label_count} times."
        )

    def test_draw_labels_with_custom_font_size(
        self, renderer, mock_context_with_font_size
    ):
        """Verify custom font size is applied when drawing labels."""
        renderer.set_label_font_size(16.0)
        transform = Matrix.identity()
        renderer._draw_axis_and_labels(
            mock_context_with_font_size, transform, 10.0, (0, 0, 0)
        )
        # Verify custom font size is set
        assert mock_context_with_font_size.font_size == 16.0


@pytest.mark.ui
class TestAxisExpectations:
    """
    Tests for axis rendering expectations under various configurations.

    These tests verify that axis labels and WCS marker positions are
    correctly calculated for different origin positions, machine
    coordinate ranges, and WCS offsets.
    """

    @pytest.mark.parametrize(
        "origin,axis_negative,wcs_offset,expected_labels,expected_marker_pos",
        AXIS_EXPECTATIONS,
    )
    def test_axis_expectations(
        self,
        origin,
        axis_negative,
        wcs_offset,
        expected_labels,
        expected_marker_pos,
    ):
        """
        Verifies axis rendering expectations for various configurations.

        Args:
            origin: Origin position ("bottom_left" or "top_right")
            axis_negative: Whether axis is negative (True) or positive (False)
            wcs_offset: WCS offset as (x, y) tuple
            expected_labels: Expected (min_label, max_label) tuple
            expected_marker_pos: Expected (marker_x, marker_y) tuple from
                canvas bottom-left origin
        """
        # Create renderer with appropriate configuration
        r = AxisRenderer(
            width_mm=120.0,
            height_mm=120.0,
            grid_size_mm=10.0,
            x_axis_right=(origin == "top_right"),
            y_axis_down=(origin == "top_right"),
            x_axis_negative=axis_negative,
            y_axis_negative=axis_negative,
        )

        # Verify renderer configuration
        if origin == "bottom_left":
            assert r.x_axis_right is False
            assert r.y_axis_down is False
        else:
            assert r.x_axis_right is True
            assert r.y_axis_down is True

        assert r.x_axis_negative == axis_negative
        assert r.y_axis_negative == axis_negative

        # Calculate WCS world position
        wcs_x, wcs_y = wcs_offset

        # Logic for determining the WCS marker visual position must match
        # AxisRenderer logic: if axis is negative, the offset value in machine
        # coords corresponds to a visual position inverted from origin.
        eff_wcs_x = -wcs_x if r.x_axis_negative else wcs_x
        eff_wcs_y = -wcs_y if r.y_axis_negative else wcs_y

        if r.x_axis_right:
            wcs_world_x = r.width_mm - eff_wcs_x
        else:
            wcs_world_x = eff_wcs_x

        if r.y_axis_down:
            wcs_world_y = r.height_mm - eff_wcs_y
        else:
            wcs_world_y = eff_wcs_y

        # The test's `expected_marker_pos` defines the final visual position
        # in canvas world coordinates (0,0 is bottom-left).
        marker_x = wcs_world_x
        marker_y = wcs_world_y

        # Verify marker position by component
        assert np.isclose(marker_x, expected_marker_pos[0])
        assert np.isclose(marker_y, expected_marker_pos[1])

        # Calculate expected label range
        # The labels represent the delta from the WCS origin in machine
        # coordinates.
        delta_to_left_edge = 0 - wcs_world_x
        delta_to_right_edge = r.width_mm - wcs_world_x

        if r.x_axis_right:
            # Machine X increases left (negative canvas delta)
            label_at_left_edge = -delta_to_left_edge
            label_at_right_edge = -delta_to_right_edge
        else:
            # Machine X increases right (positive canvas delta)
            label_at_left_edge = delta_to_left_edge
            label_at_right_edge = delta_to_right_edge

        if r.x_axis_negative:
            label_at_left_edge = -label_at_left_edge
            label_at_right_edge = -label_at_right_edge

        # The min and max labels are the min/max of the labels at the bed
        # edges.
        min_label = min(label_at_left_edge, label_at_right_edge)
        max_label = max(label_at_left_edge, label_at_right_edge)

        # Verify label range by component
        assert np.isclose(min_label, expected_labels[0])
        assert np.isclose(max_label, expected_labels[1])
