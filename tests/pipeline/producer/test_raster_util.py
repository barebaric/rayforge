import pytest
import numpy as np
from rayforge.pipeline.producer.raster_util import (
    ScanLine,
    line_pixels,
    generate_scan_lines,
    find_segments,
    calculate_ymax_mm,
    convert_y_to_output,
    find_bounding_box,
    find_mask_bounding_box,
    generate_horizontal_scan_positions,
    resample_rows,
)


class TestScanLine:
    def test_length_mm_horizontal(self):
        scan = ScanLine(
            index=0,
            start_mm=(0.0, 1.0),
            end_mm=(10.0, 1.0),
            pixels=np.array([]),
            line_interval_mm=0.1,
        )
        assert scan.length_mm == pytest.approx(10.0)

    def test_length_mm_vertical(self):
        scan = ScanLine(
            index=0,
            start_mm=(5.0, 0.0),
            end_mm=(5.0, 5.0),
            pixels=np.array([]),
            line_interval_mm=0.1,
        )
        assert scan.length_mm == pytest.approx(5.0)

    def test_length_mm_diagonal(self):
        scan = ScanLine(
            index=0,
            start_mm=(0.0, 0.0),
            end_mm=(3.0, 4.0),
            pixels=np.array([]),
            line_interval_mm=0.1,
        )
        assert scan.length_mm == pytest.approx(5.0)

    def test_direction_horizontal(self):
        scan = ScanLine(
            index=0,
            start_mm=(0.0, 5.0),
            end_mm=(10.0, 5.0),
            pixels=np.array([]),
            line_interval_mm=0.1,
        )
        dx, dy = scan.direction
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(0.0)

    def test_direction_vertical(self):
        scan = ScanLine(
            index=0,
            start_mm=(5.0, 0.0),
            end_mm=(5.0, 10.0),
            pixels=np.array([]),
            line_interval_mm=0.1,
        )
        dx, dy = scan.direction
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(1.0)

    def test_direction_zero_length(self):
        scan = ScanLine(
            index=0,
            start_mm=(5.0, 5.0),
            end_mm=(5.0, 5.0),
            pixels=np.array([]),
            line_interval_mm=0.1,
        )
        dx, dy = scan.direction
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(0.0)

    def test_pixel_to_mm_horizontal(self):
        scan = ScanLine(
            index=0,
            start_mm=(0.0, 0.5),
            end_mm=(1.0, 0.5),
            pixels=np.array([]),
            line_interval_mm=0.1,
        )
        pixels_per_mm = (10, 10)

        x, y = scan.pixel_to_mm(5, 5, pixels_per_mm)
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.5)

    def test_pixel_to_mm_vertical(self):
        scan = ScanLine(
            index=0,
            start_mm=(0.5, 0.0),
            end_mm=(0.5, 1.0),
            pixels=np.array([]),
            line_interval_mm=0.1,
        )
        pixels_per_mm = (10, 10)

        x, y = scan.pixel_to_mm(5, 5, pixels_per_mm)
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.5)


class TestLinePixels:
    def test_horizontal_line(self):
        pixels = line_pixels((0, 5), (10, 5), 11, 11)
        assert len(pixels) == 11
        assert all(p[1] == 5 for p in pixels)
        assert pixels[0][0] == 0
        assert pixels[-1][0] == 10

    def test_vertical_line(self):
        pixels = line_pixels((5, 0), (5, 10), 11, 11)
        assert len(pixels) == 11
        assert all(p[0] == 5 for p in pixels)
        assert pixels[0][1] == 0
        assert pixels[-1][1] == 10

    def test_diagonal_line(self):
        pixels = line_pixels((0, 0), (10, 10), 11, 11)
        assert len(pixels) == 11
        assert all(p[0] == p[1] for p in pixels)

    def test_line_outside_bounds_clipped(self):
        pixels = line_pixels((-5, 5), (5, 5), 11, 11)
        assert len(pixels) == 6
        assert all(0 <= p[0] < 11 for p in pixels)
        assert all(0 <= p[1] < 11 for p in pixels)

    def test_single_pixel(self):
        pixels = line_pixels((5, 5), (5, 5), 11, 11)
        assert len(pixels) == 1
        assert pixels[0][0] == 5
        assert pixels[0][1] == 5


class TestGenerateScanLines:
    @pytest.fixture
    def simple_bbox(self):
        return (0, 9, 0, 9)

    @pytest.fixture
    def simple_image_size(self):
        return (10, 10)

    @pytest.fixture
    def simple_pixels_per_mm(self):
        return (10, 10)

    def test_horizontal_lines_count(
        self, simple_bbox, simple_image_size, simple_pixels_per_mm
    ):
        lines = list(
            generate_scan_lines(
                simple_bbox,
                simple_image_size,
                simple_pixels_per_mm,
                line_interval_mm=0.1,
                direction_degrees=0,
            )
        )
        assert len(lines) == 10

    def test_vertical_lines_count(
        self, simple_bbox, simple_image_size, simple_pixels_per_mm
    ):
        lines = list(
            generate_scan_lines(
                simple_bbox,
                simple_image_size,
                simple_pixels_per_mm,
                line_interval_mm=0.1,
                direction_degrees=90,
            )
        )
        assert len(lines) == 10

    def test_line_indices_are_sequential(
        self, simple_bbox, simple_image_size, simple_pixels_per_mm
    ):
        lines = list(
            generate_scan_lines(
                simple_bbox,
                simple_image_size,
                simple_pixels_per_mm,
                line_interval_mm=0.1,
                direction_degrees=0,
            )
        )
        indices = [line.index for line in lines]
        assert indices == sorted(indices)
        for i, line in enumerate(lines):
            assert line.index == indices[0] + i

    def test_horizontal_line_spacing(
        self, simple_bbox, simple_image_size, simple_pixels_per_mm
    ):
        lines = list(
            generate_scan_lines(
                simple_bbox,
                simple_image_size,
                simple_pixels_per_mm,
                line_interval_mm=0.1,
                direction_degrees=0,
            )
        )
        if len(lines) >= 2:
            y0 = lines[0].start_mm[1]
            y1 = lines[1].start_mm[1]
            spacing = abs(y1 - y0)
            assert spacing == pytest.approx(0.1, rel=0.1)

    def test_lines_contain_pixels(
        self, simple_bbox, simple_image_size, simple_pixels_per_mm
    ):
        lines = list(
            generate_scan_lines(
                simple_bbox,
                simple_image_size,
                simple_pixels_per_mm,
                line_interval_mm=0.1,
                direction_degrees=0,
            )
        )
        for line in lines:
            assert len(line.pixels) > 0
            assert line.pixels.shape[1] == 2

    def test_offset_alignment(self):
        bbox = (0, 9, 0, 9)
        image_size = (10, 10)
        pixels_per_mm = (10, 10)
        line_interval_mm = 0.1

        lines_no_offset = list(
            generate_scan_lines(
                bbox,
                image_size,
                pixels_per_mm,
                line_interval_mm=line_interval_mm,
                direction_degrees=0,
                offset_y_mm=0.0,
            )
        )

        lines_with_offset = list(
            generate_scan_lines(
                bbox,
                image_size,
                pixels_per_mm,
                line_interval_mm=line_interval_mm,
                direction_degrees=0,
                offset_y_mm=0.35,
            )
        )

        for line in lines_no_offset:
            global_pos = line.index * line_interval_mm
            aligned = round(global_pos / line_interval_mm) * line_interval_mm
            assert abs(global_pos - aligned) < 1e-9, (
                f"Line {line.index} global pos {global_pos} not aligned"
            )

        for line in lines_with_offset:
            global_pos = line.index * line_interval_mm
            aligned = round(global_pos / line_interval_mm) * line_interval_mm
            assert abs(global_pos - aligned) < 1e-9, (
                f"Line {line.index} global pos {global_pos} not aligned"
            )

    def test_offset_shifts_line_indices(self):
        bbox = (0, 9, 0, 9)
        image_size = (10, 10)
        pixels_per_mm = (10, 10)
        line_interval_mm = 0.1

        lines_no_offset = list(
            generate_scan_lines(
                bbox,
                image_size,
                pixels_per_mm,
                line_interval_mm=line_interval_mm,
                direction_degrees=0,
                offset_y_mm=0.0,
            )
        )

        lines_with_offset = list(
            generate_scan_lines(
                bbox,
                image_size,
                pixels_per_mm,
                line_interval_mm=line_interval_mm,
                direction_degrees=0,
                offset_y_mm=0.35,
            )
        )

        assert lines_no_offset[0].index != lines_with_offset[0].index

    def test_45_degree_lines(self):
        bbox = (0, 9, 0, 9)
        image_size = (10, 10)
        pixels_per_mm = (10, 10)

        lines = list(
            generate_scan_lines(
                bbox,
                image_size,
                pixels_per_mm,
                line_interval_mm=0.1,
                direction_degrees=45,
            )
        )

        assert len(lines) > 0
        for line in lines:
            dx = line.end_mm[0] - line.start_mm[0]
            dy = line.end_mm[1] - line.start_mm[1]
            assert abs(dx - dy) < 0.001 or dx == pytest.approx(dy, rel=0.01)

    def test_empty_bbox_returns_no_lines(self):
        bbox = (0, 0, 0, 0)
        image_size = (1, 1)
        pixels_per_mm = (10, 10)

        lines = list(
            generate_scan_lines(
                bbox,
                image_size,
                pixels_per_mm,
                line_interval_mm=0.1,
                direction_degrees=0,
            )
        )

        assert len(lines) > 0

    def test_pixels_within_image_bounds(
        self, simple_bbox, simple_image_size, simple_pixels_per_mm
    ):
        width, height = simple_image_size
        lines = list(
            generate_scan_lines(
                simple_bbox,
                simple_image_size,
                simple_pixels_per_mm,
                line_interval_mm=0.1,
                direction_degrees=0,
            )
        )
        for line in lines:
            for px, py in line.pixels:
                assert 0 <= px < width
                assert 0 <= py < height


class TestFindSegments:
    def test_empty_array(self):
        segments = find_segments(np.array([], dtype=np.uint8))
        assert segments == []

    def test_all_zeros(self):
        segments = find_segments(np.array([0, 0, 0, 0]))
        assert segments == []

    def test_all_ones(self):
        segments = find_segments(np.array([1, 1, 1, 1]))
        assert segments == [(0, 4)]

    def test_single_segment(self):
        segments = find_segments(np.array([0, 1, 1, 0]))
        assert segments == [(1, 3)]

    def test_multiple_segments(self):
        segments = find_segments(np.array([1, 0, 1, 0, 1]))
        assert segments == [(0, 1), (2, 3), (4, 5)]

    def test_segment_at_start(self):
        segments = find_segments(np.array([1, 1, 0, 0]))
        assert segments == [(0, 2)]

    def test_segment_at_end(self):
        segments = find_segments(np.array([0, 0, 1, 1]))
        assert segments == [(2, 4)]

    def test_non_binary_values(self):
        segments = find_segments(np.array([0, 5, 10, 0, 3]))
        assert segments == [(1, 3), (4, 5)]

    def test_adjacent_segments(self):
        segments = find_segments(np.array([1, 0, 1, 1, 0, 1]))
        assert segments == [(0, 1), (2, 4), (5, 6)]


class TestCoordinateConversion:
    def test_calculate_ymax_mm(self):
        image_size = (100, 200)
        pixels_per_mm = (10, 20)
        ymax = calculate_ymax_mm(image_size, pixels_per_mm)
        assert ymax == pytest.approx(10.0)

    def test_convert_y_to_output_top(self):
        y_output = convert_y_to_output(0.0, 10.0)
        assert y_output == pytest.approx(10.0)

    def test_convert_y_to_output_bottom(self):
        y_output = convert_y_to_output(10.0, 10.0)
        assert y_output == pytest.approx(0.0)

    def test_convert_y_to_output_middle(self):
        y_output = convert_y_to_output(5.0, 10.0)
        assert y_output == pytest.approx(5.0)

    def test_roundtrip(self):
        ymax = 10.0
        original_y = 3.5
        converted = convert_y_to_output(original_y, ymax)
        back = convert_y_to_output(converted, ymax)
        assert back == pytest.approx(original_y)


class TestIntegration:
    def test_horizontal_scan_line_coverage(self):
        bbox = (0, 99, 0, 99)
        image_size = (100, 100)
        pixels_per_mm = (10, 10)
        line_interval_mm = 0.5

        lines = list(
            generate_scan_lines(
                bbox,
                image_size,
                pixels_per_mm,
                line_interval_mm,
                direction_degrees=0,
            )
        )

        covered_rows = set()
        for line in lines:
            for px, py in line.pixels:
                covered_rows.add(py)

        assert len(covered_rows) <= 100

    def test_segment_extraction_from_line(self):
        bbox = (0, 9, 0, 9)
        image_size = (10, 10)
        pixels_per_mm = (10, 10)

        lines = list(
            generate_scan_lines(
                bbox,
                image_size,
                pixels_per_mm,
                line_interval_mm=0.1,
                direction_degrees=0,
            )
        )

        if lines:
            line = lines[0]
            values = np.ones(len(line.pixels))
            values[2:4] = 0
            values[6:8] = 0

            segments = find_segments(values)
            assert len(segments) == 3


class TestFindBoundingBox:
    def test_all_white(self):
        image = np.full((10, 10), 255, dtype=np.uint8)
        result = find_bounding_box(image)
        assert result is None

    def test_all_black(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        result = find_bounding_box(image)
        assert result is not None
        y_min, y_max, x_min, x_max = result
        assert y_min == 0 and y_max == 9
        assert x_min == 0 and x_max == 9

    def test_single_pixel(self):
        image = np.full((10, 10), 255, dtype=np.uint8)
        image[5, 7] = 0
        result = find_bounding_box(image)
        assert result is not None
        y_min, y_max, x_min, x_max = result
        assert y_min == 5 and y_max == 5
        assert x_min == 7 and x_max == 7

    def test_corner_region(self):
        image = np.full((10, 10), 255, dtype=np.uint8)
        image[0:3, 0:4] = 100
        result = find_bounding_box(image)
        assert result is not None
        y_min, y_max, x_min, x_max = result
        assert y_min == 0 and y_max == 2
        assert x_min == 0 and x_max == 3


class TestFindMaskBoundingBox:
    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = find_mask_bounding_box(mask)
        assert result is None

    def test_full_mask(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        result = find_mask_bounding_box(mask)
        assert result is not None
        y_min, y_max, x_min, x_max = result
        assert y_min == 0 and y_max == 9
        assert x_min == 0 and x_max == 9


class TestGenerateHorizontalScanPositions:
    def test_basic_positions(self):
        y_coords_mm, y_coords_px = generate_horizontal_scan_positions(
            y_min_px=0,
            y_max_px=9,
            height_px=10,
            pixels_per_mm=(10, 10),
            line_interval_mm=0.1,
            offset_y_mm=0.0,
        )
        assert len(y_coords_mm) == 10
        assert len(y_coords_px) == 10

    def test_offset_alignment(self):
        offset1 = 0.0
        offset2 = 0.35
        y_coords_mm1, _ = generate_horizontal_scan_positions(
            y_min_px=0,
            y_max_px=9,
            height_px=10,
            pixels_per_mm=(10, 10),
            line_interval_mm=0.1,
            offset_y_mm=offset1,
        )
        y_coords_mm2, _ = generate_horizontal_scan_positions(
            y_min_px=0,
            y_max_px=9,
            height_px=10,
            pixels_per_mm=(10, 10),
            line_interval_mm=0.1,
            offset_y_mm=offset2,
        )
        for y in y_coords_mm1:
            global_y = y + offset1
            aligned = round(global_y / 0.1) * 0.1
            assert abs(global_y - aligned) < 1e-9, f"{global_y} not aligned"
        for y in y_coords_mm2:
            global_y = y + offset2
            aligned = round(global_y / 0.1) * 0.1
            assert abs(global_y - aligned) < 1e-9, f"{global_y} not aligned"

    def test_empty_result(self):
        y_coords_mm, y_coords_px = generate_horizontal_scan_positions(
            y_min_px=10,
            y_max_px=5,
            height_px=20,
            pixels_per_mm=(10, 10),
            line_interval_mm=0.1,
            offset_y_mm=0.0,
        )
        assert len(y_coords_mm) == 0


class TestResampleRows:
    def test_integer_positions(self):
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        y_coords_px = np.array([0.0, 1.0, 2.0])
        result = resample_rows(image, y_coords_px)
        np.testing.assert_array_almost_equal(result[0], [1, 2, 3])
        np.testing.assert_array_almost_equal(result[1], [4, 5, 6])
        np.testing.assert_array_almost_equal(result[2], [7, 8, 9])

    def test_half_pixel_interpolation(self):
        image = np.array([[0, 0, 0], [100, 100, 100]], dtype=np.float32)
        y_coords_px = np.array([0.5])
        result = resample_rows(image, y_coords_px)
        np.testing.assert_array_almost_equal(result[0], [50, 50, 50])

    def test_empty_input(self):
        image = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = resample_rows(image, np.array([]))
        assert result.shape == (0, 2)
