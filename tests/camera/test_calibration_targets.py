"""Tests for the pluggable calibration targets.

The interesting risk in this subsystem is the correspondence between a
detected image point and its object point. Each target derives its ids
differently (ChArUco corner indices, ``marker_id * 4 + corner`` for a
marker grid, a row-major lattice for a dot grid), so these tests render
synthetic views through a camera with known intrinsics and check that
the solver recovers them.
"""

import cv2
import numpy as np
import pytest

from rayforge.camera.calibration import (
    CalibrationTargetType,
    available_target_types,
    create_target,
)
from rayforge.camera.calibration.aruco import (
    ArucoGridConfig,
    ArucoGridTarget,
    dictionary_marker_count,
)
from rayforge.camera.calibration.calibrator import CameraCalibrator
from rayforge.camera.calibration.charuco import CharucoConfig, CharucoTarget
from rayforge.camera.calibration.dotgrid import DotGridConfig, DotGridTarget
from rayforge.camera.calibration.target import normalize_detection

ALL_TYPES = list(available_target_types())


def make_target(target_type: CalibrationTargetType, staggered=True):
    """Build a target that is small enough to render quickly.

    The dot grid defaults to a staggered sheet because that is the harder
    of the two arrangements and the one printed dot sheets usually use.
    """
    if target_type is CalibrationTargetType.CHARUCO:
        return CharucoTarget(CharucoConfig())
    if target_type is CalibrationTargetType.ARUCO_GRID:
        return ArucoGridTarget(
            ArucoGridConfig(
                markers_x=3,
                markers_y=4,
                marker_length_mm=30.0,
                marker_separation_mm=15.0,
            )
        )
    if staggered:
        return DotGridTarget(
            DotGridConfig(
                dots_x=5,
                dots_y=6,
                spacing_mm=20.0,
                dot_diameter_mm=10.0,
                row_spacing_mm=20.0 * 0.866,
                row_offset_mm=10.0,
            )
        )
    return DotGridTarget(
        DotGridConfig(
            dots_x=5, dots_y=6, spacing_mm=20.0, dot_diameter_mm=10.0
        )
    )


@pytest.mark.parametrize("staggered", [True, False])
def test_dot_grid_detects_a_rectangular_and_a_staggered_sheet(staggered):
    target = make_target(CalibrationTargetType.DOT_GRID, staggered=staggered)

    detection = target.detect(target.generate_image())

    assert detection is not None
    corners, ids = detection
    assert ids == list(range(target.point_count))
    assert len(corners) == len(ids)


@pytest.mark.parametrize("target_type", ALL_TYPES)
def test_detects_its_own_generated_pattern(target_type):
    target = make_target(target_type)

    detection = target.detect(target.generate_image())

    assert detection is not None, f"{target_type} failed to detect its pattern"
    corners, ids = detection
    assert len(corners) == len(ids)
    assert len(corners) == target.point_count


@pytest.mark.parametrize("target_type", ALL_TYPES)
def test_detection_ids_index_the_object_point_table(target_type):
    target = make_target(target_type)
    object_points = target.object_points()

    detection = target.detect(target.generate_image())

    assert detection is not None
    _corners, ids = detection
    assert min(ids) >= 0
    assert max(ids) < len(object_points)


@pytest.mark.parametrize("target_type", ALL_TYPES)
def test_detect_returns_none_for_a_blank_image(target_type):
    target = make_target(target_type)
    blank = np.full((480, 640), 255, dtype=np.uint8)

    assert target.detect(blank) is None


def test_charuco_reports_every_chessboard_corner():
    target = CharucoTarget(CharucoConfig(squares_x=5, squares_y=7))

    detection = target.detect(target.generate_image())

    assert detection is not None
    _corners, ids = detection
    assert target.point_count == (5 - 1) * (7 - 1)
    assert sorted(ids) == list(range(target.point_count))


def test_aruco_ids_expand_each_marker_into_four_corners():
    target = ArucoGridTarget(ArucoGridConfig(markers_x=3, markers_y=4))

    detection = target.detect(target.generate_image())

    assert detection is not None
    _corners, ids = detection
    # Four corners per marker, and every marker on the board is seen.
    assert target.point_count == 12 * 4
    assert sorted(ids) == list(range(48))
    # detectMarkers may return markers in any order, but each marker's
    # four corners must stay grouped under one marker id.
    for marker_id in range(12):
        base = marker_id * 4
        corners_for_marker = [i for i in ids if base <= i < base + 4]
        assert len(corners_for_marker) == 4


def test_aruco_drops_markers_outside_the_board():
    target = ArucoGridTarget(ArucoGridConfig(markers_x=2, markers_y=2))
    corners = np.array(
        [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]],
        dtype=np.float32,
    )

    # Marker 1 is the last on a 2x2 board, marker 7 is not on it at all.
    result = target._flatten_markers(corners, np.array([[1]], dtype=np.int32))
    outside = target._flatten_markers(corners, np.array([[7]], dtype=np.int32))

    assert result is not None
    assert max(result[1]) < target.point_count
    assert outside is None


def test_marker_id_offset_shifts_detected_ids():
    """A grid need not start at dictionary id 0."""
    target = ArucoGridTarget(
        ArucoGridConfig(
            markers_x=3,
            markers_y=3,
            marker_length_mm=30.0,
            marker_separation_mm=15.0,
            marker_id_offset=10,
        )
    )

    detection = target.detect(target.generate_image())

    assert detection is not None
    _corners, ids = detection
    assert sorted(ids) == list(range(target.point_count))


def test_marker_id_offset_drops_ids_outside_the_board():
    target = ArucoGridTarget(
        ArucoGridConfig(markers_x=2, markers_y=2, marker_id_offset=5)
    )
    corners = np.array(
        [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]],
        dtype=np.float32,
    )

    # Dictionary id 5 is the first board marker, id 8 the last.
    first = target._flatten_markers(corners, np.array([[5]]))
    last = target._flatten_markers(corners, np.array([[8]]))
    # Id 4 belongs to the sheet before this one, id 9 to the next.
    before = target._flatten_markers(corners, np.array([[4]]))
    after = target._flatten_markers(corners, np.array([[9]]))

    assert first is not None
    assert first[1] == [0, 1, 2, 3]
    assert last is not None
    assert last[1] == [12, 13, 14, 15]
    assert before is None
    assert after is None


def test_marker_id_offset_rejects_negative_and_overflow():
    # A 2x2 board in DICT_4X4_50 (50 markers): 46 is the last
    # offset that still fits.
    with pytest.raises(ValueError):
        ArucoGridTarget(ArucoGridConfig(marker_id_offset=-1))
    ArucoGridTarget(
        ArucoGridConfig(markers_x=2, markers_y=2, marker_id_offset=46)
    )
    with pytest.raises(ValueError):
        ArucoGridTarget(
            ArucoGridConfig(markers_x=2, markers_y=2, marker_id_offset=47)
        )


@pytest.mark.parametrize(
    "origin", ["top_left", "top_right", "bottom_left", "bottom_right"]
)
@pytest.mark.parametrize("order", ["rows", "columns"])
def test_marker_id_layouts_are_bijections(origin, order):
    """Every layout must pair each dictionary id with one position."""
    target = ArucoGridTarget(
        ArucoGridConfig(
            markers_x=3,
            markers_y=4,
            marker_id_offset=7,
            id_origin=origin,
            id_order=order,
        )
    )

    board_ids = [target._board_index(7 + rank) for rank in range(12)]
    assert sorted(
        board_id for board_id in board_ids if board_id is not None
    ) == list(range(12))
    for col in range(3):
        for row in range(4):
            board_index = row * 3 + col
            assert (
                target._board_index(target._grid_id(col, row)) == board_index
            )

    # The origin corner always holds the offset id.
    corners = {
        "top_left": (0, 0),
        "top_right": (2, 0),
        "bottom_left": (0, 3),
        "bottom_right": (2, 3),
    }
    assert target._grid_id(*corners[origin]) == 7

    # Consecutive ids step along the fast axis away from the origin.
    if order == "rows":
        step = -1 if "right" in origin else 1
        assert target._grid_id(1, 0) - target._grid_id(0, 0) == step
    else:
        step = -1 if "bottom" in origin else 1
        assert target._grid_id(0, 1) - target._grid_id(0, 0) == step


def test_marker_id_origin_bottom_left_detects_render():
    """A sheet numbered from the bottom-left must still calibrate."""
    target = ArucoGridTarget(
        ArucoGridConfig(
            markers_x=3,
            markers_y=3,
            marker_length_mm=30.0,
            marker_separation_mm=15.0,
            id_origin="bottom_left",
        )
    )
    image = target.generate_image()

    detection = target.detect(image)

    assert detection is not None
    corners, ids = detection
    assert sorted(ids) == list(range(target.point_count))
    # Dictionary id 0 is printed at the bottom-left of the sheet.
    # Detections carry point ids, and board b owns points b * 4
    # through b * 4 + 3.
    board_id = target._board_index(0)
    assert board_id is not None
    height, width = image.shape[:2]
    x, y = corners[ids.index(board_id * 4)]
    assert x < width / 2
    assert y > height / 2

    restored = create_target(
        CalibrationTargetType.ARUCO_GRID, target.config.to_dict()
    )
    assert isinstance(restored, ArucoGridTarget)
    assert restored.config.id_origin == "bottom_left"
    assert restored.config.id_order == "rows"


def test_marker_id_layout_rejects_unknown_values():
    with pytest.raises(ValueError):
        ArucoGridTarget(ArucoGridConfig(id_origin="centre"))
    with pytest.raises(ValueError):
        ArucoGridTarget(ArucoGridConfig(id_order="diagonal"))


def test_apriltag_dictionary_uses_tag_refinement():
    target = ArucoGridTarget(
        ArucoGridConfig(
            dictionary_id=cv2.aruco.DICT_APRILTAG_36h11,
        )
    )

    detection = target.detect(target.generate_image())

    assert detection is not None
    assert len(detection[0]) == target.point_count


def test_dot_grid_ids_are_a_row_major_lattice():
    target = DotGridTarget(DotGridConfig(dots_x=4, dots_y=3))

    detection = target.detect(target.generate_image())

    assert detection is not None
    _corners, ids = detection
    assert ids == list(range(12))

    object_points = target.object_points()
    spacing = target.config.spacing_mm
    assert object_points[1][0] - object_points[0][0] == pytest.approx(spacing)
    assert object_points[4][1] - object_points[0][1] == pytest.approx(spacing)


def test_dot_grid_rows_can_be_staggered():
    """A hexagonal sheet: alternate rows shifted by half a pitch."""
    config = DotGridConfig(
        dots_x=4,
        dots_y=5,
        spacing_mm=20.0,
        dot_diameter_mm=10.0,
        row_spacing_mm=17.3,
        row_offset_mm=10.0,
    )
    target = DotGridTarget(config)

    detection = target.detect(target.generate_image())

    assert detection is not None
    _corners, ids = detection
    assert ids == list(range(20))

    points = target.object_points()
    # Row 0 starts at x=0, row 1 is shifted by the offset.
    assert points[0][0] == pytest.approx(0.0)
    assert points[4][0] == pytest.approx(config.row_offset_mm)
    # Rows are one vertical pitch apart.
    assert points[4][1] == pytest.approx(config.row_spacing_mm)


def test_dot_grid_row_spacing_defaults_to_the_column_pitch():
    target = DotGridTarget(DotGridConfig(spacing_mm=25.0, dots_x=4, dots_y=4))

    assert target.row_pitch_mm == pytest.approx(25.0)
    assert not target.is_staggered


def test_dot_grid_declines_a_sheet_with_the_wrong_dot_count():
    """Reading part of a sheet would poison the solve, so refuse it."""
    sheet = DotGridTarget(
        DotGridConfig(
            dots_x=5, dots_y=5, spacing_mm=20.0, dot_diameter_mm=10.0
        )
    )
    # Configured for one more row than the sheet actually has.
    target = DotGridTarget(
        DotGridConfig(
            dots_x=5, dots_y=6, spacing_mm=20.0, dot_diameter_mm=10.0
        )
    )

    assert target.detect(sheet.generate_image()) is None


def test_dot_grid_keeps_its_reading_when_the_sheet_moves():
    """The cached row direction must not drift as the sheet is moved.

    Detection remembers the last good row direction to stay cheap, so a
    second view of the same sheet, shifted and slightly turned, has to
    come back with the same correspondence rather than a stale one.
    """
    target = make_target(CalibrationTargetType.DOT_GRID, staggered=False)
    pattern, affine = _millimetre_to_pixel_affine(target)

    assert target.detect(pattern) is not None

    object_points = target.object_points()
    height, width = pattern.shape[:2]
    # A larger canvas so the turn and shift do not clip the outer dots.
    canvas = (int(width * 1.4), int(height * 1.4))
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 7.0, 1.0)
    matrix[0, 2] += canvas[0] / 2 - width / 2 + 25
    matrix[1, 2] += canvas[1] / 2 - height / 2 + 15
    moved = cv2.warpAffine(pattern, matrix, canvas, borderValue=255)

    detection = target.detect(moved)

    assert detection is not None, "sheet lost after moving"
    corners_px, ids = detection
    selected = object_points[np.array(ids, dtype=np.int32), :2]
    homogeneous = np.column_stack([selected, np.ones(len(selected))]).astype(
        np.float32
    )
    in_pattern = (homogeneous @ affine)[:, :2]
    homogeneous = np.column_stack(
        [in_pattern, np.ones(len(in_pattern))]
    ).astype(np.float32)
    predicted = homogeneous @ matrix.T

    error = np.linalg.norm(
        np.asarray(corners_px, dtype=np.float32) - predicted, axis=1
    )
    assert error.max() < 3.0, f"reading drifted by {error.max():.2f}px"


def test_staggered_card_size_allow_for_the_shifted_rows():
    target = DotGridTarget(
        DotGridConfig(
            dots_x=4,
            dots_y=4,
            spacing_mm=20.0,
            dot_diameter_mm=10.0,
            row_spacing_mm=17.3,
            row_offset_mm=10.0,
        )
    )

    width, height = target.card_size_mm

    # Three pitches plus the half-pitch shift, plus one dot diameter.
    assert width == pytest.approx(3 * 20.0 + 10.0 + 10.0)
    assert height == pytest.approx(3 * 17.3 + 10.0)


def test_recommended_dot_grid_is_staggered():
    from rayforge.camera.calibration import recommend_target

    target = recommend_target(
        CalibrationTargetType.DOT_GRID,
        card_width_mm=100.0,
        card_height_mm=140.0,
    )

    assert isinstance(target, DotGridTarget)
    assert target.is_staggered
    assert target.row_offset_mm == pytest.approx(
        target.config.spacing_mm / 2, rel=0.01
    )
    assert target.row_pitch_mm < target.config.spacing_mm


def test_card_size_accounts_for_the_outer_feature():
    target = DotGridTarget(
        DotGridConfig(
            dots_x=4, dots_y=3, spacing_mm=20.0, dot_diameter_mm=10.0
        )
    )

    width, height = target.card_size_mm

    assert width == pytest.approx(3 * 20.0 + 10.0)
    assert height == pytest.approx(2 * 20.0 + 10.0)


def test_aruco_card_size_accounts_for_the_gaps():
    target = ArucoGridTarget(
        ArucoGridConfig(
            markers_x=4,
            markers_y=3,
            marker_length_mm=30.0,
            marker_separation_mm=15.0,
        )
    )

    width, height = target.card_size_mm

    assert width == pytest.approx(4 * 30.0 + 3 * 15.0)
    assert height == pytest.approx(3 * 30.0 + 2 * 15.0)


def test_config_round_trips_through_a_dict():
    for target_type in ALL_TYPES:
        original = make_target(target_type)
        restored = create_target(target_type, original.config.to_dict())

        assert isinstance(
            restored, (CharucoTarget, ArucoGridTarget, DotGridTarget)
        )
        assert restored.config.to_dict() == original.config.to_dict()


def test_create_target_fills_in_missing_keys():
    target = create_target(CalibrationTargetType.ARUCO_GRID, {})

    assert isinstance(target, ArucoGridTarget)
    assert target.config.markers_x == ArucoGridConfig().markers_x


def test_create_target_rejects_an_unknown_type():
    with pytest.raises(ValueError):
        create_target("not-a-target", {})  # type: ignore[arg-type]


@pytest.mark.parametrize("target_type", ALL_TYPES)
def test_editable_fields_round_trip_through_the_config(target_type):
    """The card page edits geometry by field key, so that has to hold."""
    target = make_target(target_type)
    config = target.config
    fields = type(config).FIELDS

    assert fields, f"{target_type} exposes no editable geometry"
    assert all(hasattr(config, field.key) for field in fields)

    overrides = {
        field.key: (field.lower + field.upper) / 2 for field in fields
    }
    # A ChArUco board is only constructible with the marker strictly
    # inside the square, so do not compare two independent midpoints.
    if "marker_length_mm" in overrides and "square_length_mm" in overrides:
        overrides["marker_length_mm"] = overrides["square_length_mm"] * 0.75
    if "marker_id_offset" in overrides and isinstance(config, ArucoGridConfig):
        # The board must fit inside its marker dictionary: shrink the
        # grid first, then clamp the offset to what remains.
        size = dictionary_marker_count(config.dictionary_id)
        if size is None:
            overrides["marker_id_offset"] = 0
        else:
            markers_x = max(2, int(overrides.get("markers_x", 2)))
            markers_y = max(2, int(overrides.get("markers_y", 2)))
            while markers_x * markers_y > size:
                if markers_x >= markers_y:
                    markers_x -= 1
                else:
                    markers_y -= 1
            overrides["markers_x"] = markers_x
            overrides["markers_y"] = markers_y
            middle = int(overrides["marker_id_offset"])
            overrides["marker_id_offset"] = max(
                0, min(middle, size - markers_x * markers_y)
            )
    config.apply_field_values(overrides)

    read_back = config.to_field_values()
    for field in fields:
        expected = (
            int(overrides[field.key])
            if field.integer
            else overrides[field.key]
        )
        assert read_back[field.key] == pytest.approx(expected)

    restored = create_target(target_type, config.to_dict())
    assert isinstance(
        restored, (CharucoTarget, ArucoGridTarget, DotGridTarget)
    )
    assert restored.config.to_dict() == config.to_dict()


@pytest.mark.parametrize("target_type", ALL_TYPES)
def test_recommended_config_fits_the_requested_card(target_type):
    from rayforge.camera.calibration import recommend_target

    target = recommend_target(
        target_type, card_width_mm=100.0, card_height_mm=140.0
    )

    width, height = target.card_size_mm
    # A recommended pattern is allowed to exceed the card slightly
    # because the renderer adds its own margin, but it must not be
    # wildly larger than what was asked for.
    assert width <= 100.0 * 1.5
    assert height <= 140.0 * 1.5
    assert width > 0
    assert height > 0


def test_config_field_values_tolerate_partial_updates():
    config = DotGridConfig(dots_x=6, dots_y=8)

    config.apply_field_values({"spacing_mm": 25.0})

    assert config.spacing_mm == 25.0
    assert config.dots_x == 6


def test_normalize_detection_rejects_mismatched_lengths():
    corners = np.zeros((5, 2), dtype=np.float32)
    ids = np.zeros(4, dtype=np.int32)

    assert normalize_detection(corners, ids) is None


def test_normalize_detection_rejects_too_few_points():
    corners = np.zeros((3, 2), dtype=np.float32)
    ids = np.zeros(3, dtype=np.int32)

    assert normalize_detection(corners, ids) is None


class _ViewHolder:
    """A target whose detection is swapped per synthetic view.

    The object point table is the real target's, so the test exercises
    the calibrator's public frame-adding path while keeping the detector
    out of the picture.
    """

    def __init__(self, object_points):
        self._object_points = object_points
        self._corners: list[tuple[float, float]] = []
        self._ids: list[int] = []

    def set_view(self, corners, ids):
        self._corners = corners
        self._ids = ids

    def detect(self, image):
        return self._corners, self._ids

    def object_points(self):
        return self._object_points

    @property
    def point_count(self):
        return len(self._object_points)


def _solve_with_projected_views(target, intrinsics, image_size, poses):
    """Feed projected detections through the calibrator and return a result."""
    camera_matrix = np.array(
        [
            [intrinsics[0], 0.0, image_size[0] / 2],
            [0.0, intrinsics[1], image_size[1] / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros(5, dtype=np.float64)
    object_points = target.object_points()
    ids = list(range(len(object_points)))
    blank = np.full((image_size[1], image_size[0]), 255, dtype=np.uint8)

    holder = _ViewHolder(object_points)
    calibrator = CameraCalibrator(holder)  # type: ignore[arg-type]
    for rvec_values, distance in poses:
        rvec = np.array(rvec_values, dtype=np.float64)
        rotation, _ = cv2.Rodrigues(rvec)
        center = object_points.mean(axis=0)
        translation = (np.array([0.0, 0.0, distance]) - rotation @ center)[
            :, None
        ]
        projected, _ = cv2.projectPoints(
            object_points, rvec, translation, camera_matrix, distortion
        )
        corners = [
            (float(x), float(y))
            for x, y in projected.reshape(-1, 2).astype(np.float32)
        ]
        holder.set_view(corners, list(ids))
        accepted, _count, _points = calibrator.detect_and_add_frame(blank)
        assert accepted

    return calibrator.calibrate(image_size)


def test_calibrator_recovers_known_intrinsics_from_projected_points():
    """The id-to-object-point mapping must survive the solve.

    This feeds detections straight from a known camera, which isolates
    the correspondence logic from any detector.
    """
    intrinsics = (700.0, 690.0)
    image_size = (640, 480)
    poses = [
        ((0.05, -0.04, 0.02), 420.0),
        ((-0.07, 0.03, -0.05), 460.0),
        ((0.02, 0.09, 0.10), 500.0),
        ((-0.10, -0.06, 0.03), 440.0),
        ((0.09, 0.07, -0.08), 520.0),
        ((-0.03, 0.11, 0.12), 480.0),
    ]

    for target_type in ALL_TYPES:
        target = make_target(target_type)
        result = _solve_with_projected_views(
            target, intrinsics, image_size, poses
        )

        assert result is not None, f"{target_type} produced no result"
        assert result.camera_matrix[0][0] == pytest.approx(
            intrinsics[0], rel=0.02
        )
        assert result.camera_matrix[1][1] == pytest.approx(
            intrinsics[1], rel=0.02
        )
        assert result.camera_matrix[0][2] == pytest.approx(
            image_size[0] / 2, abs=4.0
        )
        assert result.camera_matrix[1][2] == pytest.approx(
            image_size[1] / 2, abs=4.0
        )
        assert result.rms_error < 1.0


def _millimetre_to_pixel_affine(target):
    """Fit the target's own millimetre space onto its rendered pattern.

    Detecting the flat render gives image points paired with object point
    ids, and the ids index the millimetre table, so the two together
    define the renderer's layout without assuming anything about it.
    """
    image = target.generate_image()
    detection = target.detect(image)
    assert detection is not None, "flat pattern must be detectable"
    corners, ids = detection
    object_points = target.object_points()[np.array(ids, dtype=np.int32)]

    source = np.column_stack(
        [object_points[:, :2], np.ones(len(object_points))]
    ).astype(np.float32)
    destination = np.asarray(corners, dtype=np.float32)
    affine, _residuals, _rank, _singular = np.linalg.lstsq(
        source, destination, rcond=None
    )
    return image, affine


def test_ids_map_to_the_right_points_under_perspective():
    """A tilted view must still pair each point with the right id.

    Detecting a flat pattern only proves the points were found. Warping
    the sheet into an oblique view and checking every correspondence is
    what catches a mislabelled lattice, which would otherwise show up
    only as an inexplicable bad calibration.
    """
    view_size = (700, 560)

    for target_type in ALL_TYPES:
        target = make_target(target_type)
        pattern, affine = _millimetre_to_pixel_affine(target)

        object_points = target.object_points()
        minimum = object_points[:, :2].min(axis=0)
        maximum = object_points[:, :2].max(axis=0)
        corners_mm = np.array(
            [
                [minimum[0], minimum[1]],
                [maximum[0], minimum[1]],
                [maximum[0], maximum[1]],
                [minimum[0], maximum[1]],
            ],
            dtype=np.float32,
        )
        source = (
            np.column_stack([corners_mm, np.ones(4)]).astype(np.float32)
            @ affine
        )
        destination = np.array(
            [
                [140.0, 90.0],
                [560.0, 170.0],
                [520.0, 460.0],
                [110.0, 380.0],
            ],
            dtype=np.float32,
        )

        homography = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(
            pattern,
            homography,
            view_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

        detection = target.detect(warped)

        assert detection is not None, f"{target_type} lost the pattern"
        corners_px, ids = detection
        # The homography maps pattern pixels, so the millimetre table has
        # to go through the fitted affine on the way.
        selected = object_points[np.array(ids, dtype=np.int32), :2]
        homogeneous = np.column_stack(
            [selected, np.ones(len(selected))]
        ).astype(np.float32)
        in_pattern = (homogeneous @ affine)[:, :2]
        homogeneous = np.column_stack(
            [in_pattern, np.ones(len(in_pattern))]
        ).astype(np.float32)
        projected = homogeneous @ homography.T
        expected = projected[:, :2] / projected[:, 2:3]

        error = np.linalg.norm(
            np.asarray(corners_px, dtype=np.float32) - expected, axis=1
        )
        assert error.max() < 3.0, (
            f"{target_type} correspondence error up to {error.max():.2f}px"
        )
