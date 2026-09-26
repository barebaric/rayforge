"""Calibration target backed by a grid of fiducial markers."""

import logging
from dataclasses import dataclass
from typing import ClassVar

import cv2
import numpy as np

from .target import (
    CalibrationTarget,
    CalibrationTargetType,
    ConfigField,
    SummaryRow,
    TargetConfig,
    normalize_detection,
    to_gray,
)

logger = logging.getLogger(__name__)

# Each marker contributes its four corners to the calibration.
CORNERS_PER_MARKER = 4

# Marker counts of the standard dictionaries, used to validate the ID
# offset before OpenCV reads past the end of the dictionary. Keyed by
# constant name so dictionaries missing from an OpenCV build are
# simply skipped.
_DICTIONARY_SIZES = (
    ("DICT_4X4_50", 50),
    ("DICT_4X4_100", 100),
    ("DICT_4X4_250", 250),
    ("DICT_4X4_1000", 1000),
    ("DICT_5X5_50", 50),
    ("DICT_5X5_100", 100),
    ("DICT_5X5_250", 250),
    ("DICT_5X5_1000", 1000),
    ("DICT_6X6_50", 50),
    ("DICT_6X6_100", 100),
    ("DICT_6X6_250", 250),
    ("DICT_6X6_1000", 1000),
    ("DICT_7X7_50", 50),
    ("DICT_7X7_100", 100),
    ("DICT_7X7_250", 250),
    ("DICT_7X7_1000", 1000),
    ("DICT_ARUCO_ORIGINAL", 1024),
    ("DICT_APRILTAG_16h5", 30),
    ("DICT_APRILTAG_25h9", 35),
    ("DICT_APRILTAG_36h10", 2320),
    ("DICT_APRILTAG_36h11", 587),
)

_APRILTAG_DICTIONARIES = (
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
)


def dictionary_marker_count(dictionary_id: int) -> int | None:
    """Return how many markers a dictionary holds, if it is known."""
    for name, size in _DICTIONARY_SIZES:
        if getattr(cv2.aruco, name, None) == dictionary_id:
            return size
    return None


def is_apriltag_dictionary(dictionary_id: int) -> bool:
    """Return True when the dictionary holds AprilTag markers."""
    return any(
        getattr(cv2.aruco, name, None) == dictionary_id
        for name in _APRILTAG_DICTIONARIES
    )


# ID layout options. The origin names the corner holding the first id;
# the order names the axis consecutive ids run along first.
ID_ORIGINS = ("top_left", "top_right", "bottom_left", "bottom_right")
ID_ORDERS = ("rows", "columns")


@dataclass
class ArucoGridConfig(TargetConfig):
    markers_x: int = 4
    markers_y: int = 6
    marker_length_mm: float = 30.0
    marker_separation_mm: float = 15.0
    # First dictionary id on the sheet. 0 starts at the beginning of
    # the dictionary; a higher value describes a sheet whose ids do
    # not start at 0, e.g. one tile of a larger printed set.
    marker_id_offset: int = 0
    # Corner holding the first id, and the axis consecutive ids run
    # along first. Together they describe sheets numbered from any
    # corner, row by row or column by column.
    id_origin: str = "top_left"
    id_order: str = "rows"
    dictionary_id: int = cv2.aruco.DICT_4X4_50

    FIELDS: ClassVar[tuple[ConfigField, ...]] = (
        ConfigField(key="markers_x", lower=2, upper=12, integer=True),
        ConfigField(key="markers_y", lower=2, upper=14, integer=True),
        ConfigField(
            key="marker_length_mm",
            lower=1.0,
            upper=300.0,
            step=1.0,
        ),
        ConfigField(
            key="marker_separation_mm",
            lower=0.0,
            upper=300.0,
            step=1.0,
        ),
        ConfigField(
            key="marker_id_offset",
            lower=0,
            upper=10000,
            integer=True,
        ),
    )


class ArucoGridTarget(CalibrationTarget):
    """A grid of standalone fiducial markers, ArUco or AprilTag.

    Compared to a ChArUco board this tolerates partial occlusion much
    better, which suits cameras looking down at a cluttered bed. The
    trade-off is slightly worse angular accuracy, since the usable
    points are the marker corners.
    """

    target_type = CalibrationTargetType.ARUCO_GRID
    config_class = ArucoGridConfig
    MIN_MARKER_PIXELS = 10
    TARGET_MARKER_SIZE_MM = 30.0
    MAX_COLUMNS = 12
    MAX_ROWS = 14

    def __init__(self, config: ArucoGridConfig):
        if config.marker_length_mm <= 0:
            raise ValueError("Marker length must be positive")
        if config.marker_separation_mm < 0:
            raise ValueError("Marker separation cannot be negative")
        if config.marker_id_offset < 0:
            raise ValueError("Marker ID offset cannot be negative")
        if config.id_origin not in ID_ORIGINS:
            raise ValueError(f"Unknown ID origin: {config.id_origin!r}")
        if config.id_order not in ID_ORDERS:
            raise ValueError(f"Unknown ID order: {config.id_order!r}")
        marker_count = config.markers_x * config.markers_y
        dictionary_size = dictionary_marker_count(config.dictionary_id)
        if (
            dictionary_size is not None
            and config.marker_id_offset + marker_count > dictionary_size
        ):
            last = config.marker_id_offset + marker_count - 1
            raise ValueError(
                f"Marker IDs {config.marker_id_offset}..{last} "
                f"exceed dictionary size ({dictionary_size})"
            )
        self.config = config
        self._board = None
        self._detector = None
        self._object_points: np.ndarray | None = None
        self._create()

    def _create(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(
            self.config.dictionary_id
        )
        self._board = cv2.aruco.GridBoard(
            size=(self.config.markers_x, self.config.markers_y),
            markerLength=self.config.marker_length_mm,
            markerSeparation=self.config.marker_separation_mm,
            dictionary=dictionary,
        )
        detector_params = cv2.aruco.DetectorParameters()
        if is_apriltag_dictionary(self.config.dictionary_id):
            # AprilTag corners refine best with the tag-specific
            # method rather than generic sub-pixel refinement.
            method = cv2.aruco.CORNER_REFINE_APRILTAG
        else:
            method = cv2.aruco.CORNER_REFINE_SUBPIX
        detector_params.cornerRefinementMethod = method
        detector_params.cornerRefinementWinSize = 5
        detector_params.cornerRefinementMaxIterations = 30
        detector_params.cornerRefinementMinAccuracy = 0.1
        self._detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    @property
    def board(self):
        return self._board

    @property
    def marker_count(self) -> int:
        return self.config.markers_x * self.config.markers_y

    @property
    def point_count(self) -> int:
        return self.marker_count * CORNERS_PER_MARKER

    @property
    def card_size_mm(self) -> tuple[float, float]:
        return self._grid_size_mm()

    def _grid_size_mm(self) -> tuple[float, float]:
        width = (
            self.config.markers_x * self.config.marker_length_mm
            + (self.config.markers_x - 1) * self.config.marker_separation_mm
        )
        height = (
            self.config.markers_y * self.config.marker_length_mm
            + (self.config.markers_y - 1) * self.config.marker_separation_mm
        )
        return width, height

    def _grid_id(self, col: int, row: int) -> int:
        """Return the dictionary id printed at a grid position.

        ``col`` grows to the right and ``row`` grows down, matching
        the object point table. The origin corner holds the offset id
        and consecutive ids run along rows (or columns) first.
        """
        cols = self.config.markers_x
        rows = self.config.markers_y
        if self.config.id_origin in ("top_right", "bottom_right"):
            col = cols - 1 - col
        if self.config.id_origin in ("bottom_left", "bottom_right"):
            row = rows - 1 - row
        if self.config.id_order == "columns":
            rank = col * rows + row
        else:
            rank = row * cols + col
        return self.config.marker_id_offset + rank

    def _board_index(self, marker_id: int) -> int | None:
        """Return the 0-based board index for a dictionary id.

        Returns None for ids outside the described sheet, which the
        detector is then told to ignore.
        """
        cols = self.config.markers_x
        rows = self.config.markers_y
        rank = int(marker_id) - self.config.marker_id_offset
        if not 0 <= rank < cols * rows:
            return None
        if self.config.id_order == "columns":
            col, row = divmod(rank, rows)
        else:
            row, col = divmod(rank, cols)
        if self.config.id_origin in ("top_right", "bottom_right"):
            col = cols - 1 - col
        if self.config.id_origin in ("bottom_left", "bottom_right"):
            row = rows - 1 - row
        return row * cols + col

    def object_points(self) -> np.ndarray:
        """Return marker corners flattened to (N, 3).

        OpenCV orders the four corners of a detected marker the same way
        as :meth:`GridBoard.getObjPoints` lists them, so the board
        marker with index ``m`` owns rows ``m * 4`` through
        ``m * 4 + 3`` of this table. The board index is the dictionary
        id minus the configured ID offset.
        """
        if self._object_points is None:
            assert self._board is not None
            per_marker = self._board.getObjPoints()
            stacked = [
                np.asarray(points, dtype=np.float32).reshape(-1, 3)
                for points in per_marker
            ]
            self._object_points = (
                np.vstack(stacked)
                if stacked
                else np.zeros((0, 3), dtype=np.float32)
            )
        return self._object_points

    @classmethod
    def recommend_config(
        cls,
        card_width_mm: float,
        card_height_mm: float,
        camera_resolution: tuple[int, int] = (640, 480),
        surface_size_mm: tuple[float, float] | None = None,
    ) -> ArucoGridConfig:
        min_dim = min(camera_resolution)
        if surface_size_mm:
            mm_per_pixel = min(surface_size_mm) / min_dim
        else:
            mm_per_pixel = card_width_mm / min_dim

        marker_length = cls.TARGET_MARKER_SIZE_MM
        if mm_per_pixel > 0:
            estimated_marker_pixels = marker_length / mm_per_pixel
            if estimated_marker_pixels < cls.MIN_MARKER_PIXELS:
                marker_length *= (
                    cls.MIN_MARKER_PIXELS / estimated_marker_pixels
                )

        separation = marker_length * 0.5
        unit = marker_length + separation

        markers_x = max(2, min(cls.MAX_COLUMNS, int(card_width_mm / unit)))
        markers_y = max(2, min(cls.MAX_ROWS, int(card_height_mm / unit)))

        return ArucoGridConfig(
            markers_x=markers_x,
            markers_y=markers_y,
            marker_length_mm=round(marker_length, 1),
            marker_separation_mm=round(separation, 1),
        )

    def generate_image(
        self,
        output_size: tuple[int, int] | None = None,
        margin_px: int = 10,
        border_bits: int = 1,
    ) -> np.ndarray:
        """Render the grid with its configured dictionary ids.

        Markers are drawn one by one instead of using
        :meth:`GridBoard.generateImage`, whose constructor signature
        changed across OpenCV releases. The layout matches the board
        geometry exactly: row-major ids starting at the ID offset.
        """
        width_mm, height_mm = self._grid_size_mm()
        if output_size is None:
            px_per_mm = 10
            output_size = (
                int(width_mm * px_per_mm) + 2 * margin_px,
                int(height_mm * px_per_mm) + 2 * margin_px,
            )

        width_px, height_px = output_size
        image = np.full((height_px, width_px), 255, dtype=np.uint8)

        assert self._board is not None
        dictionary = self._board.getDictionary()
        # One scale for both axes so markers stay square even when the
        # requested aspect does not match the grid; the surplus is
        # centred inside the margin.
        scale = min(
            (width_px - 2 * margin_px) / max(width_mm, 1e-6),
            (height_px - 2 * margin_px) / max(height_mm, 1e-6),
        )
        scale = max(scale, 1e-6)
        marker_px = max(1, round(self.config.marker_length_mm * scale))
        step_mm = (
            self.config.marker_length_mm + self.config.marker_separation_mm
        )
        grid_w_px = (
            self.config.markers_x * self.config.marker_length_mm
            + (self.config.markers_x - 1) * self.config.marker_separation_mm
        ) * scale
        grid_h_px = (
            self.config.markers_y * self.config.marker_length_mm
            + (self.config.markers_y - 1) * self.config.marker_separation_mm
        ) * scale
        start_x = round(margin_px + (width_px - 2 * margin_px - grid_w_px) / 2)
        start_y = round(
            margin_px + (height_px - 2 * margin_px - grid_h_px) / 2
        )
        for row in range(self.config.markers_y):
            for col in range(self.config.markers_x):
                marker_id = self._grid_id(col, row)
                marker = cv2.aruco.generateImageMarker(
                    dictionary,
                    marker_id,
                    marker_px,
                    borderBits=border_bits,
                )
                x = round(start_x + col * step_mm * scale)
                y = round(start_y + row * step_mm * scale)
                image[y : y + marker_px, x : x + marker_px] = marker
        return image

    def detect(
        self, image: np.ndarray
    ) -> tuple[list[tuple[float, float]], list[int]] | None:
        assert self._detector is not None
        try:
            marker_corners, marker_ids, _ = self._detector.detectMarkers(
                to_gray(image)
            )
        except cv2.error as error:
            logger.debug("ArUco detection error: %s", error)
            return None

        if marker_corners is None or marker_ids is None:
            return None

        return self._flatten_markers(marker_corners, marker_ids)

    def _flatten_markers(
        self, marker_corners, marker_ids
    ) -> tuple[list[tuple[float, float]], list[int]] | None:
        try:
            corners_array = np.asarray(
                marker_corners, dtype=np.float32
            ).reshape(-1, CORNERS_PER_MARKER, 2)
            ids_array = np.asarray(marker_ids).reshape(-1)
        except (TypeError, ValueError) as error:
            logger.debug("Invalid ArUco detection result: %s", error)
            return None

        if len(corners_array) != len(ids_array):
            logger.debug("ArUco corners and ids are misaligned")
            return None

        points: list[tuple[float, float]] = []
        point_ids: list[int] = []
        for corners, marker_id in zip(corners_array, ids_array):
            board_index = self._board_index(marker_id)
            if board_index is None:
                continue
            base = board_index * CORNERS_PER_MARKER
            for corner, (x, y) in enumerate(corners):
                points.append((float(x), float(y)))
                point_ids.append(base + corner)

        return normalize_detection(points, point_ids)

    def summary(self) -> list[SummaryRow]:
        rows = [
            SummaryRow(
                key="grid_size",
                value=(
                    f"{self.config.markers_x} x {self.config.markers_y}"
                    " markers"
                ),
            ),
            SummaryRow(
                key="marker_size",
                measurement_mm=self.config.marker_length_mm,
            ),
            SummaryRow(
                key="marker_gap",
                measurement_mm=self.config.marker_separation_mm,
            ),
        ]
        if self.config.marker_id_offset > 0:
            rows.append(
                SummaryRow(
                    key="marker_id_offset",
                    value=str(self.config.marker_id_offset),
                )
            )
        return rows

    def draw_detection(
        self,
        image: np.ndarray,
        corners: list[tuple[float, float]],
        ids: list[int],
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        result = image.copy()
        quads: dict[int, dict[int, tuple[float, float]]] = {}
        for point, index in zip(corners, ids):
            board_index, corner = divmod(index, CORNERS_PER_MARKER)
            quads.setdefault(board_index, {})[corner] = point

        for board_index in sorted(quads):
            corners_by_offset = quads[board_index]
            if len(corners_by_offset) != CORNERS_PER_MARKER:
                continue
            quad = [
                corners_by_offset[offset]
                for offset in range(CORNERS_PER_MARKER)
            ]
            polyline = np.array(quad, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(result, [polyline], True, color, 1)
            cv2.putText(
                result,
                str(board_index + self.config.marker_id_offset),
                (int(quad[0][0]), int(quad[0][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

        return result


__all__ = [
    "CORNERS_PER_MARKER",
    "ID_ORDERS",
    "ID_ORIGINS",
    "ArucoGridConfig",
    "ArucoGridTarget",
    "dictionary_marker_count",
    "is_apriltag_dictionary",
]
