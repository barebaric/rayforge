"""Calibration target backed by a plain ArUco marker grid."""

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

# Each ArUco marker contributes its four corners to the calibration.
CORNERS_PER_MARKER = 4


@dataclass
class ArucoGridConfig(TargetConfig):
    markers_x: int = 4
    markers_y: int = 6
    marker_length_mm: float = 30.0
    marker_separation_mm: float = 15.0
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
    )


class ArucoGridTarget(CalibrationTarget):
    """A grid of standalone ArUco markers with no surrounding chessboard.

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
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
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

    def object_points(self) -> np.ndarray:
        """Return marker corners flattened to (N, 3).

        OpenCV orders the four corners of a detected marker the same way
        as :meth:`GridBoard.getObjPoints` lists them, so a marker with id
        ``m`` owns rows ``m * 4`` through ``m * 4 + 3`` of this table.
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
        if output_size is None:
            width_mm, height_mm = self._grid_size_mm()
            px_per_mm = 10
            output_size = (
                int(width_mm * px_per_mm) + 2 * margin_px,
                int(height_mm * px_per_mm) + 2 * margin_px,
            )

        assert self._board is not None
        return self._board.generateImage(
            output_size, marginSize=margin_px, borderBits=border_bits
        )

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
            index = int(marker_id)
            if not 0 <= index < self.marker_count:
                continue
            base = index * CORNERS_PER_MARKER
            for offset, (x, y) in enumerate(corners):
                points.append((float(x), float(y)))
                point_ids.append(base + offset)

        return normalize_detection(points, point_ids)

    def summary(self) -> list[SummaryRow]:
        return [
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
            marker_id, corner = divmod(index, CORNERS_PER_MARKER)
            quads.setdefault(marker_id, {})[corner] = point

        for marker_id in sorted(quads):
            corners_by_offset = quads[marker_id]
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
                str(marker_id),
                (int(quad[0][0]), int(quad[0][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

        return result


__all__ = ["CORNERS_PER_MARKER", "ArucoGridConfig", "ArucoGridTarget"]
