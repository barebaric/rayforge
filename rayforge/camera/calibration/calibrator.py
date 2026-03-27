import logging
from typing import Any, List, Optional, Tuple, cast
import cv2
import numpy as np
from blinker import Signal

from .charuco import CharucoBoard
from .result import CalibrationResult

logger = logging.getLogger(__name__)


class CameraCalibrator:
    MIN_FRAMES = 3
    MIN_CORNERS_PER_FRAME = 4
    MIN_UNIQUE_CORNERS = 10

    def __init__(self, board: CharucoBoard):
        self.board = board
        self._all_corners: List[List[Tuple[float, float]]] = []
        self._all_ids: List[List[int]] = []
        self._frame_count = 0
        self._image_size: Optional[Tuple[int, int]] = None

        self.frame_added = Signal()
        self.frame_rejected = Signal()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def total_corners(self) -> int:
        return sum(len(c) for c in self._all_corners)

    @property
    def unique_corners(self) -> int:
        return len({c for ids in self._all_ids for c in ids})

    @property
    def corners_per_frame(self) -> List[int]:
        return [len(c) for c in self._all_corners]

    def clear(self):
        self._all_corners.clear()
        self._all_ids.clear()
        self._frame_count = 0
        self._image_size = None

    def detect_and_add_frame(
        self, image: np.ndarray
    ) -> Tuple[bool, int, Optional[List[Tuple[float, float]]]]:
        detection = self.board.detect(image)

        if detection is None:
            self.frame_rejected.send(self, reason="no_detection")
            return False, 0, None

        corners, ids = detection

        if len(corners) < self.MIN_CORNERS_PER_FRAME:
            self.frame_rejected.send(
                self, reason="insufficient_corners", count=len(corners)
            )
            return False, len(corners), corners

        if self._image_size is None:
            h, w = image.shape[:2]
            self._image_size = (w, h)

        self._all_corners.append(corners)
        self._all_ids.append(ids)
        self._frame_count += 1

        self.frame_added.send(
            self, count=len(corners), total=self._frame_count
        )
        return True, len(corners), corners

    def can_calibrate(self) -> bool:
        if self._frame_count < self.MIN_FRAMES:
            return False

        total_unique_corners = len(
            {corner for ids in self._all_ids for corner in ids}
        )

        return total_unique_corners >= self.MIN_UNIQUE_CORNERS

    def get_coverage(self) -> Tuple[float, float, float, float]:
        """
        Get the spatial coverage of detected corners.

        Returns normalized coverage (0-1) for each quadrant:
        (top-left, top-right, bottom-left, bottom-right)
        """
        if self._image_size is None or not self._all_corners:
            return (0.0, 0.0, 0.0, 0.0)

        w, h = self._image_size
        mid_x, mid_y = w / 2, h / 2

        quadrants = [[0, 0], [0, 0], [0, 0], [0, 0]]
        corner_count = self.board.chessboard_corners

        for corners in self._all_corners:
            for x, y in corners:
                q = (1 if x > mid_x else 0) + (2 if y > mid_y else 0)
                quadrants[q][0] += 1

        for q in quadrants:
            q[1] = len([c for c in self._all_corners for _ in c])

        coverage = tuple(
            min(1.0, q[0] / max(1, corner_count)) for q in quadrants
        )
        return (coverage[0], coverage[1], coverage[2], coverage[3])

    def get_coverage_quality(self) -> Tuple[str, str]:
        """
        Assess the quality of spatial coverage.

        Returns (level, message) where level is 'good', 'warning', or 'poor'.
        """
        coverage = self.get_coverage()
        min_coverage = min(coverage)
        avg_coverage = sum(coverage) / 4

        if min_coverage < 0.2:
            return (
                "poor",
                "Poor coverage: move card to all corners of the view",
            )
        elif min_coverage < 0.4 or avg_coverage < 0.5:
            return (
                "warning",
                "Limited coverage: ensure card reaches image edges",
            )
        else:
            return ("good", "Good coverage")

    def calibration_status(self) -> Tuple[bool, str]:
        if self._frame_count < self.MIN_FRAMES:
            return (
                False,
                f"Need at least {self.MIN_FRAMES} frames "
                f"(have {self._frame_count})",
            )

        total_unique_corners = len(
            {corner for ids in self._all_ids for corner in ids}
        )

        if total_unique_corners < self.MIN_UNIQUE_CORNERS:
            return (
                False,
                f"Need {self.MIN_UNIQUE_CORNERS}+ unique corners "
                f"(have {total_unique_corners}). "
                f"Move card to different positions.",
            )

        coverage_level, coverage_msg = self.get_coverage_quality()
        if coverage_level == "poor":
            return (True, coverage_msg)
        elif coverage_level == "warning":
            return (True, f"Ready ({coverage_msg})")

        return True, "Ready to calibrate"

    def calibrate(
        self, image_size: Tuple[int, int]
    ) -> Optional[CalibrationResult]:
        can_calib, reason = self.calibration_status()
        if not can_calib:
            logger.error(f"Calibration not ready: {reason}")
            return None

        board_obj = self.board.board
        if board_obj is None:
            logger.error("Board not initialized")
            return None

        all_object_points = board_obj.getChessboardCorners()

        object_points_list = []
        image_points_list = []

        for corners, ids in zip(self._all_corners, self._all_ids):
            corners_array = np.array(corners, dtype=np.float32)
            ids_array = np.array(ids, dtype=np.int32)

            obj_pts = all_object_points[cast(Any, ids_array)]
            object_points_list.append(obj_pts)
            image_points_list.append(corners_array)

        try:
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
                cv2.calibrateCamera(
                    objectPoints=object_points_list,
                    imagePoints=image_points_list,
                    imageSize=image_size,
                    cameraMatrix=cast(Any, None),
                    distCoeffs=cast(Any, None),
                )
            )
        except cv2.error as e:
            logger.error(f"Calibration failed: {e}")
            return None

        reprojection_errors = []
        for i, (corners, rvec, tvec) in enumerate(
            zip(image_points_list, rvecs, tvecs)
        ):
            if corners is None or len(corners) == 0:
                continue

            obj_pts = object_points_list[i]

            projected, _ = cv2.projectPoints(
                obj_pts, rvec, tvec, camera_matrix, dist_coeffs
            )

            projected = projected.reshape(-1, 2)
            error = cv2.norm(corners, projected, cv2.NORM_L2) / len(corners)
            reprojection_errors.append(float(error))

        result = CalibrationResult(
            camera_matrix=camera_matrix,
            distortion_coeffs=dist_coeffs.flatten(),
            rms_error=rms,
            image_size=image_size,
            num_frames_used=self._frame_count,
            reprojection_errors=reprojection_errors,
        )

        logger.info(
            f"Calibration complete: RMS={rms:.4f}, frames={self._frame_count}"
        )

        return result
