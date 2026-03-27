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

    def __init__(self, board: CharucoBoard):
        self.board = board
        self._all_corners: List[List[Tuple[float, float]]] = []
        self._all_ids: List[List[int]] = []
        self._frame_count = 0

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

        return total_unique_corners >= 10

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

        if total_unique_corners < 10:
            return (
                False,
                f"Need 10+ unique corners (have {total_unique_corners}). "
                f"Move card to different positions.",
            )

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
