from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    rms_error: float
    image_size: Tuple[int, int]
    num_frames_used: int
    reprojection_errors: List[float] = field(default_factory=list)
    calibration_date: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.camera_matrix = np.array(self.camera_matrix, dtype=np.float64)
        self.distortion_coeffs = np.array(
            self.distortion_coeffs, dtype=np.float64
        ).flatten()

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    @property
    def k1(self) -> float:
        return float(self.distortion_coeffs[0])

    @property
    def k2(self) -> float:
        return float(self.distortion_coeffs[1])

    @property
    def p1(self) -> float:
        return (
            float(self.distortion_coeffs[2])
            if len(self.distortion_coeffs) > 2
            else 0.0
        )

    @property
    def p2(self) -> float:
        return (
            float(self.distortion_coeffs[3])
            if len(self.distortion_coeffs) > 3
            else 0.0
        )

    @property
    def k3(self) -> float:
        return (
            float(self.distortion_coeffs[4])
            if len(self.distortion_coeffs) > 4
            else 0.0
        )

    @property
    def quality_rating(self) -> str:
        if self.rms_error < 0.5:
            return "excellent"
        elif self.rms_error < 1.0:
            return "good"
        elif self.rms_error < 2.0:
            return "acceptable"
        else:
            return "poor"

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coeffs": self.distortion_coeffs.tolist(),
            "rms_error": self.rms_error,
            "image_size": list(self.image_size),
            "num_frames_used": self.num_frames_used,
            "reprojection_errors": self.reprojection_errors,
            "calibration_date": self.calibration_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationResult":
        return cls(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            distortion_coeffs=np.array(
                data["distortion_coeffs"], dtype=np.float64
            ),
            rms_error=data["rms_error"],
            image_size=tuple(data["image_size"]),
            num_frames_used=data["num_frames_used"],
            reprojection_errors=data.get("reprojection_errors", []),
            calibration_date=datetime.fromisoformat(data["calibration_date"]),
        )

    def get_undistort_maps(
        self, image_size: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        size = image_size or self.image_size
        if size[0] <= 0 or size[1] <= 0:
            return None

        try:
            new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.distortion_coeffs,
                size,
                1,
                size,
            )
            R = np.eye(3, dtype=np.float64)
            map1, map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion_coeffs,
                R,
                new_cam_mtx,
                size,
                cv2.CV_16SC2,
            )
            return map1, map2
        except Exception:
            return None
