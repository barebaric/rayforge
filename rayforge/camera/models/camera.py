import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Sequence, Any
import numpy as np
import logging
from blinker import Signal


logger = logging.getLogger(__name__)
Pos = Tuple[float, float]
PointList = Sequence[Pos]


class Camera:
    """A pure data model representing the configuration of a camera."""

    def __init__(self, name: str, device_id: str):
        self._name: str = name
        self._device_id: str = device_id
        self._enabled: bool = False
        # None indicates auto white balance, float for manual Kelvin
        self._white_balance: Optional[float] = None
        self._contrast: float = 50.0
        self._brightness: float = 0.0  # Default brightness (0 = no change)
        self._transparency: float = 0.2

        # Noise Reduction (0.0 to 1.0)
        # 0.0 = No denoise, 1.0 = Max smoothing (high latency/ghosting)
        self._denoise: float = 0.0

        self._prefer_yuyv: bool = False

        # Lens calibration parameters
        # Distortion coefficients: k1, k2, p1, p2, k3 (OpenCV order)
        self._distortion_k1: float = 0.0
        self._distortion_k2: float = 0.0
        self._distortion_p1: float = 0.0
        self._distortion_p2: float = 0.0
        self._distortion_k3: float = 0.0

        # Camera matrix parameters (needed for proper undistortion)
        self._camera_matrix_fx: Optional[float] = None
        self._camera_matrix_fy: Optional[float] = None
        self._camera_matrix_cx: Optional[float] = None
        self._camera_matrix_cy: Optional[float] = None

        # Calibration metadata
        self._calibration_rms: Optional[float] = None
        self._calibration_date: Optional[datetime] = None
        self._calibration_image_size: Optional[Tuple[int, int]] = None
        self._calibration_frames_used: Optional[int] = None

        # Properties for camera alignment points
        self._image_to_world: Optional[Tuple[PointList, PointList]] = None

        # Signals
        self.changed = Signal()
        self.settings_changed = Signal()
        self.extra: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        if self._name == value:
            return
        logger.debug(f"Camera name changed from '{self._name}' to '{value}'")
        self._name = value
        self.changed.send(self)

    @property
    def device_id(self) -> str:
        return self._device_id

    @device_id.setter
    def device_id(self, value: str):
        if self._device_id == value:
            return
        logger.debug(
            f"Camera device_id changed from '{self._device_id}' to '{value}'"
        )
        self._device_id = value
        self.changed.send(self)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        if self._enabled == value:
            return
        logger.debug(f"Camera enabled changed from {self._enabled} to {value}")
        self._enabled = value
        self.changed.send(self)

    @property
    def white_balance(self) -> Optional[float]:
        return self._white_balance

    @white_balance.setter
    def white_balance(self, value: Optional[float]):
        if value is not None:
            if not isinstance(value, (int, float)):
                raise ValueError("White balance must be a number or None.")
            if not (2500 <= value <= 10000):
                logger.warning(
                    f"White balance value {value} is outside range "
                    "(2500-10000). Clamping to nearest bound."
                )
                value = max(2500, min(value, 10000))
        if self._white_balance == value:
            return
        logger.debug(
            f"Camera white_balance changed from {self._white_balance} to "
            f"{value}"
        )
        self._white_balance = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def contrast(self) -> float:
        return self._contrast

    @contrast.setter
    def contrast(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("Contrast must be a number.")
        if not (0.0 <= value <= 100.0):
            logger.warning(
                f"Contrast value {value} is outside range (0.0-100.0). "
                "Clamping to nearest bound."
            )
            value = max(0.0, min(value, 100.0))
        if self._contrast == value:
            return
        logger.debug(
            f"Camera contrast changed from {self._contrast} to {value}"
        )
        self._contrast = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def brightness(self) -> float:
        return self._brightness

    @brightness.setter
    def brightness(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("Brightness must be a number.")
        if not (-100.0 <= value <= 100.0):
            logger.warning(
                f"Brightness value {value} is outside range (-100.0-100.0). "
                "Clamping to nearest bound."
            )
            value = max(-100.0, min(value, 100.0))
        if self._brightness == value:
            return
        logger.debug(
            f"Camera brightness changed from {self._brightness} to {value}"
        )
        self._brightness = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def transparency(self) -> float:
        return self._transparency

    @transparency.setter
    def transparency(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("Transparency must be a number.")
        if not (0.0 <= value <= 1.0):
            logger.warning(
                f"Transparency value {value} is outside range (0.0-1.0). "
                "Clamping to nearest bound."
            )
            value = max(0.0, min(value, 1.0))
        if self._transparency == value:
            return
        logger.debug(
            f"Camera transparency changed from {self._transparency} to {value}"
        )
        self._transparency = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def denoise(self) -> float:
        """Temporal noise reduction factor (0.0 - 1.0)."""
        return self._denoise

    @denoise.setter
    def denoise(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("Denoise must be a number.")
        # Clamp between 0.0 (off) and 0.95 (extreme smoothing)
        value = max(0.0, min(value, 0.95))
        if self._denoise == value:
            return
        logger.debug(f"Camera denoise changed from {self._denoise} to {value}")
        self._denoise = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def prefer_yuyv(self) -> bool:
        """Whether to prefer YUYV format over MJPEG (fixes green artifacts)."""
        return self._prefer_yuyv

    @prefer_yuyv.setter
    def prefer_yuyv(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("prefer_yuyv must be a boolean.")
        if self._prefer_yuyv == value:
            return
        logger.debug(
            f"Camera prefer_yuyv changed from {self._prefer_yuyv} to {value}"
        )
        self._prefer_yuyv = value
        self.changed.send(self)
        self.settings_changed.send(self)

    # --- Fisheye/Distortion Properties ---

    @property
    def distortion_k1(self) -> float:
        return self._distortion_k1

    @distortion_k1.setter
    def distortion_k1(self, value: float):
        self._distortion_k1 = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def distortion_k2(self) -> float:
        return self._distortion_k2

    @distortion_k2.setter
    def distortion_k2(self, value: float):
        self._distortion_k2 = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def distortion_p1(self) -> float:
        return self._distortion_p1

    @distortion_p1.setter
    def distortion_p1(self, value: float):
        self._distortion_p1 = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def distortion_p2(self) -> float:
        return self._distortion_p2

    @distortion_p2.setter
    def distortion_p2(self, value: float):
        self._distortion_p2 = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def distortion_k3(self) -> float:
        return self._distortion_k3

    @distortion_k3.setter
    def distortion_k3(self, value: float):
        self._distortion_k3 = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def camera_matrix_fx(self) -> Optional[float]:
        return self._camera_matrix_fx

    @camera_matrix_fx.setter
    def camera_matrix_fx(self, value: Optional[float]):
        self._camera_matrix_fx = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def camera_matrix_fy(self) -> Optional[float]:
        return self._camera_matrix_fy

    @camera_matrix_fy.setter
    def camera_matrix_fy(self, value: Optional[float]):
        self._camera_matrix_fy = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def camera_matrix_cx(self) -> Optional[float]:
        return self._camera_matrix_cx

    @camera_matrix_cx.setter
    def camera_matrix_cx(self, value: Optional[float]):
        self._camera_matrix_cx = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def camera_matrix_cy(self) -> Optional[float]:
        return self._camera_matrix_cy

    @camera_matrix_cy.setter
    def camera_matrix_cy(self, value: Optional[float]):
        self._camera_matrix_cy = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def has_calibration(self) -> bool:
        return all(
            v is not None
            for v in [
                self._camera_matrix_fx,
                self._camera_matrix_fy,
                self._camera_matrix_cx,
                self._camera_matrix_cy,
            ]
        )

    def get_camera_matrix(self) -> Optional[np.ndarray]:
        if not self.has_calibration:
            return None
        return np.array(
            [
                [self._camera_matrix_fx, 0, self._camera_matrix_cx],
                [0, self._camera_matrix_fy, self._camera_matrix_cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def get_distortion_coeffs(self) -> np.ndarray:
        return np.array(
            [
                self._distortion_k1,
                self._distortion_k2,
                self._distortion_p1,
                self._distortion_p2,
                self._distortion_k3,
            ],
            dtype=np.float64,
        )

    # ---------------------------------------------

    @property
    def image_to_world(self) -> Optional[Tuple[PointList, PointList]]:
        return self._image_to_world

    @image_to_world.setter
    def image_to_world(self, value: Optional[Tuple[PointList, PointList]]):
        if value is not None:
            if not (isinstance(value, tuple) and len(value) == 2):
                raise ValueError(
                    "Corresponding points must be a tuple of two point lists."
                )
            image_points, world_points = value
            if not (
                isinstance(image_points, Sequence)
                and isinstance(world_points, Sequence)
            ):
                raise ValueError(
                    "Both elements of corresponding points must be sequences."
                )
            if len(image_points) < 4 or len(world_points) < 4:
                raise ValueError(
                    "At least 4 corresponding points are required."
                )
            if len(image_points) != len(world_points):
                raise ValueError(
                    "Image points and world points must have the same number "
                    "of entries."
                )
            for points in [image_points, world_points]:
                for p in points:
                    if not (
                        isinstance(p, tuple)
                        and len(p) == 2
                        and isinstance(p[0], (int, float))
                        and isinstance(p[1], (int, float))
                    ):
                        raise ValueError(
                            "Each point must be a tuple of two floats "
                            "(e.g., (x, y))."
                        )
        if self._image_to_world == value:
            return
        logger.debug(
            f"Camera image_to_world changed from "
            f"{self._image_to_world} to {value}"
        )
        self._image_to_world = value
        self.changed.send(self)
        self.settings_changed.send(self)

    def set_calibration_result(self, result):
        """
        Set camera calibration from a CalibrationResult object.

        Populates both the camera matrix parameters and distortion
        coefficients into the unified storage.

        Args:
            result: CalibrationResult from the calibrator
        """
        from ..calibration.result import CalibrationResult

        if not isinstance(result, CalibrationResult):
            raise TypeError("Expected CalibrationResult object")

        self._camera_matrix_fx = float(result.camera_matrix[0, 0])
        self._camera_matrix_fy = float(result.camera_matrix[1, 1])
        self._camera_matrix_cx = float(result.camera_matrix[0, 2])
        self._camera_matrix_cy = float(result.camera_matrix[1, 2])

        dist = result.distortion_coeffs.flatten()
        self._distortion_k1 = float(dist[0]) if len(dist) > 0 else 0.0
        self._distortion_k2 = float(dist[1]) if len(dist) > 1 else 0.0
        self._distortion_p1 = float(dist[2]) if len(dist) > 2 else 0.0
        self._distortion_p2 = float(dist[3]) if len(dist) > 3 else 0.0
        self._distortion_k3 = float(dist[4]) if len(dist) > 4 else 0.0

        self._calibration_rms = result.rms_error
        self._calibration_date = result.calibration_date
        self._calibration_image_size = result.image_size
        self._calibration_frames_used = result.num_frames_used

        logger.debug(
            f"Camera calibration set from result: RMS={result.rms_error:.4f}"
        )
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def calibration_rms(self) -> Optional[float]:
        return self._calibration_rms

    @property
    def calibration_date(self) -> Optional[datetime]:
        return self._calibration_date

    @property
    def calibration_image_size(self) -> Optional[Tuple[int, int]]:
        return self._calibration_image_size

    @property
    def calibration_frames_used(self) -> Optional[int]:
        return self._calibration_frames_used

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "device_id": self.device_id,
            "enabled": self.enabled,
            "white_balance": self.white_balance,
            "contrast": self.contrast,
            "brightness": self.brightness,
            "transparency": self.transparency,
            "denoise": self.denoise,
            "prefer_yuyv": self.prefer_yuyv,
            "distortion_k1": self.distortion_k1,
            "distortion_k2": self.distortion_k2,
            "distortion_p1": self.distortion_p1,
            "distortion_p2": self.distortion_p2,
            "distortion_k3": self.distortion_k3,
        }
        if self.image_to_world is not None:
            image_points, world_points = self.image_to_world
            data["image_to_world"] = [
                {
                    "image": f"{img[0]}, {img[1]}",
                    "world": f"{wld[0]}, {wld[1]}",
                }
                for img, wld in zip(image_points, world_points)
            ]
        else:
            data["image_to_world"] = None

        if self.has_calibration:
            data["camera_matrix_fx"] = self._camera_matrix_fx
            data["camera_matrix_fy"] = self._camera_matrix_fy
            data["camera_matrix_cx"] = self._camera_matrix_cx
            data["camera_matrix_cy"] = self._camera_matrix_cy
            data["calibration_rms"] = self._calibration_rms
            data["calibration_date"] = (
                self._calibration_date.isoformat()
                if self._calibration_date
                else None
            )
            data["calibration_image_size"] = (
                list(self._calibration_image_size)
                if self._calibration_image_size
                else None
            )
            data["calibration_frames_used"] = self._calibration_frames_used

        data.update(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Camera":
        known_keys = {
            "name",
            "device_id",
            "enabled",
            "white_balance",
            "contrast",
            "brightness",
            "transparency",
            "denoise",
            "prefer_yuyv",
            "image_to_world",
            "distortion_k1",
            "distortion_k2",
            "distortion_p1",
            "distortion_p2",
            "distortion_k3",
            "camera_matrix_fx",
            "camera_matrix_fy",
            "camera_matrix_cx",
            "camera_matrix_cy",
            "calibration_rms",
            "calibration_date",
            "calibration_image_size",
            "calibration_frames_used",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        camera = cls(data["name"], data["device_id"])
        camera.enabled = data.get("enabled", camera.enabled)
        camera.white_balance = data.get("white_balance", None)
        camera.contrast = data.get("contrast", camera.contrast)
        camera.brightness = data.get("brightness", camera.brightness)
        camera.transparency = data.get("transparency", camera.transparency)
        camera.denoise = data.get("denoise", 0.0)
        camera.prefer_yuyv = data.get("prefer_yuyv", False)

        camera.distortion_k1 = data.get("distortion_k1", 0.0)
        camera.distortion_k2 = data.get("distortion_k2", 0.0)
        camera.distortion_p1 = data.get("distortion_p1", 0.0)
        camera.distortion_p2 = data.get("distortion_p2", 0.0)
        camera.distortion_k3 = data.get("distortion_k3", 0.0)

        image_to_world_data = data.get("image_to_world")
        if image_to_world_data is not None:
            image_points = []
            world_points = []
            for entry in image_to_world_data:
                image_str = entry["image"].split(",")
                world_str = entry["world"].split(",")
                image_points.append(
                    (float(image_str[0].strip()), float(image_str[1].strip()))
                )
                world_points.append(
                    (float(world_str[0].strip()), float(world_str[1].strip()))
                )
            camera.image_to_world = (image_points, world_points)
        else:
            camera.image_to_world = None

        camera._camera_matrix_fx = data.get("camera_matrix_fx")
        camera._camera_matrix_fy = data.get("camera_matrix_fy")
        camera._camera_matrix_cx = data.get("camera_matrix_cx")
        camera._camera_matrix_cy = data.get("camera_matrix_cy")
        camera._calibration_rms = data.get("calibration_rms")
        if data.get("calibration_date"):
            camera._calibration_date = datetime.fromisoformat(
                data["calibration_date"]
            )
        if data.get("calibration_image_size"):
            camera._calibration_image_size = tuple(
                data["calibration_image_size"]
            )
        camera._calibration_frames_used = data.get("calibration_frames_used")

        camera.extra = extra
        return camera

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.from_dict(data)
