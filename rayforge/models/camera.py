import json
from typing import Optional, List, Tuple, Dict, Any
import cv2
import numpy as np
import logging
from blinker import Signal


logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, name: str, device_id: str):
        self._name: str = name
        self._device_id: str = device_id
        self._enabled: bool = False
        self._image_data: Optional[np.ndarray] = None
        self._width_mm: float = 0.0
        self._height_mm: float = 0.0
        self._transform_matrix: List[float] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.changed = Signal()

    @staticmethod
    def list_available_devices() -> List[str]:
        """
        Lists available camera device IDs.
        Returns a list of strings, where each string is a device ID.
        """
        available_devices = []
        # Try device IDs from 0 up to a reasonable number (e.g., 10)
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_devices.append(str(i))
                cap.release()
            else:
                # If a device ID fails, subsequent ones might also fail,
                # so we can stop checking after a few consecutive failures.
                # For simplicity, we'll just continue for now.
                pass
        logger.info("Found available camera devices: %s", available_devices)
        return available_devices

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        if self._name == value:
            return
        logger.debug(
            "Camera name changed from '%s' to '%s'", self._name, value
        )
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
            "Camera device_id changed from '%s' to '%s'",
            self._device_id, value
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
        logger.debug(
            "Camera enabled changed from %s to %s", self._enabled, value
        )
        self._enabled = value
        self.changed.send(self)

    @property
    def image_data(self) -> Optional[np.ndarray]:
        return self._image_data

    @property
    def width_mm(self) -> float:
        return self._width_mm

    @width_mm.setter
    def width_mm(self, value: float):
        if self._width_mm == value:
            return
        logger.debug(
            "Camera width_mm changed from %f to %f", self._width_mm, value
        )
        self._width_mm = value
        self.changed.send(self)

    @property
    def height_mm(self) -> float:
        return self._height_mm

    @height_mm.setter
    def height_mm(self, value: float):
        if self._height_mm == value:
            return
        logger.debug(
            "Camera height_mm changed from %f to %f", self._height_mm, value
        )
        self._height_mm = value
        self.changed.send(self)

    @property
    def transform_matrix(self) -> List[float]:
        return self._transform_matrix

    @transform_matrix.setter
    def transform_matrix(self, value: List[float]):
        if not isinstance(value, list) or len(value) != 6:
            raise ValueError("Transform matrix must be a list of 6 floats.")
        if self._transform_matrix == value:
            return
        logger.debug(
            "Camera transform_matrix changed from %s to %s",
            self._transform_matrix, value
        )
        self._transform_matrix = value
        self.changed.send(self)

    def capture_image(self):
        """
        Captures an image from this camera device.
        Updates image_data, width_mm, and height_mm attributes.
        """
        cap = None
        try:
            device_id = self.device_id
            if device_id.isdigit():
                device_id = int(device_id)

            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                raise IOError(
                    f"Cannot open camera with device ID: {device_id}"
                )

            ret, frame = cap.read()
            if not ret:
                raise IOError("Failed to capture image from camera.")

            height, width, _ = frame.shape
            self.width_mm = float(width)
            self.height_mm = float(height)
            self._image_data = frame
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            self._image_data = None
            self.width_mm = 0.0
            self.height_mm = 0.0
        finally:
            if cap is not None and cap.isOpened():
                cap.release()

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "device_id": self.device_id,
            "enabled": self.enabled,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "transform_matrix": self.transform_matrix,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Camera':
        camera = cls(data["name"], data["device_id"])
        camera.enabled = data.get("enabled", camera.enabled)
        camera.width_mm = data.get("width_mm", camera.width_mm)
        camera.height_mm = data.get("height_mm", camera.height_mm)
        camera.transform_matrix = data.get("transform_matrix",
                                           camera.transform_matrix)
        return camera

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.from_dict(data)

    @staticmethod
    def apply_affine_transform(image_data: np.ndarray,
                               transform_matrix: List[float],
                               output_size: Tuple[int, int]):
        """
        Applies a 2x3 affine transformation to an image.

        Args:
            image_data: The input image as a NumPy array.
            transform_matrix: A list of 6 floats representing the 2x3 affine
                              matrix. [M11, M12, M13, M21, M22, M23]
            output_size: A tuple (width, height) for the output image
                         dimensions.

        Returns:
            The transformed image as a NumPy array.
        """
        if image_data is None:
            return None

        if not isinstance(transform_matrix, list) or \
           len(transform_matrix) != 6:
            raise ValueError("Transform matrix must be a list of 6 floats.")

        M = np.array(transform_matrix).reshape((2, 3))

        transformed_image = cv2.warpAffine(image_data, M, output_size)
        return transformed_image
