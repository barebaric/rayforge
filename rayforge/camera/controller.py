import sys
import time
import threading
import multiprocessing as mp
from typing import Optional, List, Tuple, TYPE_CHECKING
import cv2
import numpy as np
import logging
from blinker import Signal
from ..shared.util.glib import idle_add
from .models.camera import Camera, Pos

if TYPE_CHECKING:
    from gi.repository import GdkPixbuf

logger = logging.getLogger(__name__)


def _probe_camera_device(args):
    """Probe a single camera device. Runs in subprocess."""
    device_id, backend = args
    try:
        cap = cv2.VideoCapture(device_id, backend)
        if cap.isOpened():
            cap.release()
            return device_id
        if cap:
            cap.release()
    except Exception:
        pass
    return None


def _scan_cameras_in_subprocess() -> List[str]:
    """Scan for cameras in a separate process to isolate crashes."""
    if sys.platform.startswith("linux"):
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    elif sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    devices = []
    work = [(i, b) for i in range(10) for b in backends]

    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=1) as pool:
            async_result = pool.map_async(_probe_camera_device, work)
            results = async_result.get(timeout=30)
            for r in results:
                if r is not None and str(r) not in devices:
                    devices.append(str(r))
    except Exception as e:
        logger.warning(f"Subprocess camera scan failed: {e}")
        return _scan_cameras_fallback()

    return devices


def _scan_cameras_fallback() -> List[str]:
    """Fallback camera scan if subprocess fails."""
    devices = []
    if sys.platform.startswith("linux"):
        backends = [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "default")]
    elif sys.platform == "win32":
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "default")]
    else:
        backends = [(cv2.CAP_ANY, "default")]

    for i in range(10):
        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    devices.append(str(i))
                    cap.release()
                    break
                if cap:
                    cap.release()
            except Exception as e:
                logger.debug(f"Error probing camera {i}: {e}")

    return devices


def get_backends_for_platform():
    """Return list of (backend_constant, name) tuples for current platform."""
    if sys.platform.startswith("linux"):
        return [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "default"),
        ]
    if sys.platform == "win32":
        return [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MediaFoundation"),
            (cv2.CAP_ANY, "default"),
        ]
    return [(cv2.CAP_ANY, "default")]


def try_open_camera(device_id: int, backend: int, backend_name: str):
    """Try to open camera with specific backend. Returns cap or None."""
    logger.debug(f"Opening camera {device_id} with {backend_name} backend")
    cap = cv2.VideoCapture(device_id, backend)
    if cap.isOpened():
        logger.info(f"Camera {device_id} opened with {backend_name} backend")
        return cap
    if cap:
        cap.release()
    return None


class VideoCaptureDevice:
    """Context manager for safely opening and releasing camera devices."""

    MAX_OPEN_RETRIES = 3
    RETRY_DELAY = 0.5

    def __init__(self, device_id):
        self.device_id = device_id
        self.cap = None
        self._backend_used = None

    def __enter__(self):
        device_id_int = self._parse_device_id()
        logger.debug(f"Opening camera {device_id_int} on {sys.platform}")

        backends = get_backends_for_platform()
        last_error = None

        for backend, name in backends:
            cap = self._try_backend(device_id_int, backend, name)
            if cap:
                return cap
            last_error = self._retry_backend(
                device_id_int, backend, name, last_error
            )

        self._raise_open_error(backends, last_error)

    def _parse_device_id(self):
        if isinstance(self.device_id, str) and self.device_id.isdigit():
            return int(self.device_id)
        return self.device_id

    def _try_backend(self, device_id_int, backend, name):
        for attempt in range(self.MAX_OPEN_RETRIES):
            try:
                cap = try_open_camera(device_id_int, backend, name)
                if cap:
                    self._backend_used = name
                    self.cap = cap
                    return cap
            except cv2.error as e:
                logger.warning(
                    f"OpenCV error camera {device_id_int} {name} "
                    f"(attempt {attempt + 1}): {e}"
                )
            except Exception as e:
                logger.warning(
                    f"Error camera {device_id_int} {name} "
                    f"(attempt {attempt + 1}): {e}"
                )

            if attempt < self.MAX_OPEN_RETRIES - 1:
                time.sleep(self.RETRY_DELAY)
        return None

    def _retry_backend(self, device_id_int, backend, name, last_error):
        return last_error

    def _raise_open_error(self, backends, last_error):
        names = [b[1] for b in backends]
        msg = (
            f"Cannot open camera {self.device_id}. "
            f"Tried: {names}. Last error: {last_error}"
        )
        logger.error(msg)
        raise IOError(msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is None:
            return
        try:
            if self.cap.isOpened():
                logger.debug(
                    f"Releasing camera {self.device_id} "
                    f"(backend: {self._backend_used})"
                )
                self.cap.release()
        except Exception as e:
            logger.warning(f"Error releasing camera {self.device_id}: {e}")
        finally:
            self.cap = None


class CameraController:
    """Manages camera capture and provides image data."""

    MAX_CONSECUTIVE_FAILURES = 10
    FRAME_READ_TIMEOUT = 1 / 30
    RECONNECT_DELAY = 2.0

    def __init__(self, config: Camera):
        self.config = config
        self._image_data: Optional[np.ndarray] = None
        self._active_subscribers: int = 0
        self._capture_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._settings_dirty: bool = True  # Flag to re-apply settings
        self._consecutive_failures: int = 0

        # Signals
        self.image_captured = Signal()

        self.config.changed.connect(self._on_config_changed)
        self.config.settings_changed.connect(self._on_config_changed)

    def _on_config_changed(self, sender):
        """Reacts to changes in the data model."""
        self._settings_dirty = True
        if self.config.enabled and self._active_subscribers > 0:
            self._start_capture_stream()
        elif not self.config.enabled:
            # Also stop if it's disabled, regardless of subscribers
            self._stop_capture_stream()

    @staticmethod
    def list_available_devices() -> List[str]:
        """
        Lists available camera device IDs.
        Returns a list of strings, where each string is a device ID.
        """
        logger.debug("Scanning for camera devices...")
        devices = []
        backends = get_backends_for_platform()

        for i in range(10):
            for backend, name in backends:
                try:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        devices.append(str(i))
                        cap.release()
                        logger.debug(f"Found camera {i} via {name}")
                        break
                except cv2.error as e:
                    logger.debug(f"OpenCV error camera {i} {name}: {e}")
                except Exception as e:
                    logger.debug(f"Error camera {i}: {e}")

        logger.info(f"Available cameras: {devices}")
        return devices

    @property
    def image_data(self) -> Optional[np.ndarray]:
        return self._image_data

    @property
    def pixbuf(self) -> Optional["GdkPixbuf.Pixbuf"]:
        # Import the UI library ONLY when this method is actually called.
        from gi.repository import GLib, GdkPixbuf

        if self._image_data is None:
            return None

        height, width, channels = self._image_data.shape
        if channels == 3:
            # OpenCV uses BGR, GdkPixbuf expects RGB
            np_array = cv2.cvtColor(self._image_data, cv2.COLOR_BGR2RGB)
            has_alpha = False
        elif channels == 4:
            np_array = self._image_data
            has_alpha = True
        else:
            return None

        # Ensure the array is contiguous
        np_array = np.ascontiguousarray(np_array)

        # Create GBytes from the numpy array
        pixels = GLib.Bytes.new(np_array.tobytes())

        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            pixels,
            GdkPixbuf.Colorspace.RGB,
            has_alpha,
            8,  # bits per sample
            width,
            height,
            width * channels,  # rowstride
        )
        return pixbuf

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._image_data is None:
            return 640, 480
        height, width, _ = self._image_data.shape
        return width, height

    @property
    def aspect(self) -> float:
        return self.resolution[1] / self.resolution[0]

    def subscribe(self):
        """
        Registers a subscriber to the camera's image stream.

        The stream will start if this is the first subscriber and the camera
        is enabled.
        """
        self._active_subscribers += 1
        logger.debug(
            f"Camera {self.config.name} subscribed "
            f"(count: {self._active_subscribers})"
        )
        if self._active_subscribers > 0 and self.config.enabled:
            self._start_capture_stream()

    def unsubscribe(self):
        """
        Unregisters a subscriber.

        The stream will stop if this was the last active subscriber.
        """
        if self._active_subscribers > 0:
            self._active_subscribers -= 1
        logger.debug(
            f"Camera {self.config.name} unsubscribed "
            f"(count: {self._active_subscribers})"
        )
        if self._active_subscribers == 0:
            self._stop_capture_stream()

    def _compute_homography(self, image_height: int) -> np.ndarray:
        """
        Compute the homography matrix from corresponding points.

        Args:
            image_height: The height of the image in pixels.

        Returns:
            3x3 homography matrix mapping world to image coordinates
        """
        if self.config.image_to_world is None:
            raise ValueError("Corresponding points are not set")

        image_points_raw, world_points = self.config.image_to_world

        # Invert y-coordinates of image_points to align with world coordinates
        # (y-up)
        image_points_y_up = [
            (p[0], image_height - p[1]) for p in image_points_raw
        ]

        if (
            self.config._camera_matrix is not None
            and self.config._dist_coeffs is not None
        ):
            # Undistort image points if calibration parameters are available
            image_points_y_up = cv2.undistortPoints(
                np.array(image_points_y_up, dtype=np.float32),
                self.config._camera_matrix,
                self.config._dist_coeffs,
            )
            image_points_y_up = image_points_y_up.reshape(-1, 2)

        # Compute homography (world to image_y_up)
        H, _ = cv2.findHomography(
            np.array(world_points, dtype=np.float32),
            np.array(image_points_y_up, dtype=np.float32),
        )
        return H

    def get_work_surface_image(
        self, output_size: Tuple[int, int], physical_area: Tuple[Pos, Pos]
    ) -> Optional[np.ndarray]:
        """
        Get an aligned image of the specified physical area.

        Args:
            output_size: Desired output image size (width, height) in pixels
            physical_area: Physical area ((x_min, y_min), (x_max, y_max))
              to capture in real-world coordinates

        Returns:
            Aligned image as a NumPy array
        """
        if self.config.image_to_world is None:
            raise ValueError(
                "Corresponding points must be set before getting the"
                " work surface image"
            )
        if self._image_data is None:
            logger.warning("No image data available.")
            return None

        # Capture raw image
        raw_image = self._image_data

        # Undistort if calibration parameters are set
        if (
            self.config._camera_matrix is not None
            and self.config._dist_coeffs is not None
        ):
            try:
                undistorted_image = cv2.undistort(
                    raw_image,
                    self.config._camera_matrix,
                    self.config._dist_coeffs,
                )
            except cv2.error as e:
                logger.warning(f"Failed to undistort image: {e}")
                undistorted_image = raw_image
        else:
            undistorted_image = raw_image

        # Compute homography (world to image)
        try:
            H = self._compute_homography(raw_image.shape[0])
        except ValueError as e:
            logger.error(f"Cannot compute homography: {e}")
            return None

        # Define transformation from output pixels to world coordinates
        (x_min, y_min), (x_max, y_max) = physical_area
        width_px, height_px = output_size

        # Calculate the actual physical width and height of the area being
        # viewed
        physical_width = x_max - x_min
        physical_height = y_max - y_min

        # Calculate the scaling factors from output pixels to world coordinates
        scale_x = physical_width / width_px
        scale_y = -physical_height / height_px

        offset_x = x_min
        offset_y = y_max
        T = np.array(
            [
                [scale_x, 0, offset_x],
                [0, scale_y, offset_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Overall transformation: output pixels -> world -> image
        M = H @ T

        # Apply perspective warp
        # Log transformed corner points
        world_corners = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.float32,
        )
        world_corners_h = np.hstack((world_corners, np.ones((4, 1)))).T
        transformed_corners = M @ world_corners_h
        transformed_corners = transformed_corners[:2] / transformed_corners[2]
        transformed_corners = transformed_corners.T

        try:
            aligned_image = cv2.warpPerspective(
                undistorted_image, np.linalg.inv(M), output_size
            )
            return aligned_image
        except cv2.error as e:
            logger.error(f"Failed to apply perspective warp: {e}")
            return None

    def _apply_settings(self, cap: cv2.VideoCapture):
        """Applies the current settings to the VideoCapture object."""
        try:
            if self.config.white_balance is None:
                cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance
            else:
                cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
                cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.config.white_balance)
            cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)

            self._settings_dirty = False
            logger.debug("Applied camera hardware settings.")
        except Exception as e:
            # We log as a warning because the stream may still work
            logger.warning(f"Could not apply one or more camera settings: {e}")

    def _read_frame(self, cap) -> bool:
        """Read frame from cap. Returns True on success."""
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to capture frame from camera.")
                self._image_data = None
                return False
            self._image_data = frame
            # Emit the signal in a GLib-safe way
            idle_add(self.image_captured.send, self)
            return True
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False

    def _handle_frame_failure(self):
        """Handle a failed frame read. Returns True if should reconnect."""
        self._consecutive_failures += 1
        self._image_data = None
        logger.warning(
            f"Frame failure {self._consecutive_failures}/"
            f"{self.MAX_CONSECUTIVE_FAILURES} for {self.config.name}"
        )
        return self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES

    def _capture_frames_from_device(self, cap):
        """Capture frames from an opened device. Returns when should stop."""
        self._settings_dirty = True
        self._consecutive_failures = 0

        while self._running and cap is not None:
            if self._settings_dirty:
                self._apply_settings(cap)

            if self._read_frame(cap):
                self._consecutive_failures = 0
            elif self._handle_frame_failure():
                logger.error(
                    f"Too many failures for {self.config.name}, "
                    "reconnecting..."
                )
                return

            time.sleep(self.FRAME_READ_TIMEOUT)

    def _capture_loop(self):
        """
        Internal method to continuously capture images from the camera.
        Runs in a separate thread.
        """
        logger.info(
            f"Capture loop starting for {self.config.name} "
            f"(device: {self.config.device_id})"
        )

        while self._running:
            try:
                # Open the device ONCE
                with VideoCaptureDevice(self.config.device_id) as cap:
                    if cap is None:
                        raise IOError("VideoCapture returned None")
                    self._capture_frames_from_device(cap)
            except IOError as e:
                logger.error(f"IO error for {self.config.name}: {e}")
            except cv2.error as e:
                logger.error(f"OpenCV error for {self.config.name}: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error for {self.config.name}: {e}",
                    exc_info=True,
                )

            if self._running:
                logger.info(
                    f"Waiting {self.RECONNECT_DELAY}s before "
                    f"reconnecting {self.config.name}..."
                )
                time.sleep(self.RECONNECT_DELAY)

        logger.debug(
            f"Camera capture loop stopped for camera {self.config.name}."
        )

    def _start_capture_stream(self):
        """
        Starts a continuous image capture stream in a separate thread.
        """
        if self._running:
            logger.debug(
                f"Capture stream already running for camera {self.config.name}"
            )
            return

        logger.debug(f"Starting capture stream for camera {self.config.name}.")
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop)
        self._capture_thread.daemon = True  # Allow the main program to exit
        self._capture_thread.start()

    def _stop_capture_stream(self):
        """
        Stops the continuous image capture stream.
        """
        if not self._running:
            logger.debug(
                f"Capture stream not running for camera {self.config.name}."
            )
            return

        logger.debug(f"Stopping capture stream for camera {self.config.name}.")
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)  # Wait for thread to finish
            if self._capture_thread.is_alive():
                logger.warning("Capture thread did not terminate gracefully.")
        self._capture_thread = None

    def capture_image(self):
        """
        Captures a single image from this camera device.
        """
        try:
            with VideoCaptureDevice(self.config.device_id) as cap:
                if cap is None:
                    logger.error(
                        f"Cannot capture: VideoCapture is None for "
                        f"{self.config.device_id}"
                    )
                    self._image_data = None
                    return
                # Apply settings before capturing the single frame
                self._apply_settings(cap)
                self._read_frame(cap)
        except IOError as e:
            logger.error(f"IO error capturing image: {e}")
            self._image_data = None
        except cv2.error as e:
            logger.error(f"OpenCV error capturing image: {e}")
            self._image_data = None
        except Exception as e:
            logger.error(
                f"Unexpected error capturing image: {e}", exc_info=True
            )
            self._image_data = None
