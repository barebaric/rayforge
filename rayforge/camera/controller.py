import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from blinker import Signal

from ..image.util.srgb import resize_linear_nd
from ..shared.util.glib import idle_add
from .models.camera import Camera
from .source import (
    create_camera_source,
    list_local_device_ids,
)

if TYPE_CHECKING:
    from gi.repository import GdkPixbuf

    from .source import CameraSource

logger = logging.getLogger(__name__)


def __getattr__(name: str):
    if name == "_to_videocapture_arg":
        from .source import _to_videocapture_arg as func

        return func
    raise AttributeError(name)


# A comprehensive list of standard resolutions to populate the UI dropdown.
COMMON_RESOLUTIONS = [
    (320, 240),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 960),
    (1600, 1200),
    (1920, 1080),
    (2048, 1536),
    (2560, 1440),
    (2592, 1944),
    (3264, 2448),
    (3840, 2160),
    (4096, 2160),
    (5120, 3840),
    (6144, 3456),
    (7680, 4320),
]
Pos = tuple[float, float]


class CameraController:
    """Manages camera capture and provides image data."""

    MAX_CONSECUTIVE_FAILURES = 10
    FRAME_READ_TIMEOUT = 1 / 30
    RECONNECT_DELAY = 2.0
    STOP_JOIN_TIMEOUT = 2.0

    def __init__(self, config: Camera):
        self.config = config
        self._image_data: np.ndarray | None = None
        self._raw_image_data: np.ndarray | None = None
        # For Temporal Smoothing
        self._accumulator: np.ndarray | None = None
        self._active_subscribers: int = 0
        self._capture_thread: threading.Thread | None = None
        self._running: bool = False
        self._settings_dirty: bool = True  # Flag to re-apply settings
        self._consecutive_failures: int = 0
        self._last_frame_warning_log_time: float | None = None
        self._last_reconnect_warning_log_time: float | None = None
        self._active_source: CameraSource | None = None
        self._last_source_key = self._source_key()
        self._disposed: bool = False

        # Stream lifecycle: _lifecycle_lock serializes start/stop so that
        # at most one capture thread can ever own the device at a time
        # (concurrent start/stop calls -- e.g. a config-change signal
        # racing with an explicit subscribe/unsubscribe -- is what caused
        # the historic double-open crash on device swap). _stop_event
        # makes the open-retry loop and the inter-frame/reconnect sleeps
        # interruptible so a stop request is honored promptly instead of
        # after a multi-second timeout. _thread_stuck is set if a capture
        # thread ever refuses to terminate; once set, this controller
        # permanently refuses to start a new stream rather than risk
        # opening the same device twice.
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread_stuck: bool = False

        # Protects the frame buffers (_image_data/_raw_image_data/
        # _accumulator), which are written by the capture thread and read
        # from the UI thread, so readers never observe a torn state (e.g.
        # a fresh _image_data paired with a stale _raw_image_data) during
        # a reconnect or source swap.
        self._frame_lock = threading.Lock()

        # We no longer probe hardware directly because V4L2 and DirectShow
        # drivers often crash or drop buffers when aggressively queried.
        self._available_resolutions: list[tuple[int, int]] = COMMON_RESOLUTIONS
        self._resolutions_probed: bool = True

        # Signals
        self.image_captured = Signal()
        self.resolutions_probed = Signal()

        self.config.changed.connect(self._on_config_changed)
        self.config.settings_changed.connect(self._on_config_changed)

    def _source_key(self) -> tuple:
        """A tuple identifying which physical source is being captured.

        Used to detect when the camera's source (e.g. its local device
        ID) changed underneath a running capture stream, which requires
        closing the old device and opening the new one rather than just
        continuing to read from the already-open capture.
        """
        return (self.config.source_type, self.config.device_id)

    def _on_config_changed(self, sender):
        """Reacts to changes in the data model."""
        if self._disposed:
            return
        self._settings_dirty = True
        with self._frame_lock:
            self._accumulator = None  # Reset smoothing if settings change
        new_source_key = self._source_key()
        source_changed = new_source_key != self._last_source_key
        self._last_source_key = new_source_key
        if self.config.enabled and self._active_subscribers > 0:
            if source_changed and self._running:
                # The physical source changed (e.g. a different local
                # camera was selected). Close the old device before
                # opening the new one instead of leaving it running.
                self._stop_capture_stream()
            self._start_capture_stream()
        elif not self.config.enabled:
            # Also stop if it's disabled, regardless of subscribers
            self._stop_capture_stream()

    @staticmethod
    def list_available_devices() -> list[str]:
        """
        Lists available camera device IDs.
        Returns a list of strings, where each string is a device ID.
        On Linux, prefers persistent /dev/v4l/by-id/ paths.
        """
        logger.debug("Scanning for local camera devices...")
        devices = list_local_device_ids()
        logger.info("Available cameras: %s", devices)
        return devices

    @property
    def image_data(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._image_data

    @property
    def has_active_source(self) -> bool:
        return self._active_source is not None

    @property
    def raw_image_data(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._raw_image_data

    @property
    def pixbuf(self) -> Optional["GdkPixbuf.Pixbuf"]:
        # Import the UI library ONLY when this method is actually called.
        from gi.repository import GdkPixbuf, GLib

        # Snapshot under the lock; the capture thread always replaces the
        # buffer with a fresh array rather than mutating it in place, so
        # the snapshot itself is safe to use without the lock held.
        with self._frame_lock:
            image = self._image_data

        if image is None:
            return None

        height, width, channels = image.shape
        if channels == 3:
            # OpenCV uses BGR, GdkPixbuf expects RGB
            np_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            has_alpha = False
        elif channels == 4:
            np_array = image
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
    def resolution(self) -> tuple[int, int]:
        with self._frame_lock:
            image = self._image_data
        if image is None:
            return 640, 480
        height, width, _ = image.shape
        return width, height

    @property
    def available_resolutions(self) -> list[tuple[int, int]]:
        return self._available_resolutions

    @property
    def aspect(self) -> float:
        return self.resolution[1] / self.resolution[0]

    def subscribe(self):
        """
        Registers a subscriber to the camera's image stream.

        The stream will start if this is the first subscriber and the camera
        is enabled.
        """
        if self._disposed:
            logger.warning(
                f"Ignoring subscribe() on disposed controller for "
                f"{self.config.name}"
            )
            return
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
        else:
            logger.warning(
                f"Unbalanced unsubscribe() for camera {self.config.name}: "
                f"subscriber count is already zero"
            )
        logger.debug(
            f"Camera {self.config.name} unsubscribed "
            f"(count: {self._active_subscribers})"
        )
        if self._active_subscribers == 0:
            self._stop_capture_stream()

    def dispose(self) -> None:
        """Final teardown: stop the stream and drop model signal links.

        Must be called exactly once, when this controller is permanently
        retired (e.g. its camera was removed from the machine). Without
        disconnecting from config.changed/settings_changed, a destroyed
        controller stays wired to the model and a later, unrelated edit
        reaching it could resurrect its capture thread.
        """
        if self._disposed:
            return
        self._disposed = True
        self._stop_capture_stream()
        self.config.changed.disconnect(self._on_config_changed)
        self.config.settings_changed.disconnect(self._on_config_changed)

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

        # Compute homography (world to image_y_up)
        H, _ = cv2.findHomography(
            np.array(world_points, dtype=np.float32),
            np.array(image_points_y_up, dtype=np.float32),
        )
        return H

    def get_work_surface_image(
        self,
        output_size: tuple[int, int],
        physical_area: tuple[Pos, Pos],
    ) -> np.ndarray | None:
        """
        Get an image aligned to world coordinates.

        For cameras with perspective calibration (image_to_world), this uses
        homography transformation. For cameras without calibration, this
        applies a simple resize to match the output size.

        Both the canvas and stock detection addon should use this method
        to ensure consistent image transformation.

        The returned image has pixel coordinates that correspond to world
        coordinates via:
            world_x = pixel_x * (physical_width / output_width) + x_min
            world_y = pixel_y * (physical_height / output_height) + y_min

        Note: Y=0 in the output image corresponds to y_min in world coords,
        and Y increases downward (image space) while world Y increases upward.

        Args:
            output_size: Desired output image size (width, height) in pixels
            physical_area: Physical area ((x_min, y_min), (x_max, y_max))
              in real-world coordinates (mm)

        Returns:
            Aligned image as a NumPy array in BGR format, or None on failure
        """
        # Snapshot under the lock so a concurrent frame update cannot
        # change the buffer (or its dimensions) mid-transformation.
        with self._frame_lock:
            image = self._image_data

        if image is None:
            logger.warning("No image data available.")
            return None

        if self.config.image_to_world is not None:
            return self._transform_with_homography(
                image, output_size, physical_area
            )

        out_width, out_height = output_size
        try:
            return resize_linear_nd(image, (out_width, out_height))
        except cv2.error as e:
            logger.error(f"Failed to resize image: {e}")
            return None

    def _transform_with_homography(
        self,
        image: np.ndarray,
        output_size: tuple[int, int],
        physical_area: tuple[Pos, Pos],
    ) -> np.ndarray | None:
        """
        Transform an image using homography to world coordinates.

        Args:
            image: Source image to transform
            output_size: Desired output image size (width, height) in pixels
            physical_area: Physical area ((x_min, y_min), (x_max, y_max))

        Returns:
            Transformed image, or None on failure
        """
        if self.config.image_to_world is None:
            logger.error("Cannot transform: no calibration points set")
            return None

        try:
            H = self._compute_homography(image.shape[0])
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

        try:
            return cv2.warpPerspective(image, np.linalg.inv(M), output_size)
        except cv2.error as e:
            logger.error(f"Failed to apply perspective warp: {e}")
            return None

    def _apply_settings(self, source) -> None:
        """Applies the current settings to the VideoCapture object."""
        try:
            source.apply_settings()
            self._settings_dirty = False
            logger.debug("Applied camera hardware settings.")
        except (cv2.error, OSError, ValueError) as e:
            # We log as a warning because the stream may still work
            logger.warning(f"Could not apply one or more camera settings: {e}")

    def _get_effective_calibration(self, h, w):
        if self.config.has_calibration:
            calib_size = self.config.calibration_image_size
            cam_mat = self.config.get_camera_matrix()
            dist = self.config.get_distortion_coeffs()

            if calib_size is not None and cam_mat is not None:
                calib_w, calib_h = calib_size
                if calib_w != w or calib_h != h:
                    scale_x = w / calib_w
                    scale_y = h / calib_h
                    cam_mat = cam_mat.copy()
                    cam_mat[0, 0] *= scale_x
                    cam_mat[1, 1] *= scale_y
                    cam_mat[0, 2] *= scale_x
                    cam_mat[1, 2] *= scale_y

            return cam_mat, dist

        k1 = self.config.distortion_k1
        k2 = self.config.distortion_k2
        p1 = self.config.distortion_p1
        p2 = self.config.distortion_p2
        k3 = self.config.distortion_k3

        if k1 != 0.0 or k2 != 0.0 or p1 != 0.0 or p2 != 0.0 or k3 != 0.0:
            f = max(h, w)
            cam_mat = np.array(
                [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32
            )
            dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
            return cam_mat, dist_coeffs

        return None, None

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Applies denoise, fisheye correction, and boundary stretching."""

        # 0. Temporal Denoising (Accumulate Weighted)
        denoise_strength = getattr(self.config, "denoise", 0.0)

        if denoise_strength > 0.0:
            with self._frame_lock:
                if (
                    self._accumulator is None
                    or self._accumulator.shape != frame.shape
                ):
                    self._accumulator = frame.astype(np.float32)
                else:
                    alpha = 1.0 - denoise_strength
                    cv2.accumulateWeighted(frame, self._accumulator, alpha)

                # Use the denoised result for subsequent processing
                frame_to_process = self._accumulator.astype(np.uint8)
        else:
            with self._frame_lock:
                self._accumulator = None
            frame_to_process = frame

        h, w = frame_to_process.shape[:2]

        # 1. Undistort (Fisheye Correction)
        cam_mat, dist = self._get_effective_calibration(h, w)
        if cam_mat is not None and dist is not None:
            try:
                frame_to_process = cv2.undistort(
                    frame_to_process, cam_mat, dist
                )
            except cv2.error as e:
                logger.error(f"Failed to undistort frame: {e}")

        return frame_to_process

    def _read_frame(self, source) -> bool:
        """Read frame from cap. Returns True on success."""
        try:
            frame = source.read_frame()
            if frame is None:
                with self._frame_lock:
                    self._image_data = None
                    self._raw_image_data = None
                return False

            raw = frame.copy()
            self._last_frame_warning_log_time = None
            self._last_reconnect_warning_log_time = None

            # Apply all visual corrections
            processed = self._process_frame(frame)

            # Publish both buffers atomically so a reader never observes a
            # fresh _image_data paired with a stale _raw_image_data (or
            # vice versa).
            with self._frame_lock:
                self._raw_image_data = raw
                self._image_data = processed

            # Emit the signal in a GLib-safe way
            idle_add(self.image_captured.send, self)
            return True
        except (cv2.error, OSError, ValueError) as e:
            logger.error(f"Error reading frame: {e}")
            return False

    def _should_log_warning(
        self, last_log_time: float | None, interval: float | None
    ) -> bool:
        if interval is None:
            return True
        now = time.monotonic()
        return last_log_time is None or now - last_log_time >= interval

    def _handle_frame_failure(self):
        """Handle a failed frame read. Returns True if should reconnect."""
        self._consecutive_failures += 1
        with self._frame_lock:
            self._image_data = None
        return self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES

    def _log_frame_failure(self, source) -> None:
        interval = getattr(source, "warning_log_interval_seconds", None)
        if not self._should_log_warning(
            self._last_frame_warning_log_time, interval
        ):
            return
        self._last_frame_warning_log_time = time.monotonic()
        logger.warning(
            "Frame failure %s/%s for %s",
            self._consecutive_failures,
            self.MAX_CONSECUTIVE_FAILURES,
            self.config.name,
        )

    def _log_reconnect(self, source) -> None:
        delay = getattr(
            source, "reconnect_delay_seconds", self.RECONNECT_DELAY
        )
        interval = getattr(source, "warning_log_interval_seconds", None)
        if self._should_log_warning(
            self._last_reconnect_warning_log_time, interval
        ):
            self._last_reconnect_warning_log_time = time.monotonic()
            logger.info(
                "Waiting %ss before reconnecting %s...",
                delay,
                self.config.name,
            )

    def _capture_frames_from_source(self, source):
        """Capture frames from an opened device. Returns when should stop."""
        self._settings_dirty = True
        self._consecutive_failures = 0
        with self._frame_lock:
            self._accumulator = None

        while self._running:
            if self._settings_dirty:
                self._apply_settings(source)

            if self._read_frame(source):
                self._consecutive_failures = 0
            elif self._handle_frame_failure():
                logger.error(
                    f"Too many failures for {self.config.name}, "
                    "reconnecting..."
                )
                return
            else:
                self._log_frame_failure(source)

            # Interruptible pacing wait: a stop() request wakes this up
            # immediately instead of after the full frame interval.
            if self._stop_event.wait(self.FRAME_READ_TIMEOUT):
                return

    def _capture_loop(self):
        """
        Internal method to continuously capture images from the camera.
        Runs in a separate thread.
        """
        logger.info(
            "Capture loop starting for %s (source: %s)",
            self.config.name,
            self.config.source_display_value(),
        )

        while self._running:
            source = create_camera_source(self.config)
            # Not all test doubles implement the full CameraSource
            # interface; binding the cancel event is best-effort so a
            # source that omits it still behaves like an ordinary,
            # non-interruptible source.
            bind_cancel_event = getattr(source, "bind_cancel_event", None)
            if bind_cancel_event is not None:
                bind_cancel_event(self._stop_event)
            self._active_source = source
            try:
                # Open the device ONCE
                source.open()
                self._capture_frames_from_source(source)
            except OSError as exc:
                logger.error("IO error for %s: %s", self.config.name, exc)
            except cv2.error as exc:
                logger.error("OpenCV error for %s: %s", self.config.name, exc)
            except Exception:
                logger.exception("Unexpected error for %s", self.config.name)
            finally:
                source.close()
                if self._active_source is source:
                    self._active_source = None

            if self._running:
                reconnect_delay = getattr(
                    source, "reconnect_delay_seconds", self.RECONNECT_DELAY
                )
                self._log_reconnect(source)
                if self._stop_event.wait(reconnect_delay):
                    break

        logger.debug(
            f"Camera capture loop stopped for camera {self.config.name}."
        )

    def _start_capture_stream(self):
        """
        Starts a continuous image capture stream in a separate thread.

        Serialized with `_stop_capture_stream` via `_lifecycle_lock` so
        that a start racing a stop can never leave two capture threads
        alive for the same controller.
        """
        with self._lifecycle_lock:
            self._start_locked()

    def _start_locked(self):
        """Starts the capture thread. Caller must hold `_lifecycle_lock`."""
        if self._thread_stuck:
            logger.error(
                f"Refusing to start capture stream for {self.config.name}: "
                "a previous capture thread never terminated and may "
                "still hold the device open. Starting a new one here "
                "could open the hardware twice."
            )
            return
        if self._running:
            logger.debug(
                f"Capture stream already running for camera {self.config.name}"
            )
            return

        logger.debug(f"Starting capture stream for camera {self.config.name}.")
        self._stop_event.clear()
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"CameraCapture-{self.config.name}",
        )
        self._capture_thread.daemon = True  # Allow the main program to exit
        self._capture_thread.start()

    def _stop_capture_stream(self):
        """
        Stops the continuous image capture stream.

        Serialized with `_start_capture_stream` via `_lifecycle_lock`.
        """
        with self._lifecycle_lock:
            self._stop_locked()

    def _stop_locked(self):
        """Stops the capture thread. Caller must hold `_lifecycle_lock`.

        Blocks until the capture thread has actually exited (or forces
        its source closed and gives it one more chance to exit), so a
        subsequent start can never leave two threads holding the same
        device. If the thread still refuses to die, the controller marks
        itself stuck and permanently refuses to start a new stream.
        """
        if not self._running:
            logger.debug(
                f"Capture stream not running for camera {self.config.name}."
            )
            return

        logger.debug(f"Stopping capture stream for camera {self.config.name}.")
        self._running = False
        self._stop_event.set()
        if self._active_source is not None:
            # Interrupt a potentially blocking read/open before waiting
            # on the thread.
            self._active_source.stop()

        thread = self._capture_thread
        if thread is None or not thread.is_alive():
            self._capture_thread = None
            return

        thread.join(timeout=self.STOP_JOIN_TIMEOUT)
        if thread.is_alive():
            logger.warning(
                f"Capture thread for {self.config.name} did not exit "
                f"within {self.STOP_JOIN_TIMEOUT}s; forcing its source "
                "closed to try to unblock it."
            )
            if self._active_source is not None:
                try:
                    self._active_source.close()
                except Exception:
                    logger.exception(
                        f"Error force-closing source for {self.config.name}"
                    )
            thread.join(timeout=self.STOP_JOIN_TIMEOUT)

        if thread.is_alive():
            self._thread_stuck = True
            logger.error(
                f"Capture thread for {self.config.name} refused to "
                "terminate even after a forced close. Refusing to start "
                "any new stream on this controller to avoid opening the "
                "device twice."
            )
            return

        self._capture_thread = None

    def _clear_image_data(self) -> None:
        """Drops the processed frame under the frame lock."""
        with self._frame_lock:
            self._image_data = None

    def capture_image(self, *, apply_settings: bool = True):
        """
        Captures a single image from this camera device.
        """
        source = create_camera_source(self.config)
        try:
            if not apply_settings and hasattr(source, "open_for_preview"):
                source.open_for_preview()
            else:
                source.open()
            if apply_settings:
                self._apply_settings(source)
            elif hasattr(source, "apply_preview_settings"):
                source.apply_preview_settings()
            self._read_frame(source)
        except OSError as exc:
            logger.error("IO error capturing image: %s", exc)
            self._clear_image_data()
        except cv2.error as exc:
            logger.error("OpenCV error capturing image: %s", exc)
            self._clear_image_data()
        except Exception:
            logger.exception("Unexpected error capturing image")
            self._clear_image_data()
        finally:
            source.close()

    def read_current_source_settings(self) -> dict[str, object]:
        source = create_camera_source(self.config)
        try:
            source.open()
            return source.read_current_settings()
        finally:
            source.close()
