"""Direct USB/UDP driver for Ruida laser controllers.

Wraps the external RuidaDriver from ruida-protocol-analyzer for direct
communication over USB or UDP.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ruidadriver.ruida_driver import RdDriver

from rayforge.pipeline.encoder.base import EncodedOutput

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from rayforge.core.doc import Doc
    from rayforge.machine.driver.ruidarpa.rpa_encoder import (
        RuidaRPAEncoder,
    )
    from rayforge.machine.models.machine import Machine

_logger = logging.getLogger(__name__)


class RpaDirectDriver:
    """Direct connection to a Ruida laser controller via USB or UDP.

    Wraps RdDriver lifecycle and listener management. All callbacks
    fire from background daemon threads.
    """

    def __init__(self) -> None:
        self._driver: RdDriver | None = None

    # --- Lifecycle ---

    def start(
        self,
        udp_host: str | None = None,
        usb_device: str | None = None,
        magic: int | None = None,
    ) -> bool:
        """Start connection to the Ruida controller.

        Idempotent: starting with unchanged parameters is a no-op, while
        a different host or device restarts the connection. None reuses
        the previously used value.

        Args:
            udp_host: UDP hostname/IP (e.g. '192.168.1.100').
            usb_device: USB device path.
            magic: Optional controller magic number (0x00-0xFF).

        Returns:
            True if connection succeeded.
        """
        driver = self._ensure_driver()
        kwargs: dict = {"udp_host": udp_host, "usb_device": usb_device}
        if magic is not None:
            kwargs["magic"] = magic
        result = driver.start(**kwargs)
        if result:
            _logger.info(
                "RPA direct driver connected; udp=%s, usb=%s",
                udp_host,
                usb_device,
            )
        else:
            _logger.warning("RPA direct driver failed to connect")
        return result

    def stop(self) -> None:
        """Disconnect the driver.

        The underlying RdDriver instance is retained so that
        connection parameters survive a restart.
        """
        if self._driver is not None:
            try:
                self._driver.stop()
            except Exception:
                _logger.exception("Error stopping RPA direct driver")

    @property
    def is_connected(self) -> bool:
        """Whether the underlying driver is connected."""
        return self._driver is not None and self._driver.is_connected

    # --- Run control ---

    def run(self, script: list[str], auto_checksum: bool = False) -> None:
        """Run an Rpascript.

        Queues the raw script without head/tail composition.

        Args:
            script: List of Rpascript command strings.
            auto_checksum: Whether to auto-calculate checksums.
        """
        if not script:
            return
        driver = self._require_connected()
        driver.run(script, auto_checksum=auto_checksum)

    # --- Encoder integration ---

    @property
    def gluescript(self) -> RdDriver:
        """The wrapped RdDriver, created on first use.

        The encoder authors ops directly into this GlueScript so a live
        job can be re-encoded into the connected driver and run via
        ``run_job()``.
        """
        return self._ensure_driver()

    def run_job(
        self, job: list[str] | None = None, auto_checksum: bool = False
    ) -> None:
        """Run a job, composing head + job + tail on the connected driver.

        Args:
            job: Rpascript job lines; None runs the staged/current job.
            auto_checksum: Whether to auto-calculate checksums.
        """
        self._require_connected().run_job(job, auto_checksum=auto_checksum)

    @staticmethod
    def create_encoder() -> RuidaRPAEncoder:
        """Create an RPA encoder for converting Ops to rpascript."""
        from rayforge.machine.driver.ruidarpa.rpa_encoder import (
            RuidaRPAEncoder,
        )

        return RuidaRPAEncoder()

    def run_encoded(
        self, encoded: EncodedOutput, auto_checksum: bool = False
    ) -> None:
        """Run an EncodedOutput by extracting its text lines.

        Args:
            encoded: The encoder output containing rpascript text.
            auto_checksum: Whether to auto-calculate checksums.
        """
        text_lines = [
            line.strip() for line in encoded.text.splitlines() if line.strip()
        ]
        self.run(text_lines, auto_checksum=auto_checksum)

    def encode_and_run(
        self,
        ops: Ops,
        machine: Machine,
        doc: Doc,
        auto_checksum: bool = False,
    ) -> EncodedOutput:
        """Encode Ops to rpascript and run it on the controller.

        Args:
            ops: Ops object from raygeo containing commands to encode.
            machine: The machine configuration.
            doc: The document being processed.
            auto_checksum: Whether to auto-calculate checksums.

        Returns:
            The EncodedOutput produced by the encoder.
        """
        encoder = self.create_encoder()
        encoded = encoder.encode(ops, machine, doc)
        self.run_encoded(encoded, auto_checksum=auto_checksum)
        return encoded

    def cancel_script(self) -> None:
        """Cancel the currently running script."""
        if self._driver is not None:
            self._driver.cancel_script()

    def set_protect(self, enabled: bool) -> None:
        """Enable or disable protect mode.

        When enabled, the machine will not execute SET_SETTING commands,
        allowing safe dry-run testing. May be called before ``start()``.
        """
        self._ensure_driver().set_protect(enabled)

    @property
    def protect_enabled(self) -> bool:
        """Whether protect mode is currently enabled."""
        if self._driver is None:
            return False
        return self._driver.protect_enabled

    # --- Jog / Home ---

    def home(self) -> None:
        """Home the X and Y axes.

        The wrapped RdDriver auto-sends the generated lines when
        connected; the returned lines are deliberately discarded so each
        home command is sent exactly once.
        """
        self._require_connected().home()

    def home_z(self) -> None:
        """Home the Z axis.

        The wrapped RdDriver auto-sends the generated lines when
        connected; the returned lines are deliberately discarded so each
        home command is sent exactly once.
        """
        self._require_connected().home_z()

    def jog_xy_to(self, x: float, y: float) -> None:
        """Jog the XY axes to an absolute position in mm.

        The wrapped RdDriver auto-sends the generated lines when
        connected; the returned lines are deliberately discarded so each
        jog is sent exactly once.
        """
        self._require_connected().jog_xy_to(x, y)

    def jog_xy_rel(
        self, x: float | None = None, y: float | None = None
    ) -> None:
        """Jog the XY axes relative to the current position in mm.

        The wrapped RdDriver auto-sends the generated lines when
        connected; the returned lines are deliberately discarded so each
        jog is sent exactly once.
        """
        self._require_connected().jog_xy_rel(x, y)

    def jog_x_rel(self, x: float | None = None) -> None:
        """Jog the X axis relative to the current position in mm.

        The wrapped RdDriver auto-sends the generated lines when
        connected; the returned lines are deliberately discarded so each
        jog is sent exactly once.
        """
        self._require_connected().jog_x_rel(x)

    def jog_y_rel(self, y: float | None = None) -> None:
        """Jog the Y axis relative to the current position in mm.

        The wrapped RdDriver auto-sends the generated lines when
        connected; the returned lines are deliberately discarded so each
        jog is sent exactly once.
        """
        self._require_connected().jog_y_rel(y)

    def jog_z_rel(self, z: float | None = None) -> None:
        """Jog the Z axis relative to the current position in mm.

        The wrapped RdDriver auto-sends the generated lines when
        connected; the returned lines are deliberately discarded so each
        jog is sent exactly once.
        """
        self._require_connected().jog_z_rel(z)

    def jog_u_rel(self, u: float | None = None) -> None:
        """Jog the U axis relative to the current position in mm.

        The wrapped RdDriver auto-sends the generated lines when
        connected; the returned lines are deliberately discarded so each
        jog is sent exactly once.
        """
        self._require_connected().jog_u_rel(u)

    def jog_set_xy_speed(self, speed: float) -> None:
        """Set the XY jog speed in mm/s.

        Delegates without requiring a connection: RdDriver stores the
        jog speed as session-less state, so the setter works while
        disconnected.
        """
        self._ensure_driver().jog_set_xy_speed(speed)

    def jog_set_z_speed(self, speed: float) -> None:
        """Set the Z jog speed in mm/s.

        Delegates without requiring a connection: RdDriver stores the
        jog speed as session-less state, so the setter works while
        disconnected.
        """
        self._ensure_driver().jog_set_z_speed(speed)

    def jog_set_u_speed(self, speed: float) -> None:
        """Set the U jog speed in mm/s.

        Delegates without requiring a connection: RdDriver stores the
        jog speed as session-less state, so the setter works while
        disconnected.
        """
        self._ensure_driver().jog_set_u_speed(speed)

    # --- Job control ---

    def pause(self) -> None:
        """Pause the running job.

        The wrapped RdDriver auto-sends the generated lines when
        connected; the returned lines are deliberately discarded so each
        pause command is sent exactly once.
        """
        self._require_connected().pause()

    def resume(self) -> None:
        """Resume the paused job.

        The wrapped RdDriver auto-sends the generated lines when
        connected; the returned lines are deliberately discarded so each
        resume command is sent exactly once.
        """
        self._require_connected().resume()

    def stop_job(self) -> None:
        """Stop the running job.

        The wrapped RdDriver auto-sends the generated lines when
        connected; the returned lines are deliberately discarded so each
        stop_job command is sent exactly once.
        """
        self._require_connected().stop_job()

    def reset(self) -> None:
        """Reset the controller.

        Stops the current job and homes the X/Y axes. The wrapped
        RdDriver auto-sends the generated lines when connected; the
        returned lines are deliberately discarded so each reset command
        is sent exactly once.
        """
        self._require_connected().reset()

    # --- Status ---

    @property
    def machine_status(self) -> dict:
        """Current machine status dict.

        Returns an empty dict while disconnected, matching the RPC
        client's disconnected semantics.
        """
        if self._driver is None or not self._driver.is_connected:
            return {}
        return self._driver.machine_status

    # --- Listeners ---

    def register_status_listener(self, callback: Callable) -> None:
        """Register a status listener.

        Callback signature: callable(status_event: str)
        Status events: CONNECTED, DISCONNECTED, SCRIPT_ERROR, etc.
        """
        self._ensure_driver().register_status_listener(callback)

    def register_error_listener(self, callback: Callable) -> None:
        """Register an error listener.

        Callback signature: callable(error_message: str)
        """
        self._ensure_driver().register_error_listener(callback)

    def register_reply_listener(self, callback: Callable) -> None:
        """Register a reply listener.

        Callback signature: callable(reply_data: bytes)
        """
        self._ensure_driver().register_reply_listener(callback)

    def unregister_status_listener(self, listener: Callable) -> None:
        """Unregister a status listener.

        Args:
            listener: The exact callable previously passed to
                :meth:`register_status_listener`.
        """
        self._ensure_driver().unregister_status_listener(listener)

    def unregister_error_listener(self, listener: Callable) -> None:
        """Unregister an error listener.

        Args:
            listener: The exact callable previously passed to
                :meth:`register_error_listener`.
        """
        self._ensure_driver().unregister_error_listener(listener)

    def unregister_reply_listener(self, listener: Callable) -> None:
        """Unregister a reply listener.

        Args:
            listener: The exact callable previously passed to
                :meth:`register_reply_listener`.
        """
        self._ensure_driver().unregister_reply_listener(listener)

    # --- Internal helpers ---

    def _ensure_driver(self) -> RdDriver:
        """Return the RdDriver instance, creating it on first use."""
        if self._driver is None:
            self._driver = RdDriver()
        return self._driver

    def _require_connected(self) -> RdDriver:
        """Raise RuntimeError if not connected, otherwise return the driver."""
        driver = self._ensure_driver()
        if not driver.is_connected:
            raise RuntimeError("RPA direct driver is not connected")
        return driver
