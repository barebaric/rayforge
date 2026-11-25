"""Smoothieware over telnet, backed by the native raydriver driver.

The protocol stack runs in Rust via the ``raydriver`` package; this
binding only supplies the Rayforge-side setup and probing.
"""

import asyncio
import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, cast

import raydriver.smoothie

from ....core.varset import HostnameVar, PortVar, VarSet
from ....core.varset.hostnamevar import is_valid_hostname_or_ip
from ...transport import TransportStatus
from ..driver import DriverPrecheckError, DriverSetupError
from ..native import NativeDriver
from .smoothie_util import SmoothieProbeResult, build_smoothie_profile

if TYPE_CHECKING:
    from ....context import RayforgeContext
    from ...device.profile import DeviceProfile

logger = logging.getLogger(__name__)


class SmoothieDriver(NativeDriver):
    """
    Handles Smoothie-based devices via Telnet, with the entire
    protocol stack running in Rust via the ``raydriver`` package.
    """

    NATIVE = raydriver.smoothie.SmoothieDriver

    label = _("Smoothie")
    subtitle = _("Smoothieware via a Telnet connection")
    supports_settings = False
    supports_probing = True

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        """Checks if the hostname is a valid format."""
        host = cast(str, kwargs.get("host", ""))
        if not is_valid_hostname_or_ip(host):
            raise DriverPrecheckError(
                _("Invalid hostname or IP address: '{host}'").format(host=host)
            )

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                HostnameVar(
                    key="host",
                    label=_("Hostname or IP address"),
                    description=_("The IP address or hostname of the device"),
                ),
                PortVar(
                    key="port",
                    label=_("Port"),
                    description=_("The Telnet port number"),
                    default=23,
                ),
            ]
        )

    @classmethod
    async def probe(
        cls, context: "RayforgeContext", **kwargs: Any
    ) -> tuple["DeviceProfile", list[str]]:
        """Connect to a Smoothie device, query its configuration, and
        return an auto-populated ``(DeviceProfile, warnings)`` tuple.
        """
        from ...models.machine import Machine

        machine = Machine(context)
        driver = cls(context, machine)
        driver.setup(**kwargs)

        connected = asyncio.Event()

        def _on_status(sender, status=None, message=None, **kw):
            if status == TransportStatus.CONNECTED:
                connected.set()

        driver.connection_status_changed.connect(_on_status)
        try:
            await driver.connect()
            await asyncio.wait_for(connected.wait(), timeout=15.0)
            # Smoothieware needs a moment after the link comes up
            # before it accepts commands.
            await asyncio.sleep(0.2)
            query = driver.execute_interactive_command
            probe = SmoothieProbeResult(
                version=await query("version"),
                alpha_max=await query("config-get alpha_max"),
                beta_max=await query("config-get beta_max"),
                alpha_max_rate=await query("config-get alpha_max_rate"),
                beta_max_rate=await query("config-get beta_max_rate"),
                acceleration=await query("config-get acceleration"),
            )
        finally:
            driver.connection_status_changed.disconnect(_on_status)
            await driver.cleanup()
            context.dialect_mgr.dialects_changed.disconnect(
                machine._on_dialects_changed
            )

        profile, warnings = build_smoothie_profile(probe)
        profile.machine_config.driver = cls.__name__
        profile.machine_config.driver_args = kwargs
        return profile, warnings

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        host = cast(str, kwargs.get("host", ""))
        port = kwargs.get("port", 23)

        if not host:
            raise DriverSetupError(_("Hostname must be configured."))

        return {
            "host": host,
            "tcp_port": int(port),
        }

    def get_setting_vars(self) -> list["VarSet"]:
        return [VarSet()]
