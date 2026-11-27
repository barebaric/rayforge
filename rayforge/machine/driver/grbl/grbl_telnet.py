import logging
from gettext import gettext as _
from typing import Any, cast

import raygeo.driver.grbl
from raygeo.driver.core.discovery.fingerprints import TcpFingerprint
from raygeo.driver.core.discovery.spec import (
    DiscoverySpec,
    MdnsRecognizer,
    TcpRecognizer,
)

from ....core.varset import HostnameVar, PortVar, Var, VarSet
from ....core.varset.hostnamevar import is_valid_hostname_or_ip
from ..driver import DriverPrecheckError, DriverSetupError
from .grbl_serial import GrblSerialDriver

logger = logging.getLogger(__name__)


class GrblTelnetDriver(GrblSerialDriver):
    """
    GRBL-compatible controller connected over raw TCP (telnet).

    Intended for networked grblHAL controllers with the "raw" telnet
    service enabled, and for ESP3D firmware's telnet bridge. The
    entire GRBL protocol stack runs in Rust via the ``raygeo``
    session; only the transport selection differs from the serial
    driver.
    """

    NATIVE = raygeo.driver.grbl.GrblTelnetDriver

    label = _("grblHAL (Telnet)")
    subtitle = _(
        "Raw TCP/telnet connection (grblHAL networking, ESP3D telnet bridge)"
    )
    # Serial-port discovery is inherited from GrblSerialDriver but
    # makes no sense here: a telnet connection has no serial port and
    # must not claim devices found on one. grblHAL builds with LWIP
    # mDNS announce _telnet._tcp, so that fast path is fingerprinted
    # with $I -> [VER: before a board is claimed. Boards whose firmware
    # announces nothing stay discoverable through raygeo's subnet
    # scan, which probes the raw telnet port with the same
    # confirmation.
    DISCOVERY = DiscoverySpec(
        mdns=MdnsRecognizer(
            fingerprint_services=("_telnet._tcp",),
            fingerprint=TcpFingerprint(
                probe_command="$I\n",
                match_prefix="[VER:",
            ),
        ),
        tcp=TcpRecognizer(port=23, probe_command="$I\n", match_prefix="[VER:"),
    )

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
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
                    description=_("TCP port for the raw/telnet service"),
                    default=23,
                ),
                Var(
                    key="poll_status_while_running",
                    label=_("Poll device status during jobs"),
                    description=_(
                        "Periodically query the device for position and "
                        "status while a job is running. Warning: Some "
                        "devices have trouble maintaining a stable "
                        "connection if this is used!"
                    ),
                    var_type=bool,
                    default=False,
                ),
                Var(
                    key="deadlock_detection",
                    label=_("Deadlock detection"),
                    description=_(
                        "Detect and recover from serial communication "
                        "deadlocks during jobs. If disabled, the driver "
                        "will simply wait for the machine to respond. "
                        "Disable if you experience false ALARM:3 errors."
                    ),
                    var_type=bool,
                    default=False,
                ),
            ]
        )

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        host = cast(str, kwargs.get("host", ""))
        port = cast(int, kwargs.get("port", 23))

        if not host:
            raise DriverSetupError(_("Hostname must be configured."))
        if not port:
            raise DriverSetupError(_("Port must be configured."))

        config = {
            "host": host,
            "tcp_port": int(port),
            "poll_status_while_running": bool(
                kwargs.get("poll_status_while_running", False)
            ),
            "deadlock_detection": bool(
                kwargs.get("deadlock_detection", False)
            ),
            "homing_timeout": self._estimate_homing_timeout(),
        }
        cached_rx_buffer_size = self.config.get("rx_buffer_size")
        if cached_rx_buffer_size is not None:
            config["cached_rx_buffer_size"] = int(cached_rx_buffer_size)

        return config
