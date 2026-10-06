import logging
from gettext import gettext as _
from typing import Any, cast

from raydriver.grbl import GrblSession

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
    entire GRBL protocol stack runs in Rust via the ``raydriver``
    session; only the transport selection differs from the serial
    driver.
    """

    label = _("GRBL (Telnet)")
    subtitle = _("GRBL-compatible controller over a raw TCP/telnet connection")
    # Serial-port discovery is inherited from GrblSerialDriver but
    # makes no sense here: a telnet connection has no serial port and
    # must not claim devices found on one.
    DISCOVERY = None

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
                    label=_("Hostname"),
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

    def _setup_implementation(self, **kwargs: Any) -> None:
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
        }
        cached_rx_buffer_size = self.config.get("rx_buffer_size")
        if cached_rx_buffer_size is not None:
            config["cached_rx_buffer_size"] = int(cached_rx_buffer_size)

        self._session = GrblSession(
            config=config,
            dialect=self._dialect_templates(),
            event_callback=self._on_session_event,
        )
