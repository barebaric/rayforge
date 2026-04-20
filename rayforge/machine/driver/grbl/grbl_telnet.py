import logging
from typing import Any, Optional, cast
from gettext import gettext as _
from ....core.varset import Var, VarSet, HostnameVar, PortVar
from ...transport import TelnetTransport
from ...transport.grbl import GrblSerialTransport
from ...transport.validators import is_valid_hostname_or_ip
from ..driver import DriverPrecheckError, DriverSetupError
from .grbl_serial import GrblSerialDriver


logger = logging.getLogger(__name__)


class GrblTelnetDriver(GrblSerialDriver):
    """
    GRBL-compatible controller connected over raw TCP (telnet).

    Intended for networked grblHAL controllers with the "raw" telnet
    service enabled, and for ESP3D firmware's telnet bridge. Reuses the
    full GRBL protocol stack from GrblSerialDriver, swapping only the
    underlying byte transport from SerialTransport to TelnetTransport.
    """

    label = _("GRBL (Telnet)")
    subtitle = _("GRBL-compatible controller over a raw TCP/telnet connection")

    def __init__(self, context, machine):
        super().__init__(context, machine)
        self._host: Optional[str] = None
        self._port: Optional[int] = None

    @property
    def resource_uri(self) -> Optional[str]:
        if self._host and self._port:
            return f"tcp://{self._host}:{self._port}"
        return None

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
            ]
        )

    def _setup_implementation(self, **kwargs: Any) -> None:
        host = cast(str, kwargs.get("host", ""))
        port = cast(int, kwargs.get("port", 23))
        self._poll_status_while_running = bool(
            kwargs.get("poll_status_while_running", False)
        )

        if not host:
            raise DriverSetupError(_("Hostname must be configured."))
        if not port:
            raise DriverSetupError(_("Port must be configured."))

        self._host = host
        self._port = port

        telnet_transport = TelnetTransport(host, port)
        self.grbl_transport = GrblSerialTransport(telnet_transport)
        self.grbl_transport.received.connect(self.on_serial_data_received)
        self.grbl_transport.status_changed.connect(
            self.on_serial_status_changed
        )
