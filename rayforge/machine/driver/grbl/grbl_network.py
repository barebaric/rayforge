import logging
from gettext import gettext as _
from typing import Any, cast

from raydriver.core.discovery.spec import DiscoverySpec, MdnsRecognizer
from raydriver.esp3d import Esp3dSession

from ....core.varset import ChoiceVar, HostnameVar, PortVar, VarSet
from ....core.varset.hostnamevar import is_valid_hostname_or_ip
from ..driver import DriverPrecheckError, DriverSetupError
from .grbl_fingerprint import fingerprint_grbl_http
from .grbl_serial import GrblSerialDriver

logger = logging.getLogger(__name__)


class GrblNetworkDriver(GrblSerialDriver):
    """
    A driver for GRBL-compatible controllers that use a modern file
    upload API and allow reading/writing device settings.

    The ESP3D/FluidNC protocol stack runs in Rust via the raydriver
    library: commands are HTTP GETs and replies arrive over a
    WebSocket; jobs are uploaded and executed instead of being
    streamed line by line.
    """

    label = _("FluidNC / ESP3D (Web API)")
    subtitle = _(
        "FluidNC or ESP3D controller via its HTTP/WebSocket interface"
    )
    reports_granular_progress = False
    # ESP3D v3 firmware advertises a dedicated _esp3d._tcp service;
    # FluidNC only announces the generic _http._tcp and is confirmed
    # by fingerprinting its [ESP800] firmware-info endpoint.
    DISCOVERY = DiscoverySpec(
        mdns=MdnsRecognizer(
            services=("_esp3d._tcp.local.",),
            fingerprint=fingerprint_grbl_http,
        )
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
                    label=_("HTTP Port"),
                    description=_("The HTTP port for the device"),
                    default=80,
                ),
                PortVar(
                    key="ws_port",
                    label=_("WebSocket Port"),
                    description=_(
                        "The WebSocket port for the device. For FluidNC,"
                        " use the same port as HTTP (usually 80)."
                    ),
                    default=81,
                ),
                ChoiceVar(
                    key="protocol",
                    label=_("Protocol variant"),
                    description=_(
                        "Standard, ESP3D, Longer, or FluidNC variant"
                    ),
                    default="ESP3D",
                    choices=["ESP3D", "Longer", "FluidNC"],
                    null_label=_("Standard"),
                ),
            ]
        )

    def _setup_implementation(self, **kwargs: Any) -> None:
        host = cast(str, kwargs.get("host", ""))
        port = cast(int, kwargs.get("port", 80))
        ws_port = cast(int, kwargs.get("ws_port", 81))
        protocol = cast(str, kwargs.get("protocol", "ESP3D"))
        if not host:
            raise DriverSetupError(_("Hostname must be configured."))

        config = {
            "host": host,
            "http_port": int(port),
            "ws_port": int(ws_port),
            "protocol": protocol,
        }
        self._session = Esp3dSession(
            config=config,
            dialect=self._dialect_templates(),
            event_callback=self._on_session_event,
        )
