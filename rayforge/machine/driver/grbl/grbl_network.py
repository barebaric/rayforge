import logging
from gettext import gettext as _
from typing import Any, cast

import raydriver.grbl
from raydriver.core.discovery.fingerprints import HttpFingerprint
from raydriver.core.discovery.spec import DiscoverySpec, MdnsRecognizer

from ....core.varset import (
    ChoiceVar,
    HostnameVar,
    IntVar,
    PortVar,
    VarSet,
)
from ....core.varset.hostnamevar import is_valid_hostname_or_ip
from ..driver import DriverPrecheckError, DriverSetupError
from .grbl_serial import GrblSerialDriver

logger = logging.getLogger(__name__)


def _resolve_ws_port(port: int, ws_port: int, protocol: str) -> int:
    """Resolve the WebSocket port for the ESP3D/FluidNC transport.

    ``0`` means "use the HTTP port", which FluidNC v4 requires. An
    explicitly configured positive port is preserved so existing
    profiles (and split-port forks) keep working.
    """
    if ws_port <= 0:
        return int(port)
    if protocol == "FluidNC" and ws_port != int(port):
        logger.warning(
            f"FluidNC expects the WebSocket port to match the HTTP "
            f"port ({port}); using {ws_port} as configured."
        )
    return ws_port


class GrblNetworkDriver(GrblSerialDriver):
    """
    A driver for GRBL-compatible controllers that use a modern file
    upload API and allow reading/writing device settings.

    The ESP3D/FluidNC protocol stack runs in Rust via the raydriver
    library: commands are HTTP GETs and replies arrive over a
    WebSocket; jobs are uploaded and executed instead of being
    streamed line by line.
    """

    NATIVE = raydriver.grbl.GrblNetworkDriver

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
            fingerprint=HttpFingerprint(
                path="/command?plain=%5BESP800%5D&PAGEID=",
                match_regex=r"\b(fluidnc|esp3d|grblhal|grbl)\b",
            ),
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
                IntVar(
                    key="ws_port",
                    label=_("WebSocket Port"),
                    description=_(
                        "The WebSocket port for the device. Leave at 0 "
                        "to use the HTTP port (required for FluidNC v4); "
                        "older ESP3D firmware uses 81."
                    ),
                    default=0,
                    min_val=0,
                    max_val=65535,
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

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        host = cast(str, kwargs.get("host", ""))
        port = int(cast(int, kwargs.get("port", 80)))
        protocol = cast(str, kwargs.get("protocol", "ESP3D"))
        if not host:
            raise DriverSetupError(_("Hostname must be configured."))
        ws_port = _resolve_ws_port(
            port, int(kwargs.get("ws_port") or 0), protocol
        )

        return {
            "host": host,
            "http_port": int(port),
            "ws_port": ws_port,
            "protocol": protocol,
            "homing_timeout": self._estimate_homing_timeout(),
        }
