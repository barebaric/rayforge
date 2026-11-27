import json
import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, cast

import raygeo.driver.octoprint
from raygeo.driver.core.discovery.spec import DiscoverySpec, MdnsRecognizer

from ....context import RayforgeContext
from ....core.varset import (
    AppKeyVar,
    HostnameVar,
    PortVar,
    VarSet,
)
from ....core.varset.hostnamevar import is_valid_hostname_or_ip
from ....core.varset.var import Var
from ..driver import DriverPrecheckError, DriverSetupError
from ..native import NativeDriver
from .octoprint_util import build_octoprint_profile

if TYPE_CHECKING:
    from ...device.profile import DeviceProfile

logger = logging.getLogger(__name__)


def _normalize_path_prefix(path: str) -> str:
    """Normalizes a URL path prefix for constructing API base URLs.

    ``/`` (the common case) yields an empty prefix so URLs stay
    clean (``http://host:port/api/…``). A non-root prefix like
    ``/octoprint`` is kept as-is (no trailing slash); request paths
    already start with ``/`` and are appended directly.
    """
    if not path:
        return ""
    prefix = path.strip()
    while len(prefix) > 1 and prefix.endswith("/"):
        prefix = prefix[:-1]
    if prefix == "/":
        return ""
    return prefix


def _extract_api_key(data: str | None) -> str | None:
    if not data:
        return None
    try:
        tokens = json.loads(data)
        if isinstance(tokens, dict):
            return tokens.get("api_key") or tokens.get("access_token")
        return str(tokens)
    except (json.JSONDecodeError, TypeError):
        return data.strip() if data else None


class OctoPrintDriver(NativeDriver):
    """
    Submits G-code jobs to an OctoPrint server via its REST API and
    monitors live printer state through SockJS WebSocket push updates
    with a polling REST fallback.

    The protocol stack runs in Rust via the ``raygeo`` package;
    this binding supplies the connection-setup form and probing.
    """

    NATIVE = raygeo.driver.octoprint.OctoPrintDriver

    label = _("OctoPrint")
    subtitle = _("Submit G-code to an OctoPrint server")
    supports_settings = False
    reports_granular_progress = False
    uses_gcode = True
    supports_probing = True
    # Advertised by OctoPrint's discovery plugin; used for network
    # device discovery. The TXT record's ``path`` key (the URL
    # prefix the server is mounted under, e.g. ``/`` or
    # ``/octoprint``) is forwarded into the ``path`` setup-var so
    # connections target the right base path.
    DISCOVERY = DiscoverySpec(
        mdns=MdnsRecognizer(
            services=("_octoprint._tcp.local.",),
            txt_map={"path": "path"},
        )
    )

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        host = cast(str, kwargs.get("host", ""))
        if host and not is_valid_hostname_or_ip(host):
            raise DriverPrecheckError(
                _("Invalid hostname or IP address: '{host}'").format(host=host)
            )

    @classmethod
    async def probe(
        cls, context: RayforgeContext, **kwargs: Any
    ) -> tuple["DeviceProfile", list[str]]:
        """Query an OctoPrint server for its specs and return an
        auto-populated ``(DeviceProfile, warnings)`` tuple.

        Works without an API key when the server has access control
        disabled (the common case for freshly discovered devices).
        """
        host = cast(str, kwargs.get("host", ""))
        port = cast(int, kwargs.get("port", 80))
        path = cast(str, kwargs.get("path", "/"))
        raw_key = kwargs.get("api_key", "")
        api_key = _extract_api_key(str(raw_key) if raw_key else "")

        session = raygeo.driver.octoprint.OctoPrintDriver(
            config={
                "host": host,
                "port": int(port),
                "path": _normalize_path_prefix(path),
                "api_key": api_key or "",
            }
        )
        version_info = None
        printer_info = None
        try:
            version_info, printer_info = await session.probe_info()
        except (ConnectionError, TimeoutError, OSError):
            pass

        profile, warnings = build_octoprint_profile(version_info, printer_info)
        profile.machine_config.driver = cls.__name__
        profile.machine_config.driver_args = kwargs
        return profile, warnings

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                HostnameVar(
                    key="host",
                    label=_("Hostname or IP address"),
                    description=_(
                        "IP address or hostname of the OctoPrint server"
                    ),
                ),
                PortVar(
                    key="port",
                    label=_("Port"),
                    description=_("HTTP port of the OctoPrint server"),
                    default=80,
                ),
                Var(
                    key="path",
                    label=_("Path"),
                    description=_(
                        "URL prefix the OctoPrint server is mounted "
                        'under (discovered automatically; usually "/").'
                    ),
                    var_type=str,
                    default="/",
                ),
                AppKeyVar(
                    key="api_key",
                    label=_("API Key"),
                    app_name="RayForge",
                    probe_url="http://{host}:{port}/plugin/appkeys/probe",
                    request_url="http://{host}:{port}/plugin/appkeys/request",
                    poll_url="http://{host}:{port}"
                    "/plugin/appkeys/request"
                    "/{{app_token}}",
                    description=_(
                        "Enter an API key manually or click "
                        "'Request Access' to obtain one via "
                        "OctoPrint's Application Keys plugin."
                    ),
                ),
            ]
        )

    def get_setting_vars(self) -> list["VarSet"]:
        return []

    async def clear_alarm(self) -> None:
        """OctoPrint resets a firmware alarm with M999."""
        await self._require_session().clear_alarm()

    async def move_to(
        self,
        pos_x: float,
        pos_y: float,
        pos_z: float | None = None,
        speed: float | None = None,
    ) -> None:
        """Moves to an absolute position via the printhead jog API."""
        await self._require_session().move_to_absolute(
            self._to_machine_length(float(pos_x)),
            self._to_machine_length(float(pos_y)),
            self._to_machine_length(pos_z) if pos_z is not None else None,
            self._to_machine_speed(speed) if speed is not None else None,
        )

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        host = cast(str, kwargs.get("host", ""))
        port = cast(int, kwargs.get("port", 80))
        if not host:
            raise DriverSetupError(_("Hostname must be configured."))

        raw_key = kwargs.get("api_key", "")
        api_key = _extract_api_key(str(raw_key) if raw_key else "")
        if not api_key:
            raise DriverSetupError(
                _(
                    "API key must be configured. Use the "
                    "'Request Access' button or enter an API "
                    "key manually."
                )
            )

        return {
            "host": host,
            "port": int(port),
            "path": _normalize_path_prefix(cast(str, kwargs.get("path", "/"))),
            "api_key": api_key,
        }
