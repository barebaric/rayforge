"""The generic remote-machine driver.

A remote machine is attached to a raydriver daemon on another
computer. The protocol it speaks is the daemon's knowledge: this
driver connects to a *device* (``raydriver.remote.client.open_session``)
and learns the protocol and its capabilities from the daemon as
info. It never names a protocol or selects a protocol-specific
driver — that is the layer boundary this design preserves.

The local protocol-specific shells (GRBL/Marlin/Smoothie) are used
only for locally attached hardware.
"""

import asyncio
import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raydriver.protocols import protocol_info
from raydriver.remote.client import open_session

from ...context import RayforgeContext
from ...core.varset import RemoteDeviceVar, Var, VarSet
from ...shared.units.system import UnitSystem
from ..transport import TransportStatus
from .driver import DriverPrecheckError, DriverSetupError
from .grbl.grbl_util import apply_setting_to_varset, get_grbl_setting_varsets
from .remote_setup import endpoint_config, endpoint_vars
from .session_driver import SessionDriverBase

if TYPE_CHECKING:
    from ..device.profile import DeviceProfile
    from ..models.machine import Machine

logger = logging.getLogger(__name__)

#: Hidden setup-var key carrying the daemon-resolved protocol. It is
#: info from raydriver, used only to look up capabilities/dialect —
#: never to select a driver.
PROTOCOL_KEY = "_raydriver_protocol"


def _capabilities(driver_args: dict[str, Any]) -> dict[str, Any]:
    name = driver_args.get(PROTOCOL_KEY)
    if not name:
        return {}
    try:
        return protocol_info(name).capabilities
    except KeyError:
        return {}


class RaydriverRemoteDriver(SessionDriverBase):
    """A machine hosted by a raydriver daemon, protocol-agnostic."""

    label = _("Remote machine (raydriver)")
    subtitle = _("A device attached to a raydriver daemon on another computer")
    uses_gcode = True
    # Per-device capabilities come from the daemon's metadata; the
    # class-level defaults are conservative and overridden below.
    supports_settings = False
    supports_probing = False
    supports_unit_detection = False

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self._device: str | None = None
        self._endpoint: dict[str, Any] = {}

    # -- capabilities from daemon metadata -----------------------------

    @classmethod
    def supports_probing_for(cls, driver_args: dict[str, Any]) -> bool:
        return bool(_capabilities(driver_args).get("supports_probing", False))

    def _apply_capabilities(self, caps: dict[str, Any]) -> None:
        self.supports_settings = bool(caps.get("supports_settings", False))
        self.supports_unit_detection = bool(
            caps.get("supports_unit_detection", False)
        )

    # -- setup ---------------------------------------------------------

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        if not kwargs.get("remote_host"):
            raise DriverPrecheckError(_("Daemon hostname must be configured."))

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                *endpoint_vars(),
                RemoteDeviceVar(
                    key="device",
                    label=_("Device"),
                    description=_(
                        "The machine on the daemon. The list is read from "
                        "the daemon once its hostname is set."
                    ),
                ),
                Var(
                    key=PROTOCOL_KEY,
                    label=_("Protocol"),
                    var_type=str,
                    default="",
                    optional=True,
                    visible_when=lambda values: False,
                ),
            ]
        )

    def _setup_implementation(self, **kwargs: Any) -> None:
        self._endpoint = endpoint_config(kwargs)
        self._device = kwargs.get("device") or None
        if not self._endpoint.get("remote_host"):
            raise DriverSetupError(_("Daemon hostname must be configured."))
        if not self._device:
            raise DriverSetupError(_("Select a device on the daemon."))
        self._apply_capabilities(_capabilities(kwargs))
        self._session = open_session(
            self._endpoint,
            device=self._device,
            dialect=self._dialect_templates(),
            event_callback=self._on_session_event,
        )

    async def _connect_implementation(self):
        await super()._connect_implementation()
        if self._session is not None and self._session.metadata:
            self._apply_capabilities(self._session.metadata["capabilities"])

    # -- settings / errors (protocol-aware via metadata) ---------------

    def _dialect_family(self) -> str:
        name = None
        if self._session is not None:
            name = self._session.protocol
        if not name:
            name = self._machine.driver_args.get(PROTOCOL_KEY)
        if not name:
            return "grbl"
        try:
            return protocol_info(name).dialect_family
        except KeyError:
            return "grbl"

    def get_setting_vars(self) -> list["VarSet"]:
        if self._dialect_family() == "grbl":
            return get_grbl_setting_varsets()
        return []

    async def read_settings(self) -> None:
        pairs = await self._require_session().read_settings()
        known_varsets = self.get_setting_vars()
        key_to_varset_map = {
            var_key: varset
            for varset in known_varsets
            for var_key in varset.keys()  # noqa: SIM118
        }
        unknown_vars = VarSet(
            title=_("Unknown Settings"),
            description=_(
                "Settings reported by the device not in the standard list."
            ),
        )
        for key, value_str in pairs:
            target_varset = key_to_varset_map.get(key)
            if target_varset:
                apply_setting_to_varset(target_varset, key, value_str)
            else:
                unknown_vars.add(
                    Var(
                        key=key,
                        label=f"${key}",
                        var_type=str,
                        value=value_str,
                        description=_("Unknown setting from device"),
                    )
                )
        result = known_varsets
        if len(unknown_vars) > 0:
            result.append(unknown_vars)
        self.settings_read.send(self, settings=result)

    async def write_setting(self, key: str, value: Any) -> None:
        if isinstance(value, bool):
            value = 1 if value else 0
        await self._require_session().write_setting(key, str(value))

    def get_error(self, error_code: str) -> Any:
        if self._dialect_family() == "grbl":
            from raydriver.grbl.parser import error_code_to_device_error

            from .session_state import error_from_session_state

            return error_from_session_state(
                error_code_to_device_error(error_code)
            )
        return None

    # -- probing -------------------------------------------------------

    @classmethod
    async def probe(
        cls, context: "RayforgeContext", **kwargs: Any
    ) -> tuple["DeviceProfile", list[str]]:
        kwargs = await cls._ensure_protocol(context, **kwargs)
        family = "grbl"
        name = kwargs.get(PROTOCOL_KEY)
        if name:
            try:
                family = protocol_info(name).dialect_family
            except KeyError:
                family = "grbl"
        if family == "grbl":
            from .grbl.grbl_probe import probe_grbl_device

            return await probe_grbl_device(cls, context, **kwargs)
        return await cls._probe_generic(context, family, **kwargs)

    @classmethod
    async def _ensure_protocol(
        cls, context: "RayforgeContext", **kwargs: Any
    ) -> dict[str, Any]:
        if kwargs.get(PROTOCOL_KEY):
            return kwargs
        session = open_session(
            endpoint_config(kwargs), device=kwargs.get("device"), dialect={}
        )
        try:
            await asyncio.wait_for(session.connect(), timeout=20.0)
            name = session.protocol
        finally:
            await session.disconnect()
        if name:
            kwargs = {**kwargs, PROTOCOL_KEY: name}
        return kwargs

    @classmethod
    async def _probe_generic(
        cls, context: "RayforgeContext", family: str, **kwargs: Any
    ) -> tuple["DeviceProfile", list[str]]:
        from ..device.profile import DeviceMeta, DeviceProfile, MachineConfig
        from ..models.machine import Machine

        machine = Machine(context)
        driver = cls(context, machine)
        driver.setup(**kwargs)
        connected = asyncio.Event()

        def _on_status(sender, status=None, message=None, **kw):
            if status == TransportStatus.CONNECTED:
                connected.set()

        driver.connection_status_changed.connect(_on_status)
        command = "M115" if family == "marlin" else "version"
        try:
            await driver.connect()
            await asyncio.wait_for(connected.wait(), timeout=30.0)
            build_info = await driver.execute_interactive_command(command)
        finally:
            driver.connection_status_changed.disconnect(_on_status)
            await driver.cleanup()
            context.dialect_mgr.dialects_changed.disconnect(
                machine._on_dialects_changed
            )

        dialect_config: dict[str, Any] = {}
        dialect = context.dialect_mgr.get(family)
        if dialect is not None:
            dialect_config = dialect.to_template_dict()
        name = next(
            (line for line in build_info if line and "ok" not in line), None
        )
        profile = DeviceProfile(
            meta=DeviceMeta(
                name=name or _("Remote machine"),
                description=_("Auto-configured via probe wizard"),
            ),
            machine_config=MachineConfig(unit_system=UnitSystem.METRIC),
            dialect_config=dialect_config,
        )
        profile.machine_config.driver = cls.__name__
        profile.machine_config.driver_args = kwargs
        return profile, []
