import asyncio
import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, cast

import raydriver.simulator
from raygeo.ops.axis import Axis

from ...context import RayforgeContext
from ...core.varset import VarSet
from .driver import Pos
from .native import NativeDriver

if TYPE_CHECKING:
    from ..models.machine import Machine

logger = logging.getLogger(__name__)


class NoDeviceDriver(NativeDriver):
    """
    A simulated machine used when the user has no device.

    The simulation itself runs in the native raydriver simulator; this
    binding supplies the machine state and the presentation data.
    """

    NATIVE = raydriver.simulator.NoDeviceDriver

    label = _("No driver")
    subtitle = _("No connection")
    supports_settings = False
    reports_granular_progress = True
    # Delay between simulated commands. Tests that only care about the
    # resulting calls (not real-time progress) set this to 0.
    simulation_step_delay_s = 0.01

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        # The simulator is usable without an explicit setup() call;
        # tests and the UI construct it directly.
        self._session = self._create_session(self.build_config({}))

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates")

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        pass

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(title=_("No settings"))

    def get_setting_vars(self) -> list["VarSet"]:
        return [VarSet(title=_("No settings"))]

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        coordinate_systems = cast(
            dict[str, Pos],
            {
                key: tuple(value.offset)
                for key, value in self._machine.coordinate_systems.items()
            },
        )
        return {
            "step_delay": self.simulation_step_delay_s,
            "active_wcs": self._machine.active_wcs,
            "coordinate_systems": coordinate_systems,
        }

    def _create_session(self, config: dict[str, Any]) -> Any:
        return raydriver.simulator.NoDeviceDriver(
            config=config,
            event_callback=self._on_session_event,
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        offsets = await super().read_wcs_offsets()
        self.wcs_updated.send(self, offsets=offsets)
        return offsets

    async def read_parser_state(self) -> str | None:
        return self._machine.active_wcs

    async def run_probe_cycle(
        self, axis: Axis, target: float, feed_rate: int
    ) -> Pos | None:
        self.probe_status_changed.send(
            self, message=f"Simulating probe cycle for axis {axis.name}..."
        )
        await asyncio.sleep(0.5)
        simulated_pos = (10.0, 15.0, -1.0)
        self.probe_status_changed.send(
            self, message=f"Probe triggered at {simulated_pos}"
        )
        return simulated_pos
