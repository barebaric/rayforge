from .driver import Driver
from ..models.ops import Ops
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from ..models.machine import Machine


class NoDeviceDriver(Driver):
    """
    A dummy driver that is used if the user has no machine.
    """

    label = _("No driver")
    subtitle = _("No connection")
    supports_settings = False

    async def connect(self) -> None:
        pass

    async def run(self, ops: Ops, machine: "Machine") -> None:
        pass

    async def set_hold(self, hold: bool = True) -> None:
        pass

    async def cancel(self) -> None:
        pass

    async def home(self) -> None:
        pass

    async def move_to(self, pos_x, pos_y) -> None:
        pass

    async def read_settings(self) -> None:
        pass

    async def write_setting(self, key: str, value: Any) -> None:
        pass

    def get_setting_definitions(self) -> dict[str, str]:
        return {}
