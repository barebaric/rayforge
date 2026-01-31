import pytest
import pytest_asyncio
from typing import TYPE_CHECKING

from rayforge.machine.driver.dummy import NoDeviceDriver
from rayforge.machine.driver.grbl_serial import GrblSerialDriver
from rayforge.machine.models.profile import PROFILES
from rayforge.shared import tasker

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.machine.models.machine import Machine


@pytest_asyncio.fixture
async def sculpfun_icube_machine(
    context_initializer: "RayforgeContext",
) -> "Machine":
    """Provides a Machine instance from the Sculpfun iCube profile."""
    profile = next((p for p in PROFILES if p.name == "Sculpfun iCube"), None)
    assert profile is not None, (
        "Sculpfun iCube profile not found in PROFILES list."
    )

    machine = profile.create_machine(context_initializer)
    context_initializer.machine_mgr.add_machine(machine)

    tasker.task_mgr.wait_until_settled(5000)

    return machine


@pytest.mark.asyncio
async def test_sculpfun_icube_driver_assignment(
    sculpfun_icube_machine: "Machine",
):
    """
    Tests that creating a machine from the Sculpfun iCube profile
    correctly assigns the GrblSerialDriver instead of NoDeviceDriver.
    """
    machine = sculpfun_icube_machine

    assert machine.driver_name == "GrblSerialDriver", (
        f"Expected driver_name to be 'GrblSerialDriver', "
        f"got '{machine.driver_name}'"
    )

    controller = machine.controller
    tasker.task_mgr.wait_until_settled(5000)

    assert not isinstance(controller.driver, NoDeviceDriver), (
        "Driver should not be NoDeviceDriver after profile creation"
    )

    assert isinstance(controller.driver, GrblSerialDriver), (
        f"Expected driver to be GrblSerialDriver, "
        f"got {type(controller.driver).__name__}"
    )
