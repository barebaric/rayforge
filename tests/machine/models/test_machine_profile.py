import pytest
import pytest_asyncio

from typing import TYPE_CHECKING
from rayforge.core.doc import Doc
from rayforge.core.ops import Ops
from rayforge.machine.models.machine import Machine
from rayforge.machine.device.package import load
from rayforge.config import BUILTIN_DEVICES_DIR
from rayforge.shared import tasker

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext


@pytest_asyncio.fixture
async def carvera_air_machine(
    context_initializer: "RayforgeContext",
) -> "Machine":
    """Provides a Machine instance configured from the Carvera Air device."""
    pkg = load(BUILTIN_DEVICES_DIR / "carvera-air")
    machine = pkg.create_machine(context_initializer)
    tasker.task_mgr.wait_until_settled(5000)
    return machine


@pytest.mark.asyncio
async def test_carvera_air_gcode_generation(carvera_air_machine: "Machine"):
    """
    Tests that a simple line move operation generates the correct G-code
    for the Carvera Air profile, including its custom dialect settings for
    preamble, postscript, and command templates.
    """
    # --- Arrange ---
    machine = carvera_air_machine
    ops = Ops()
    ops.job_start()
    ops.set_cut_speed(600)
    ops.set_power(0.5)
    ops.line_to(10.123, 20.456, 0)
    ops.job_end()
    doc = Doc()

    # --- Act ---
    gcode_str, _ = machine.encode_ops(ops, doc)

    # --- Assert ---
    # The Carvera profile specifies gcode_precision=4.
    # The dialect definition in the profile dictates the command format.
    expected_gcode = [
        # Preamble from profile's dialect_definition
        "M321",
        "G0Z0",
        "G00 G54",
        "M3",
        "G21 ; Set units to mm",
        "G90 ; Absolute positioning",
        # WCS is NOT emitted because inject_wcs_after_preamble=False
        # linear_move: "G1 X{x} Y{y} Z{z}{s_command}{f_command}"
        # The profile's laser head has max_power=1.0, so 50% power is S0.5.
        "G1 X10.123 Y20.456 Z0 S0.5 F600",
        # JobEndCommand triggers _laser_off, which is "G1 S0" for Carvera
        "G1 S0",
        # Postscript from profile's dialect_definition
        "M5 ; Ensure laser is off",
        "G0 X0 Y0 ; Return to origin",
        ";USER END SCRIPT",
        "M322",
        ";USER END SCRIPT",
        "M2",
        "",  # Final newline from G-code encoder's _finalize method
    ]

    assert gcode_str == "\n".join(expected_gcode)


@pytest.mark.asyncio
async def test_inject_wcs_after_preamble_flag(carvera_air_machine: "Machine"):
    """
    Tests that inject_wcs_after_preamble flag controls whether
    WCS is injected after the preamble.
    """
    from rayforge.machine.models.dialect import GcodeDialect

    # --- Arrange ---
    machine = carvera_air_machine
    ops = Ops()
    ops.job_start()
    ops.set_cut_speed(600)
    ops.set_power(0.5)
    ops.line_to(10.123, 20.456, 0)
    ops.job_end()
    doc = Doc()

    # --- Act: With inject_wcs_after_preamble=True (default) ---
    gcode_str, _ = machine.encode_ops(ops, doc)
    gcode_lines = gcode_str.split("\n")

    # --- Assert: WCS should be present ---
    # Carvera Air profile has "G00 G54" in preamble
    assert "G00 G54" in gcode_lines

    # --- Act: With inject_wcs_after_preamble=False ---
    # Create a custom dialect with the flag disabled
    custom_dialect = GcodeDialect(
        label="No WCS Dialect",
        description="Dialect without WCS injection",
        laser_on="",
        laser_off="",
        focus_laser_on="",
        tool_change="",
        set_speed="",
        travel_move="",
        linear_move="",
        arc_cw="",
        arc_ccw="",
        bezier_cubic="",
        air_assist_on="",
        air_assist_off="",
        home_all="",
        home_axis="",
        move_to="",
        jog="",
        clear_alarm="",
        set_wcs_offset="",
        probe_cycle="",
        preamble=["G21", "G90"],
        postscript=["M5"],
        inject_wcs_after_preamble=False,
    )
    machine.context.dialect_mgr.register(custom_dialect)
    machine.set_dialect_uid(custom_dialect.uid)

    gcode_str, _ = machine.encode_ops(ops, doc)
    gcode_lines = gcode_str.split("\n")

    # --- Assert: WCS should NOT be present ---
    assert "G54" not in gcode_lines


@pytest.mark.asyncio
async def test_builtin_devices_all_load():
    """All bundled device packages can be loaded."""
    for d in sorted(BUILTIN_DEVICES_DIR.iterdir()):
        if d.is_dir():
            pkg = load(d)
            assert pkg.name
            assert pkg.dialect_config


@pytest.mark.asyncio
async def test_device_without_rotary_modules(
    context_initializer: "RayforgeContext",
):
    """Devices without rotary_modules create machines with none."""
    pkg = load(BUILTIN_DEVICES_DIR / "sculpfun-icube")
    machine = pkg.create_machine(context_initializer)
    assert machine.rotary_modules == {}
