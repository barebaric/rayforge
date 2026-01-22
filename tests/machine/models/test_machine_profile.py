import pytest
import pytest_asyncio
from typing import TYPE_CHECKING
from rayforge.core.doc import Doc
from rayforge.core.ops import (
    Ops,
    JobStartCommand,
    JobEndCommand,
    LineToCommand,
    SetCutSpeedCommand,
    SetPowerCommand,
)
from rayforge.machine.models.profile import PROFILES
from rayforge.shared import tasker

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.machine.models.machine import Machine


@pytest_asyncio.fixture
async def carvera_air_machine(
    context_initializer: "RayforgeContext",
) -> "Machine":
    """Provides a Machine instance configured from the Carvera Air profile."""
    # Find the Carvera Air profile from the list of built-in profiles.
    profile = next((p for p in PROFILES if p.name == "Carvera Air"), None)
    assert profile is not None, (
        "Carvera Air profile not found in PROFILES list."
    )

    # Create the machine instance from the profile.
    # This also creates and registers the custom dialect defined in the
    # profile.
    machine = profile.create_machine(context_initializer)

    # Wait for any pending tasks (like rebuild-driver) to complete
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
    ops.commands = [
        JobStartCommand(),
        SetCutSpeedCommand(speed=600),
        SetPowerCommand(power=0.5),  # 50% power
        LineToCommand(end=(10.123, 20.456, 0)),
        JobEndCommand(),
    ]
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
    from rayforge.machine.models.dialect import (
        GcodeDialect,
        register_dialect,
    )

    # --- Arrange ---
    machine = carvera_air_machine
    ops = Ops()
    ops.commands = [
        JobStartCommand(),
        SetCutSpeedCommand(speed=600),
        SetPowerCommand(power=0.5),
        LineToCommand(end=(10.123, 20.456, 0)),
        JobEndCommand(),
    ]
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
        tool_change="",
        set_speed="",
        travel_move="",
        linear_move="",
        arc_cw="",
        arc_ccw="",
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
    register_dialect(custom_dialect)
    machine.set_dialect_uid(custom_dialect.uid)

    gcode_str, _ = machine.encode_ops(ops, doc)
    gcode_lines = gcode_str.split("\n")

    # --- Assert: WCS should NOT be present ---
    assert "G54" not in gcode_lines
