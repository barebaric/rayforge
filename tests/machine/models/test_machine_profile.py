import pytest
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

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.machine.models.machine import Machine


@pytest.fixture
def carvera_air_machine(context_initializer: "RayforgeContext") -> "Machine":
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
        # WCS is always emitted after the preamble for safety.
        "G54",
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
