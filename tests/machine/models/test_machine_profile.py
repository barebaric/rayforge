import pytest
import pytest_asyncio

from typing import TYPE_CHECKING
from rayforge.core.doc import Doc
from rayforge.core.ops import Ops
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.profile import MachineProfile, PROFILES
from rayforge.shared import tasker

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext


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
async def test_profile_without_rotary_modules(
    context_initializer: "RayforgeContext",
):
    """Profiles without rotary_modules create machines with none."""
    profile = MachineProfile(name="No Rotary Test")
    machine = profile.create_machine(context_initializer)
    assert machine.rotary_modules == {}


@pytest.mark.asyncio
async def test_profile_with_rotary_modules(
    context_initializer: "RayforgeContext",
):
    """Profile with rotary_modules creates machine with those modules."""
    profile = MachineProfile(
        name="Rotary Test",
        rotary_modules=[
            {
                "name": "Chuck A",
                "model_id": "rotary/standard.glb",
            },
            {
                "name": "Chuck B",
            },
        ],
    )
    machine = profile.create_machine(context_initializer)
    assert len(machine.rotary_modules) == 2

    modules = list(machine.rotary_modules.values())
    names = {m.name for m in modules}
    assert names == {"Chuck A", "Chuck B"}

    chuck_a = next(m for m in modules if m.name == "Chuck A")
    assert chuck_a.model_id == "rotary/standard.glb"

    chuck_b = next(m for m in modules if m.name == "Chuck B")
    assert chuck_b.model_id is None


@pytest.mark.asyncio
async def test_profile_rotary_modules_have_unique_uids(
    context_initializer: "RayforgeContext",
):
    """Each rotary module from a profile gets a unique UID."""
    profile = MachineProfile(
        name="UID Test",
        rotary_modules=[
            {"name": "Module 1"},
            {"name": "Module 2"},
        ],
    )
    machine = profile.create_machine(context_initializer)
    uids = list(machine.rotary_modules.keys())
    assert len(uids) == 2
    assert uids[0] != uids[1]


@pytest.mark.asyncio
async def test_builtin_profiles_have_no_rotary_modules():
    """Built-in profiles do not define rotary_modules yet."""
    for profile in PROFILES:
        assert profile.rotary_modules is None


@pytest.mark.asyncio
async def test_profile_with_rotary_and_model_id(
    context_initializer: "RayforgeContext",
):
    """Verify model_id flows through profile → machine → serialization."""
    profile = MachineProfile(
        name="Model ID Test",
        rotary_modules=[
            {
                "name": "Rotary Axis",
                "model_id": "rotary/carvera_standard.glb",
            },
        ],
    )
    machine = profile.create_machine(context_initializer)
    assert len(machine.rotary_modules) == 1
    rm = list(machine.rotary_modules.values())[0]
    assert rm.model_id == "rotary/carvera_standard.glb"
