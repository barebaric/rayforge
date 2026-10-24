---
description:
  "Advanced machine settings in Rayforge — custom G-code, homing, and fine-tuning for experienced
  laser cutter users."
---

# Advanced Settings

The Advanced page in Machine Settings contains additional configuration options for specialized use
cases.

![Advanced Settings](/screenshots/machine-settings-advanced.webp)

## Connection Behavior

Settings that control how Rayforge interacts with your machine during connection.

### Home on Connect

When enabled, Rayforge automatically sends a homing command ($H) when connecting to the machine.

- **Enable if**: Your machine has reliable limit switches
- **Disable if**: Your machine has no limit switches or homing is unreliable

### Park After Homing

When enabled, Rayforge moves the head to the configured park position (a machine-coordinate X/Y
pair) after homing completes. The move uses the "Move To Park" command template (default
`G53 G0 X{x} Y{y}`, editable in Machine Settings → G-code), so it is independent of the active
work coordinate system. It only runs after homing all axes.

- **Enable if**: You want the head at a known spot after homing, or homing leaves it parked outside
  the soft-limit range. Some firmwares (e.g. the Creality Falcon series) retract past the machine
  origin during the homing pull-off, which makes the first jog fail with a soft-limit error
  ("error:15") until a diagonal jog moves the head back inside — parking at (0, 0) fixes that
- **Disable if**: Your machine parks safely inside its travel range after homing, or a fixture is
  mounted at the park position

### Clear Alarms on Connect

When enabled, Rayforge automatically clears any alarm state when connecting.

- **Enable if**: Your machine frequently starts in alarm state
- **Disable if**: You want to manually investigate alarms before clearing

### Allow Single Axis Homing

When enabled, you can home individual axes independently (X, Y, or Z) rather than requiring all axes
to home together. This is useful for machines where one axis may already be positioned correctly.

### Z Homing

Chooses how the Z axis finds its reference position when you home it. This setting only appears when
the machine's driver supports more than one method.

- **Home Switch**: Z moves to its home switch, like the X and Y axes
- **Probe**: Z uses the controller's auto-focus probe to find the work surface, then moves to the
  configured focus distance

## Arc and Curve Settings

Settings for controlling how curved paths are converted to G-code movements.

### Support Arcs

When enabled, Rayforge generates arc commands (G2/G3) for curved paths instead of breaking them into
many small linear moves. This produces more compact G-code and smoother motion on most controllers.

When disabled, all curves are converted to linear segments (G1 commands), which provides maximum
compatibility with controllers that don't support arcs.

### Support Bézier Curves

When enabled, Rayforge generates native cubic Bézier commands (such as the G5 command used by
LinuxCNC) for curved paths. This produces very smooth motion and compact G-code on controllers that
support it. You should disable this setting if your machine's firmware does not understand Bézier
commands, in which case the curves will be broken down into linear segments instead.

### Arc and Curve Tolerance

This setting controls the maximum allowed deviation when fitting arcs and curves to curved paths,
specified in millimeters. A smaller value produces more accurate paths but may require more
commands. A larger value allows more deviation but generates fewer commands.

Typical values range from 0.01mm for precision work to 0.1mm for faster processing.

## Bidirectional Scan Offset by Speed

The Bidirectional Scan Offset by Speed table corrects the horizontal misalignment between
left-to-right and right-to-left raster passes. On many machines this skew is not constant: the
faster the laser head moves, the further the right-to-left passes drift out of alignment, which
shows up as banding that gets worse at higher engraving speeds.

![Bidirectional Scan Offset by Speed](/screenshots/machine-settings-advanced-bidir-offset-table.webp)

Each row of the table pairs an engraving speed with the offset, in millimeters, that right-to-left
passes need at that speed:

- Use **Add Row** to add a row and the trash button on a row to remove it
- Positive values shift right-to-left passes to the right, negative values to the left
- Rows are kept sorted by speed; if two rows have the same speed, the last one wins

When an engrave step runs, Rayforge looks up the step's speed in this table and interpolates the
offset linearly between the surrounding rows. Speeds below the first row or above the last row use
the nearest row's offset, and an empty table means every speed uses an offset of 0.

**Without correction** — the misalignment grows with the engraving speed:

![Raster passes misaligned at 1000, 4000 and 8000 mm/min](/images/bidir-scan-offset-uncorrected.webp)

**With the offset table applied:**

![Raster passes aligned at every speed](/images/bidir-scan-offset-table-applied.webp)

The table only applies to engrave steps whose own **Bidirectional Scan Offset** is set to 0; a
non-zero per-step offset always takes precedence. See
[Bidirectional Scan Offset](../features/operations/engrave.md#bidirectional-scan-offset) for how the
misalignment looks and how to calibrate it.

## See Also

- [Hardware Settings](hardware) - Axis origin and flip settings
- [Device Settings](device) - GRBL-specific settings
