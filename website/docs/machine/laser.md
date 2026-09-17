---
description:
  "Configure laser settings in Rayforge — set power, speed, and mode for your laser tube or diode
  for optimal cutting and engraving results."
---

# Laser Settings

The Laser page in Machine Settings configures your laser head(s) and their properties.

![Laser Settings](/screenshots/machine-settings-laser.webp)

## Laser Heads

Rayforge supports machines with multiple laser heads. Each laser head has its own configuration.

### Adding a Laser Head

Click the **Add Laser** button to create a new laser head configuration.

### Laser Head Properties

Each laser head has the following settings:

#### Name

A descriptive name for this laser head.

Examples:

- "10W Diode"
- "CO2 Tube"
- "Infrared Laser"

#### Tool Number

The tool index for this laser head. Used in G-code with the T command.

- Single-head machines: Use 0
- Multi-head machines: Assign unique numbers (0, 1, 2, etc.)

#### Maximum Power

The maximum power value for your laser.

- **GRBL typical**: 1000 (S0-S1000 range)
- **Some controllers**: 255 (S0-S255 range)
- **Percentage mode**: 100 (S0-S100 range)

This value should match your firmware's $30 setting.

#### Frame Power

The power level used for framing operations (outlining without cutting).

- Set to 0 to disable framing
- Adjust based on your laser and material

#### Frame Speed

The speed at which the laser head moves during framing. This is set per laser head, so if your
machine has multiple lasers with different characteristics you can choose an appropriate speed for
each one. Slower speeds make the frame path easier to follow by eye.

#### Focus Power

The power level used when focus mode is enabled. Focus mode turns on the laser at low power to act
as a "laser pointer" for positioning.

- Set to 0 to disable the focus mode feature
- Use for visual alignment and positioning

<!-- prettier-ignore-start -->
:::tip[Using Focus Mode]
Click the focus button (laser icon) in the toolbar to toggle focus mode. The
laser will turn on at this power level, helping you see exactly where the laser is positioned. See
[Workpiece Positioning](../features/workpiece-positioning.md) for more information.
:::
<!-- prettier-ignore-end -->

#### Spot Size

The physical size of your focused laser beam in millimeters.

- Enter both X and Y dimensions
- Most lasers have a circular spot (e.g., 0.1 x 0.1)
- Affects engraving quality calculations

<!-- prettier-ignore-start -->
:::tip[Measuring Spot Size]
To measure your spot size:

1. Fire a short pulse at low power on a test material
2. Measure the resulting mark with calipers
3. Use the average of multiple measurements
:::
<!-- prettier-ignore-end -->

#### Color

The color used to display this laser's operations (cuts and engraving) in the canvas and 3D preview.
This helps you visually distinguish which laser will perform each operation when working with
multiple laser heads.

- Click the color swatch to open a color picker
- Choose a color that contrasts well with your material preview
- Default colors are assigned automatically

<!-- prettier-ignore-start -->
:::tip[Multi-Laser Workflows]
When using multiple laser heads, assigning different colors to each
laser makes it easy to see which operations will be performed by which laser. For example, use red
for your main cutting laser and blue for a secondary engraving laser.
:::
<!-- prettier-ignore-end -->

#### Laser Type

Choose the type of laser head from the dropdown:

- **Diode**: Standard diode lasers (most common for hobbyist machines)
- **CO2**: CO2 tube lasers
- **Fiber**: Fiber lasers

When CO2 or Fiber is selected, additional **PWM settings** become available (see below). For diode
lasers, the PWM section is hidden since it does not apply.

The laser type also sets a default **wavelength** (used by the physical burn model) when no explicit
value is entered below.

#### Wavelength (nm)

The emission wavelength of your laser, in nanometers. This feeds the
[physical burn model](../ui/3d-preview.md#physical-burn-model) in the 3D preview: together with the
material's [absorption](../application-settings/materials.md#absorption) data, it determines how
much laser energy the stock absorbs.

When set to 0, Rayforge falls back to the typical wavelength for the selected laser type (e.g. 445
nm for diode, 1064 nm for fiber, 10600 nm for CO2).

#### Max Optical Power (W)

The optical output power of your laser at full power, in watts. This is the actual light output, not
the electrical input. Together with the spot size and scan speed, it determines the fluence (J/cm²)
used by the [physical burn model](../ui/3d-preview.md#physical-burn-model).

When set to 0, a mid-range desktop default is used.

#### PWM Settings

When a CO2 or Fiber laser type is selected, the following PWM controls appear:

- **PWM Frequency**: The default PWM frequency in Hz for this laser head. Typical values range from
  500 Hz to several kHz depending on your controller and power supply.
- **Max PWM Frequency**: The upper limit for the frequency setting. This prevents entering values
  that your hardware cannot handle.
- **Pulse Width**: The default pulse width in microseconds. This controls how long each pulse is on
  during a cycle.
- **Min/Max Pulse Width**: Bounds for the pulse width setting.

These defaults carry through to your operation steps, where they can be overridden per step if
needed.

#### Pointer Offset

If your machine has a separate pointer laser (a small red dot laser) mounted at a fixed distance
from the cutting beam, you can tell Rayforge about that distance so it can compensate for it.

- **Use Pointer Offset**: Enables the compensation. Off by default.
- **Pointer Offset X / Y**: The distance from the cutting beam spot to the pointer dot, in
  millimeters, along the machine X and Y axes.

When enabled, three things change:

1. **Set Work Zero at Current Position** (and the Zero X / Zero Y buttons) places the work origin
   where the _pointer dot_ marks the stock, not where the (invisible) cutting beam is.
2. The canvas shows a yellow pointer dot next to the red beam dot, marking where the pointer dot is
   on your material.
3. A **Pointer Alignment** switch becomes available in the move-to popover (see below).

#### Pointer Alignment

Pointer alignment is a runtime switch in the move-to popover (the compass icon next to the position
readout). While it is on, all absolute aiming operations — Move-To, the corner shortcuts, moving to
the WCS origin, Click-to-Move, Move-Head-Here, and framing — are shifted so the _pointer dot_ lands
on the aimed position. The canvas pointer dot is drawn filled while alignment is on and hollow while
it is off.

The typical workflow:

1. Jog the machine until the pointer dot marks your reference point on the stock.
2. **Set Work Zero** there — with the pointer offset enabled, the origin lands exactly where the
   pointer pointed.
3. Turn on **Pointer Alignment** in the move-to popover.
4. Frame and move with the pointer dot: everything you aim at is marked by the pointer.
5. When you press **Send**, a warning reminds you that the job burns with the beam at the WCS
   positions — you can turn alignment off and burn, burn anyway, or cancel.

Two things are never shifted: **jog** (a relative move needs no compensation) and **jobs** — cutting
always happens with the beam at the WCS positions, so your G-code output is identical whether
alignment is on or off. Together with zeroing by the pointer dot this stays consistent: the origin
sits at `beam + offset`, aiming shifts every target by `-offset`, and the burn is unshifted.

Pointer alignment is a session-only setting: it is not saved to the machine profile and resets when
you switch machines.

<!-- prettier-ignore-start -->
:::tip[Measuring the Offset]
1. Jog the machine until the pointer dot marks a visible point on the stock.
2. Turn on [focus mode](#focus-power) and jog until the *cutting beam* burns a mark on exactly the
   same point (or carefully move the beam there at focus power).
3. The offset is the pointer position minus the beam position. For example, if the pointer marked
   X=100 and you had to jog the beam to X=88 to hit the same spot, enter X = 12.0 — the pointer dot
   sits 12 mm ahead of the beam.

If a test cut comes out shifted, flip the sign of the corresponding axis.
:::
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
:::note[Rotary Mode]
When the rotary attachment is active, the Y axis is replaced by the rotary roller, so the Y
component of the pointer offset does not apply meaningfully. Set it to 0 for rotary jobs.
:::
<!-- prettier-ignore-end -->

#### 3D Model

Each laser head can have a 3D model assigned to it. This model is rendered in the
[3D view](../ui/3d-preview.md) and follows the toolpath during simulation.

Click the model selection row to browse available models. Once a model is selected, you can adjust
its scale, rotation (X/Y/Z), and focal distance to match your physical laser head.

## See Also

- [Device Settings](device) - GRBL laser mode settings
- [Workpiece Positioning](../features/workpiece-positioning.md) - Using focus mode and other
  positioning methods
