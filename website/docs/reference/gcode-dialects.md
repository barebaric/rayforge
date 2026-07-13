---
description: "G-code dialects supported by Rayforge. Configure output for different GRBL versions and Smoothieware-compatible laser controllers."
---

# G-code Dialect Support

Rayforge supports multiple G-code dialects to work with different controller
firmware.

## Supported Dialects

Rayforge currently supports these G-code dialects:

| Dialect                        | Firmware     | Common Use                  |
| ------------------------------ | ------------ | --------------------------- |
| **Grbl (Compat)**              | GRBL 1.1+    | Diode lasers, hobby CNC     |
| **Grbl (Compat, no Z axis)**   | GRBL 1.1+    | 2D laser cutters without Z  |
| **Grbl Raster**                | GRBL 1.1+    | Optimized for raster work   |
| **GRBL Dynamic (Depth-Aware)** | GRBL 1.1+    | Depth-Aware laser engraving |
| **GRBL Dynamic (no Z axis)**   | GRBL 1.1+    | Depth-Aware laser engraving |
| **LinuxCNC**                   | LinuxCNC     | Native Bézier (G5) support  |
| **Mach4 (M67 Analog)**         | Mach4        | High-speed raster engraving |
| **Smoothieware**               | Smoothieware | Laser cutters, CNC          |
| **Marlin**                     | Marlin 2.0+  | 3D printers with laser      |

:::note Recommended Dialects
:::

**Grbl (Compat)** is the most tested and recommended dialect for standard laser
applications.

**Grbl Raster** is optimized for raster engraving on GRBL controllers. It keeps
the laser in dynamic power mode (M4) continuously and omits redundant feedrate
commands, resulting in smoother and more compact G-code output.

**GRBL Dynamic (Depth-Aware)** is recommended for Depth-Aware laser engraving
where power varies during cuts (e.g., variable depth engraving).

**LinuxCNC** supports native cubic Bézier curves through the G5 command, which
produces very smooth and compact G-code for curved paths. When using this
dialect, enable the "Support Bézier Curves" option in Advanced Machine Settings
to take advantage of G5 output.

---

## Mach4 (M67 Analog)

The **Mach4 (M67 Analog)** dialect is designed for high-speed raster engraving
with Mach4 controllers. It uses the M67 command with analog output for precise
laser power control.

### Key Features

- **M67 Analog Output**: Uses `M67 E0 Q<0-255>` for laser power instead of
  inline S commands
- **Reduced Buffer Pressure**: By separating power commands from motion
  commands, the controller buffer is less stressed during high-speed operations
- **High-Speed Raster**: Optimized for fast raster engraving operations

### When to Use

Use this dialect when:

- You have a Mach4 controller with analog output capability
- You need high-speed raster engraving
- Your controller experiences buffer overflow with standard inline S commands

### Command Format

The dialect generates G-code like:

```gcode
M67 E0 Q127  ; Set laser power to 50% (127/255)
G1 X100 Y200 F1000  ; Move to position
M67 E0 Q0    ; Turn laser off
```

---

## Creating a Custom Dialect

To create a custom G-code dialect based on a built-in dialect:

1. Open **Machine Settings** → **G-code Dialect**
2. Click the **Copy** icon on a built-in dialect to create a new custom dialect
3. Edit the dialect settings as needed
4. Save your custom dialect

Each custom dialect is an independent copy. Changing one dialect never affects
others, so you can freely experiment without worrying about breaking an existing
setup. Custom dialects are stored in your configuration directory and can be
shared.

### Dialect Settings

When editing a custom dialect, the Settings page offers these options:

**Continuous Laser Mode** keeps the laser in dynamic power mode (M4) active
throughout the entire job instead of toggling M4/M5 between segments. This is
useful for raster engraving where the laser needs to stay on continuously
during scan lines.

**Modal Feedrate** omits the feedrate parameter (F) from motion commands when
it has not changed since the last command. This produces more compact G-code
and reduces the amount of data sent to the controller.

### Separate Laser-On Command for Focusing

Some dialects support configuring a separate command for turning the laser on at
low power, which is useful for focus mode. This lets you use a different command
for the visual "laser pointer" behavior than what is used during actual cutting
or engraving. Check your dialect's settings page for this option.

---

## Template Placeholders

When creating or editing a custom dialect, each command template uses
[Python format string](https://docs.python.org/3/library/string.html#format-string-syntax)
placeholders to inject dynamic values. Use `{name}` or `{name:.0f}` syntax
(e.g., `{power:.0f}` to format as a decimal with no fractional digits).

### Available Placeholders by Template

| Template           | Placeholders                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| **Laser On**       | `power`                                                                                                      |
| **Focus Laser On** | `power`                                                                                                      |
| **Laser Off**      | _(none)_                                                                                                     |
| **Tool Change**    | `tool_number`                                                                                                |
| **Set Speed**      | `speed`                                                                                                      |
| **Travel Move**    | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`                              |
| **Linear Move**    | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `power`           |
| **Arc (CW)**       | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `power`           |
| **Arc (CCW)**      | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `power`           |
| **Bezier Cubic**   | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `p`, `q`, `power` |
| **Air On/Off**     | _(none)_                                                                                                     |
| **Home All**       | _(none)_                                                                                                     |
| **Home Axis**      | `axis_letter`                                                                                                |
| **Move To**        | `speed`, `x`, `y`, `z`                                                                                       |
| **Jog**            | `speed`                                                                                                      |
| **Clear Alarm**    | _(none)_                                                                                                     |
| **Set WCS Offset** | `p_num`, `x`, `y`, `z`                                                                                       |
| **Probe Cycle**    | `axis_letter`, `max_travel`, `feed_rate`                                                                     |
| **Dwell**          | `seconds`, `milliseconds`                                                                                    |

### Placeholder Reference

#### Coordinates

| Placeholder | Description                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------------------- |
| `x`         | Target X coordinate as a float (e.g., `100.0`)                                                                |
| `y`         | Target Y coordinate as a float (e.g., `200.0`)                                                                |
| `z`         | Target Z coordinate as a float (e.g., `5.0`)                                                                  |
| `x_cmd`     | X-axis command string, e.g., `" X100.0"`. Omitted when unchanged (if "Omit unchanged coordinates" is enabled) |
| `y_cmd`     | Y-axis command string, e.g., `" Y200.0"`. Omitted when unchanged                                              |
| `z_cmd`     | Z-axis command string, e.g., `" Z5.0"`. Omitted when unchanged                                                |
| `extra_cmd` | Extra axes command string for A, B, C axes (e.g., `" A90.0"`). Empty if no extra axes are configured          |

#### Motion

| Placeholder | Description                                                                                           |
| ----------- | ----------------------------------------------------------------------------------------------------- |
| `f_command` | Feedrate command string, e.g., `" F3000"`. Omitted when modal and unchanged                           |
| `s_command` | Spindle/power command string, e.g., `" S500"`. Used in dynamic/raster modes and continuous laser mode |
| `i`         | Arc or Bézier control point X offset from start position                                              |
| `j`         | Arc or Bézier control point Y offset from start position                                              |
| `p`         | Second Bézier control point X offset from end position (Bezier Cubic only)                            |
| `q`         | Second Bézier control point Y offset from end position (Bezier Cubic only)                            |

#### Power and Speed

| Placeholder   | Description                                                                                         |
| ------------- | --------------------------------------------------------------------------------------------------- |
| `power`       | Absolute laser power value (float). Supports format specifiers, e.g., `{power:.0f}` for no decimals |
| `speed`       | Speed value (for Move To and Jog commands)                                                          |
| `tool_number` | Tool/laser head number                                                                              |

#### Machine and Probing

| Placeholder   | Description                                                             |
| ------------- | ----------------------------------------------------------------------- |
| `axis_letter` | Single axis letter, e.g., `"X"`, `"Y"`, `"Z"` (for Home Axis and Probe) |
| `p_num`       | WCS P-number (e.g., `1` for G54)                                        |
| `max_travel`  | Maximum probe travel distance (Probe Cycle only)                        |
| `feed_rate`   | Probe feedrate (Probe Cycle only)                                       |

#### Dwell

| Placeholder    | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| `seconds`      | Dwell duration in seconds as a float (e.g., `1.5`)          |
| `milliseconds` | Dwell duration in milliseconds as an integer (e.g., `1500`) |

### Tips

- **Format specifiers** are supported: `{power:.0f}` formats power as an integer,
  `{power:.2f}` as two decimal places.
- The **"Omit unchanged coordinates"** setting controls whether `x_cmd`, `y_cmd`,
  and `z_cmd` are left empty when the axis position has not changed since the
  last command. This reduces G-code size.
- The **"Modal Feedrate"** setting controls whether `f_command` is omitted when
  the feedrate has not changed.
- Leave a template field **empty** to skip that command entirely (e.g., setting
  `bezier_cubic` to `""` disables native Bézier output and falls back to
  linearization).

---

## Related Pages

- [Exporting G-code](../files/exporting.md) - Export settings
- [Firmware Compatibility](firmware) - Firmware versions
- [Device Settings](../machine/device.md) - GRBL configuration
- [Macros & Hooks](../machine/hooks-macros.md) - Custom G-code injection
