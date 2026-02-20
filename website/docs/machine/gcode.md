# G-code Settings

The G-code page in Machine Settings configures how Rayforge generates G-code for your machine.

![G-code Settings](/screenshots/machine-gcode.png)

## G-code Dialect

Select the G-code dialect that matches your controller firmware. Different controllers use slightly different commands and formats.

### Available Dialects

- **GRBL**: Most common for hobby laser cutters. Uses M3/M5 for laser control.
- **Smoothieware**: For Smoothieboard and similar controllers.
- **Marlin**: For Marlin-based controllers.
- **GRBL-compatible**: For controllers that mostly follow GRBL syntax.

:::info
The dialect affects how laser power, movements, and other commands are formatted in the output G-code.
:::

## Custom G-code

You can customize the G-code that Rayforge generates at specific points in the job.

### Program Start

G-code commands executed at the beginning of every job, before any cutting operations.

Common uses:
- Set units (G21 for mm)
- Set positioning mode (G90 for absolute)
- Initialize the machine state

### Program End

G-code commands executed at the end of every job, after all cutting operations.

Common uses:
- Turn off laser (M5)
- Return to origin (G0 X0 Y0)
- Park the head

### Tool Change

G-code commands executed when switching between laser heads (for multi-laser machines).

## See Also

- [G-code Basics](../general-info/gcode-basics) - Understanding G-code
- [G-code Dialects](../reference/gcode-dialects) - Detailed dialect differences
- [Hooks & Macros](hooks-macros) - Custom G-code injection points
