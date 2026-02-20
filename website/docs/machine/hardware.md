# Hardware Settings

The Hardware page in Machine Settings configures the physical dimensions and axis orientation of your machine.

![Hardware Settings](/screenshots/machine-hardware.png)

## Work Area Dimensions

Define the size of your machine's working area in millimeters.

### Width

The horizontal (X-axis) dimension of your working area.

- Measure the actual cutting area, not the machine exterior
- Account for any obstructions or limits
- Example: 400mm for a typical K40 laser

### Height

The vertical (Y-axis) dimension of your working area.

- Measure the actual cutting area
- Consider the laser head travel limits
- Example: 300mm for a typical K40 laser

:::tip Measuring Work Area
To find your true work area:
1. Home the machine
2. Manually jog to the maximum X and Y positions
3. Measure from the homing corner to the maximum reach
:::

## Coordinate Origin

Select where your machine's coordinate origin (0,0) is located. This determines how coordinates are interpreted.

### Available Options

- **Bottom Left**: Most common for GRBL devices. X increases to the right, Y increases upward.
- **Top Left**: Common for some CNC-style machines. X increases to the right, Y increases downward.
- **Top Right**: X increases to the left, Y increases downward.
- **Bottom Right**: X increases to the left, Y increases upward.

### Finding Your Origin

1. Home your machine using the Home button
2. Observe where the laser head moves to
3. That position is your (0,0) origin

:::info
The coordinate origin setting affects how G-code is generated. Make sure it matches your firmware's homing configuration.
:::

## See Also

- [General Settings](general) - Machine name and speed settings
- [Device Settings](device) - GRBL homing and axis settings
