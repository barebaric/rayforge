# Advanced Settings

The Advanced page in Machine Settings contains additional configuration options for specialized use cases.

![Advanced Settings](/screenshots/machine-advanced.png)

## Connection Behavior

Settings that control how Rayforge interacts with your machine during connection.

### Home on Connect

When enabled, Rayforge automatically sends a homing command ($H) when connecting to the machine.

- **Enable if**: Your machine has reliable limit switches
- **Disable if**: Your machine has no limit switches or homing is unreliable

### Clear Alarms on Connect

When enabled, Rayforge automatically clears any alarm state when connecting.

- **Enable if**: Your machine frequently starts in alarm state
- **Disable if**: You want to manually investigate alarms before clearing

## Flip Axes

These settings invert the direction of axis movements.

### Flip X Axis

Inverts the X-axis direction. When enabled, positive X moves left instead of right.

### Flip Y Axis

Inverts the Y-axis direction. When enabled, positive Y moves down instead of up.

:::info
Flip axes are useful when:
- Your machine's coordinate system doesn't match expected behavior
- You've wired your motors in reverse
- You want to match the behavior of another machine
:::

## See Also

- [Hardware Settings](hardware) - Axis origin configuration
- [Device Settings](device) - GRBL axis direction settings
