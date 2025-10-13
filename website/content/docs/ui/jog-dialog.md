# Jog Dialog

The Jog Dialog provides manual control over your laser cutter's position, allowing you to precisely move the laser head for setup, alignment, and testing purposes.

## Accessing the Jog Dialog

You can access the Jog Dialog in several ways:

- **Menu**: Machine → Jog Controls...
- **Toolbar**: Click the jog button (joystick icon)
- **Keyboard Shortcut**: `Ctrl+J`

!!! note "Connection Required"
    The jog controls are only available when connected to a machine that supports jogging operations.

## Dialog Layout

The jog dialog consists of three main sections:

1. **Homing Controls**: Home individual axes or all axes at once
2. **Manual Jog Controls**: Directional buttons for precise movement
3. **Jog Settings**: Configure speed and distance for jog operations

## Homing Controls

The homing section provides buttons to home your machine's axes:

| Button   | Function       | Description                       |
| -------- | -------------- | --------------------------------- |
| Home X   | Homes X axis   | Moves X axis to its home position |
| Home Y   | Homes Y axis   | Moves Y axis to its home position |
| Home Z   | Homes Z axis   | Moves Z axis to its home position |
| Home All | Homes all axes | Homes all axes simultaneously     |

!!! tip "Homing Sequence"
    It's recommended to home all axes before starting any job to ensure accurate positioning.

## Manual Jog Controls

The jog controls provide a visual interface for moving the laser head:

### Directional Movement

The jog controls are arranged in a grid pattern:

```
  ↖  ↑  ↗
  ←  •  →
  ↙  ↓  ↘
```

| Button           | Movement                        | Keyboard Shortcut |
| ---------------- | ------------------------------- | ----------------- |
| ↑                | Y+ (Y- if machine is Y-flipped) | Up Arrow          |
| ↓                | Y- (Y+ if machine is Y-flipped) | Down Arrow        |
| ←                | X- (left)                       | Left Arrow        |
| →                | X+ (right)                      | Right Arrow       |
| ↖ (top-left)     | X- Y+/- (diagonal)              | -                 |
| ↗ (top-right)    | X+ Y+/- (diagonal)              | -                 |
| ↙ (bottom-left)  | X- Y-/+ (diagonal)              | -                 |
| ↘ (bottom-right) | X+ Y-/+ (diagonal)              | -                 |
| Z+               | Z axis up                       | Page Up           |
| Z-               | Z axis down                     | Page Down         |

!!! note "Focus Required"
    Keyboard shortcuts only work when the jog dialog has focus. Click anywhere in the dialog to ensure it has focus.

### Visual Feedback

The jog buttons provide visual feedback:

- **Normal**: Button is enabled and safe to use
- **Warning (orange)**: Movement would approach or exceed soft limits
- **Disabled**: Movement is not supported by the machine or machine is not connected

## Jog Settings

Configure the behavior of jog operations:

### Jog Speed

- **Range**: 1-10,000 mm/min
- **Default**: 1,000 mm/min
- **Purpose**: Controls how fast the laser head moves during jog operations

!!! tip "Speed Selection"
    - Use lower speeds (100-500 mm/min) for precise positioning
    - Use higher speeds (1,000-3,000 mm/min) for larger movements
    - Very high speeds may cause missed steps on some machines

### Jog Distance

- **Range**: 0.1-1,000 mm
- **Default**: 10.0 mm
- **Purpose**: Controls how far the laser head moves with each button press

!!! tip "Distance Selection"
    - Use small distances (0.1-1.0 mm) for fine-tuning
    - Use medium distances (5-20 mm) for general positioning
    - Use large distances (50-100 mm) for quick repositioning

## Machine Compatibility

The jog dialog adapts to your machine's capabilities:

### Axis Support

- **X/Y Axis**: Supported by virtually all laser cutters
- **Z Axis**: Only available on machines with Z-axis control
- **Diagonal Movement**: Requires support for both X and Y axes

### Machine Types

| Machine Type       | Jog Support | Notes                     |
| ------------------ | ----------- | ------------------------- |
| GRBL (v1.1+)       | Full        | Supports all jog features |
| Smoothieware       | Full        | Supports all jog features |
| Custom Controllers | Variable    | Depends on implementation |

## Safety Features

### Soft Limits

When soft limits are enabled in your machine profile:

- Buttons show orange warning when approaching limits
- Movement is automatically limited to prevent exceeding bounds
- Provides visual feedback to prevent crashes

### Connection Status

- All controls are disabled when not connected to a machine
- Buttons update sensitivity based on machine state
- Prevents accidental movement during operation

## Workflow Tips

### Initial Setup

1. **Connect to Machine**: Ensure proper connection
2. **Home All Axes**: Establish reference position
3. **Set Jog Speed**: Choose appropriate speed for task
4. **Set Jog Distance**: Choose appropriate distance for task

### Precision Positioning

1. Use large distance for rough positioning
2. Reduce distance for fine-tuning
3. Use keyboard arrows for precise adjustments
4. Watch for warning indicators near limits

### Testing and Alignment

1. Position laser over test area
2. Use low speeds for test cuts
3. Verify alignment with workpiece
4. Adjust position as needed

## Troubleshooting

### Jog Controls Not Working

**Possible Causes:**

- Machine not connected
- Machine doesn't support jogging
- Machine is in alarm or error state
- Soft limits preventing movement

**Solutions:**

- Check connection status in main window
- Verify machine supports jog commands
- Reset machine if in alarm state
- Check soft limit configuration

### Keyboard Shortcuts Not Responding

**Possible Causes:**

- Dialog doesn't have focus
- Another window is active
- System shortcuts intercepting keys

**Solutions:**

- Click on the jog dialog to give it focus
- Close other windows that might capture keys
- Check system keyboard shortcut settings

### Movement Direction Reversed

**Possible Causes:**

- Y-axis direction configured incorrectly
- Machine orientation different from expected

**Solutions:**

- Check Y-axis direction in machine profile
- Adjust "Y-axis down" setting if needed
- Test with small movements first

---

**Related Pages:**

- [Machine Setup](../machine/index.md) - Configure your machine
- [Keyboard Shortcuts](../reference/shortcuts.md) - Complete shortcut reference
- [Main Window](main-window.md) - Main interface overview
- [Machine Settings](../machine/device-config.md) - Device configuration
