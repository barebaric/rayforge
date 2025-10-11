# Custom G-code Automation

Learn advanced macro and hook workflows to automate repetitive tasks and customize your laser cutter's behavior.

## Overview

Rayforge supports custom G-code through macros and hooks, allowing you to automate complex sequences, add custom commands, and integrate with external systems.

## Prerequisites

- Basic understanding of G-code
- Familiarity with your machine's command set
- Knowledge of [G-code Macros & Hooks](../features/macros-hooks.md)

## Use Cases

- **Pre-job routines**: Homing, air assist check, focus verification
- **Post-job actions**: Return to origin, air assist off, status reporting
- **Custom operations**: Multi-tool changes, material probing, camera capture
- **Integration**: Send notifications, log jobs, update external systems

## Creating Macros

### Basic Macro Structure

Macros are reusable G-code sequences:

```gcode
; Macro: Home and Center
G28 X Y          ; Home X and Y axes
G0 X150 Y150     ; Move to center position
M3 S0            ; Laser off
```

### Macro Best Practices

- **Add comments** to explain each command
- **Test incrementally** with safe movements
- **Include safety checks** (laser off, safe Z-height)
- **Make macros reusable** with parameters where possible

### Common Macro Examples

#### Pre-Job Homing Sequence
```gcode
G28 X Y Z        ; Home all axes
G0 Z10           ; Safe Z height
M3 S0            ; Ensure laser is off
```

#### Material Thickness Probe
```gcode
; Probe material surface
G38.2 Z-10 F50   ; Probe down slowly
G10 L20 Z0       ; Set work zero at surface
G0 Z5            ; Lift to safe height
```

#### Job Complete Routine
```gcode
M5               ; Laser off
M9               ; Air assist off
G0 X0 Y0         ; Return to origin
M117 Job Complete ; Display message
```

## Using Hooks

Hooks automatically run G-code at specific events.

### Available Hooks

- **Pre-Job**: Runs before starting a job
- **Post-Job**: Runs after job completion
- **Pre-Operation**: Runs before each operation
- **Post-Operation**: Runs after each operation
- **Emergency Stop**: Runs when emergency stop is triggered

### Hook Configuration

1. **Open Settings** → **Macros & Hooks**
2. **Select hook type** from dropdown
3. **Enter G-code** or reference a macro
4. **Test the hook** with a small job
5. **Save configuration**

### Hook Examples

#### Pre-Job Hook with Air Assist Check
```gcode
G28 X Y          ; Home X and Y
M8               ; Turn on air assist
G4 P2            ; Wait 2 seconds
M117 Check Air   ; Prompt user to verify
```

#### Multi-Tool Selection Hook
```gcode
; Change to high-power laser
T1               ; Select tool 1
M6               ; Tool change
G43 H1           ; Apply tool offset
```

#### Post-Job Notification
```gcode
M5               ; Laser off
M9               ; Air assist off
G0 X0 Y0         ; Home position
M117 Done        ; Status message
```

## Advanced Techniques

### Conditional Execution

Some controllers support conditional commands:

```gcode
; Example: Only run if value is true
O100 if [#<_value> GT 0]
  M3 S100        ; Turn on laser
O100 endif
```

### Subroutines

Create reusable subroutines for complex operations:

```gcode
; Subroutine definition
O100 sub
  G0 Z5          ; Lift
  M5             ; Laser off
  M9             ; Air off
O100 endsub

; Call subroutine
O100 call
```

### External Integration

Use custom commands to integrate with external systems:

```gcode
M118 Job:START   ; Send message to console
; ... job operations ...
M118 Job:END     ; Send completion message
```

## Debugging and Testing

### Testing Macros Safely

1. **Test without laser power**: Use S0 or M5
2. **Test with dry run**: Use simulation mode
3. **Verify movements** are within bounds
4. **Monitor console output** for errors

### Common Issues

**Macro doesn't run**: Check syntax and controller compatibility

**Unexpected behavior**: Verify modal state (G90/G91, inches/mm)

**Timing issues**: Add dwell commands (G4) where needed

**Controller errors**: Consult GRBL or controller documentation

## Best Practices

- **Keep macros simple** and focused on single tasks
- **Document your code** with clear comments
- **Test thoroughly** before production use
- **Back up your configurations** regularly
- **Share useful macros** with the community

## Safety Considerations

- **Always include laser-off** commands (M5)
- **Verify safe Z-heights** before moves
- **Test in simulation mode** first
- **Never bypass safety features**
- **Monitor first run** of any new macro

## Related Topics

- [G-code Macros & Hooks](../features/macros-hooks.md)
- [G-code Dialect Support](../reference/gcode-dialects.md)
- [GRBL Settings](../machine/grbl-settings.md)
