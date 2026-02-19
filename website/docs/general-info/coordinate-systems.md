# Coordinate Systems

Understanding how Rayforge handles coordinate systems is essential for positioning your work correctly.

## Work Coordinate System (WCS) vs Machine Coordinates

Rayforge uses two main coordinate systems:

### Work Coordinate System (WCS)

The WCS is your job's coordinate system. When you position a design at (50, 100) on the canvas, those are WCS coordinates.

- **Origin**: Defined by you (default is G54)
- **Purpose**: Design and job positioning
- **Multiple systems**: G54-G59 available for different setups

### Machine Coordinates

Machine coordinates are absolute positions relative to the machine's home position.

- **Origin**: Machine home (0,0,0) - fixed by hardware
- **Purpose**: Physical positioning on the bed
- **Fixed**: Cannot be changed by software

**Relationship**: WCS offsets define how your job coordinates map to machine coordinates. If G54 offset is (100, 50, 0), then your design at WCS (0, 0) cuts at machine position (100, 50).

## Configuring Coordinates in Rayforge

### Setting the WCS Origin

To position your job on the machine:

1. **Home the machine** first (`$H` command or Home button)
2. **Jog the laser head** to your desired job origin
3. **Set WCS zero** using the Control Panel:
   - Click "Zero X" to set current X as origin
   - Click "Zero Y" to set current Y as origin
4. Your job will now start from this position

### Selecting a WCS

Rayforge supports G54-G59 work coordinate systems:

| System | Use Case |
|--------|----------|
| G54 | Default, primary work area |
| G55-G59 | Additional fixture positions |

Select the active WCS in the Control Panel. Each system stores its own offset from machine origin.

### Y-Axis Direction

Some machines have Y increasing downward instead of upward. Configure this in:

**Settings → Machine → Hardware → Axes**

If your jobs come out mirrored vertically, toggle the Y-axis direction setting.

## Common Issues

### Job in Wrong Position

- **Check WCS offset**: Send `G10 L20 P1` to view G54 offset
- **Verify homing**: Machine must be homed for consistent positioning
- **Check Y-axis direction**: May be inverted

### Coordinates Drift Between Jobs

- **Always home before jobs**: Establishes consistent reference
- **Check for G92 offsets**: Clear with `G92.1` command

---

## Related Pages

- [Work Coordinate Systems (WCS)](work-coordinate-systems) - Managing WCS in Rayforge
- [Control Panel](../ui/control-panel) - Jog controls and WCS buttons
- [Exporting G-code](../files/exporting) - Job positioning options
