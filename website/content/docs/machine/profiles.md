# Machine Profiles

Machine profiles are pre-configured templates that contain all the settings needed to connect to and operate your laser cutter. They make setup quick and easy by providing default values for your specific hardware.

## Overview

Machine profiles contain:

- **Device driver**: The communication method (Serial, Network, WiFi)
- **Machine dimensions**: Working area size (width  height)
- **Speeds**: Maximum travel and cutting speeds
- **G-code dialect**: Command format (GRBL, Smoothie, etc.)
- **Laser head configuration**: Power ranges, spot size, framing settings
- **Motion settings**: Coordinate system orientation
- **Optional features**: Home-on-start, alarm clearing, hook scripts

<!-- SCREENSHOT
id: ui-machine-profile-selection
type: screenshot
size: dialog
description: |
  Machine profile selection dialog showing:
  - List of available profiles (Sculpfun iCube, Sculpfun S30, Other Device)
  - Profile details preview
  - Create/Edit/Delete buttons
setup:
  - action: open_preferences
  - action: navigate_to
    section: machine
  - action: open_profile_selector
  - action: capture
    region: dialog
filename: UI-MachineProfile-Selection.png
alt: "Machine profile selection dialog"
-->

<!-- ![Machine profile selection](../images/UI-Machine-Profile-Selection.png) -->

## Using Built-in Profiles

Rayforge includes profiles for common laser machines. These provide tested default settings to get you running quickly.

### Available Profiles

**Sculpfun iCube:**
- Small desktop laser engraver
- Working area: 120mm  120mm
- GRBL-based control
- Serial connection
- Includes homing support

**Sculpfun S30:**
- Medium-format laser cutter
- Working area: 400mm  400mm
- GRBL-based control
- Serial connection
- Higher power capabilities

**Other Device:**
- Generic GRBL-compatible machine
- Customizable dimensions
- Use this as a starting point for unlisted machines
- Requires manual configuration

### Selecting a Profile

When you first launch Rayforge:

1. Profile selection dialog appears automatically
2. Choose your machine from the list
3. Rayforge applies all settings from the profile
4. Connection to device begins automatically

**To change profiles later:**

- **Menu:** Preferences  Machine  Change Profile
- Or create a new machine instance for different hardware

<!-- SCREENSHOT
id: ui-first-launch-profile-select
type: screenshot
size: full-window
description: |
  First launch screen showing profile selection prompt
setup:
  - action: simulate_first_launch
  - action: capture
    region: window
filename: UI-FirstLaunch-Profile.png
alt: "First launch profile selection screen"
-->

<!-- ![First launch profile selection](../images/UI-First-Launch-Profile.png) -->

## Creating Custom Profiles

For machines not in the built-in list, create a custom profile with your specific settings.

### Step 1: Start with a Base Profile

1. Select "Other Device" as your starting point
2. This provides a generic GRBL configuration
3. You'll customize it to match your hardware

### Step 2: Configure Basic Settings

<!-- SCREENSHOT
id: ui-machine-basic-settings
type: screenshot
size: dialog
description: |
  Machine configuration dialog showing:
  - Machine name field
  - Driver selection dropdown
  - Dimensions inputs (width x height)
  - Y-axis orientation toggle
setup:
  - action: open_preferences
  - action: navigate_to
    section: machine
  - action: open_basic_settings
  - action: capture
    region: dialog
filename: UI-MachineBasic-Settings.png
alt: "Machine basic settings dialog"
-->

<!-- ![Machine basic settings](../images/UI-Machine-Basic-Settings.png) -->

**Machine Name:**
- Descriptive name for this profile
- Examples: "Workshop K40", "Garage Diode Laser"

**Driver Selection:**
- Choose how Rayforge communicates with your machine
- See [Device Configuration](device-config.md) for driver details

**Dimensions (width  height):**
- Your machine's working area in millimeters
- Example: 400  300 for a 400mm wide, 300mm deep bed
- Measure actual cutting area, not machine exterior

**Y-Axis Orientation:**
- Unchecked: Y increases upward (most common)
- Checked: Y increases downward (some CNC-style machines)
- Test by moving Y+ and observing direction

### Step 3: Set Speed Limits

**Max Travel Speed:**
- Fastest non-cutting movement speed
- Used for rapid positioning moves
- Typical range: 2000-5000 mm/min
- Check your machine's specifications

**Max Cutting Speed:**
- Maximum speed while laser is firing
- Used to limit cutting operations
- Typical range: 500-2000 mm/min
- Lower than travel speed for accuracy

!!! tip "Finding Safe Speeds"
    Start conservative (travel: 3000, cut: 1000) and increase gradually. Observe your machine for:
    - Belt skipping or motor stalling
    - Excessive vibration
    - Loss of positioning accuracy

### Step 4: Configure Laser Head(s)

Each laser head in your system needs configuration:

<!-- SCREENSHOT
id: ui-laser-head-config
type: screenshot
size: dialog
description: |
  Laser head configuration dialog showing:
  - Head name field
  - Tool number selector
  - Max power setting (0-1000)
  - Frame power setting
  - Spot size inputs (X, Y in mm)
setup:
  - action: open_preferences
  - action: navigate_to
    section: machine
  - action: open_head_config
    head_index: 0
  - action: capture
    region: dialog
filename: UI-LaserHead-Config.png
alt: "Laser head configuration dialog"
-->

<!-- ![Laser head configuration](../images/UI-Laser-Head-Config.png) -->

**Head Name:**
- Descriptive label (e.g., "10W Diode", "CO Tube")
- Useful for multi-laser machines

**Tool Number:**
- For multi-head machines, the tool index (T0, T1, etc.)
- Single-head machines use 0
- Must match your controller's tool mapping

**Max Power (0-1000):**
- Maximum laser power value in your G-code
- GRBL typically uses 0-1000 range (S1000 = 100%)
- Some controllers use 0-255
- Check your controller documentation

**Frame Power:**
- Power level for framing (outlining without cutting)
- Set to 0 to disable framing
- Typical values: 5-20 (just visible, won't mark)
- Higher values for diode lasers (blue is dimmer)

**Spot Size (mm):**
- Physical size of focused laser beam
- X and Y dimensions (usually circular: 0.1  0.1)
- Affects fill pattern spacing and kerf calculations
- Measure: Test cut a circle, measure diameter difference

!!! info "Multiple Laser Heads"
    Some machines have multiple laser types (e.g., diode + CO). Configure each head separately and use tool selection commands to switch between them. See [Multi-Laser Setup](multi-laser.md).

### Step 5: Additional Settings

**G-code Dialect:**
- Command format your controller understands
- Options: GRBL (most common), Smoothie, Marlin
- Mismatched dialect causes command errors
- See [Device Configuration](device-config.md)

**G-code Precision:**
- Decimal places for coordinates
- Default: 3 (e.g., X12.345)
- Lower values (1-2) produce smaller files
- Higher values (4-5) increase precision
- 3 is optimal for most machines

**Home on Start:**
- Enable: Machine homes (finds origin) on connection
- Disable: No automatic homing
- Only enable if your machine has home switches
- Recommended if hardware supports it

**Clear Alarm on Connect:**
- Enable: Automatically clears GRBL alarm state on connection
- Useful if machine frequently enters alarm state
- May hide important safety alerts
- Generally leave disabled unless needed

## Saving and Managing Profiles

### Profile Storage

Rayforge automatically saves profile changes to:

```
~/.config/rayforge/machines/<machine-id>.yaml
```

Each machine instance is saved separately, allowing multiple profiles.

### Exporting Profiles

To share a profile with others:

1. Locate your machine YAML file in the config directory
2. Copy the file
3. Share via email, forum, or repository
4. Others can import by placing in their config directory

### Creating Multiple Machine Instances

For different machines or configurations:

1. **Menu:** File  New Machine
2. Configure settings for the new machine
3. Switch between machines using the machine selector

<!-- SCREENSHOT
id: ui-machine-selector-dropdown
type: screenshot
size: custom
region:
  x: 0
  y: 0
  width: 0
  height: 0
description: |
  Machine selector dropdown showing multiple configured machines
setup:
  - action: create_test_machines
    machines:
      - name: "K40 Laser"
      - name: "Diode Engraver"
      - name: "Workshop CO2"
  - action: open_machine_selector
  - action: capture
    region: dropdown
filename: UI-MachineSelector.png
alt: "Machine selector dropdown"
-->

<!-- ![Machine selector](../images/UI-Machine-Selector.png) -->

**Use cases for multiple profiles:**
- Different physical machines
- Same machine with different laser heads
- Test vs. production configurations
- Different material-specific speed limits

## Troubleshooting Profiles

### Machine won't connect

**Check:**
- Driver is correct for your connection type
- Driver arguments match your setup (port, baud rate, IP address)
- Physical connection is secure
- See [Connection Troubleshooting](../troubleshooting/connection.md)

### Wrong coordinate system

**Symptoms:**
- Jobs run mirrored or inverted
- Origin in wrong corner

**Fix:**
- Toggle "Y-Axis Down" setting
- Verify origin location matches your machine
- Test with a small square in corner

### Speeds too fast/slow

**Symptoms:**
- Motors skipping steps
- Excessive vibration
- Jobs take too long

**Fix:**
- Reduce max speeds in profile
- Test with [Material Test Grid](../features/operations/material-test-grid.md)
- Check belt tension and mechanical condition

### Power levels incorrect

**Symptoms:**
- Laser too weak at 100%
- Laser too strong at low percentages

**Check:**
- Max power setting matches controller range
- Some controllers use 0-255, others 0-1000
- Frame power should be barely visible, not cutting
- See [GRBL Settings](grbl-settings.md) for $30 and $31

### G-code commands rejected

**Symptoms:**
- "Unknown command" errors in log
- Commands not executing

**Fix:**
- Verify G-code dialect matches your controller
- Check GRBL vs. Smoothie vs. Marlin
- See [Device Configuration](device-config.md)

## Profile Best Practices

### Before Creating a Profile

 **Document your machine:**
- Take photos of controller and labeling
- Record working area dimensions
- Note any special features (auto-focus, air assist, rotary)
- Check manufacturer specifications

 **Test connection manually:**
- Verify you can connect with other software
- Confirm baud rate and port
- Test basic G-code commands

### Safe Testing Procedure

When testing a new profile:

1. **Start conservative:**
   - Lower speeds (50% of max)
   - Lower power (20-30%)
   - Small test jobs

2. **Test systematically:**
   - Verify connection first
   - Test framing (no cutting)
   - Run a small engraving (low power)
   - Progress to cutting slowly

3. **Use simulation:**
   - Preview in [Simulation Mode](../features/simulation-mode.md) first
   - Verify speeds look reasonable
   - Check path is correct

### Documentation

For complex or unusual machines:

- Keep notes on settings that work
- Document any quirks or special requirements
- Record material test results
- Share profiles with community (forums, wiki)

## Common Profile Examples

### Typical K40 Laser

```yaml
name: "K40 CO Laser"
driver: GrblSerialDriver
dimensions: [300, 200]
y_axis_down: false
max_travel_speed: 3000
max_cut_speed: 1000
dialect: GRBL
heads:
  - frame_power: 10
    max_power: 1000
    spot_size_mm: [0.1, 0.1]
```

### Diode Engraver

```yaml
name: "20W Diode Engraver"
driver: GrblSerialDriver
dimensions: [400, 400]
y_axis_down: false
max_travel_speed: 5000
max_cut_speed: 2000
dialect: GRBL
heads:
  - frame_power: 20
    max_power: 1000
    spot_size_mm: [0.15, 0.15]
```

### Large Format CO

```yaml
name: "1000600 CO Laser"
driver: GrblNetworkDriver
dimensions: [1000, 600]
y_axis_down: false
max_travel_speed: 4000
max_cut_speed: 1500
dialect: GRBL
heads:
  - frame_power: 8
    max_power: 1000
    spot_size_mm: [0.08, 0.08]
```

## Related Topics

- **[Device Configuration](device-config.md)** - Driver setup and connection details
- **[GRBL Settings](grbl-settings.md)** - Controller-level parameters
- **[Multi-Laser Setup](multi-laser.md)** - Configuring machines with multiple laser heads
- **[Connection Troubleshooting](../troubleshooting/connection.md)** - Fixing connection issues
