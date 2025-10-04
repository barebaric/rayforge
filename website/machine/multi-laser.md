# Multi-Laser Setup

Some laser machines have multiple laser modules (e.g., a diode laser and a CO tube, or multiple diodes at different wavelengths). RayForge supports multi-laser setups through tool selection and head configuration.

## Overview

Multi-laser systems allow:

- **Multiple laser types**: Combine different laser technologies on one machine
- **Different wavelengths**: Blue diode for marking, IR for cutting
- **Varied power levels**: Low-power engraving head + high-power cutting head
- **Specialized applications**: UV for special materials, fiber for metals

RayForge handles multi-laser machines through:

- **Laser head configuration**: Define each laser's properties
- **Tool selection**: Switch between lasers via G-code commands
- **Per-operation assignment**: Choose which laser executes each operation

<!-- SCREENSHOT
id: ui-multi-laser-config
type: screenshot
size: dialog
description: |
  Multi-laser configuration dialog showing:
  - List of configured laser heads
  - Add/Remove head buttons
  - Head properties for each laser
  - Tool number assignments
setup:
  - action: open_preferences
  - action: navigate_to
    section: machine
  - action: open_heads_config
  - action: add_second_head
  - action: capture
    region: dialog
filename: UI-MultiLaser-Config.png
alt: "Multi-laser configuration dialog"
-->

<!-- ![Multi-laser configuration](../images/UI-Multi-Laser-Config.png) -->

## Common Multi-Laser Configurations

### Diode + CO

**Use case:** Versatility - marking with diode, cutting with CO

**Example setup:**
- **Head 1 (T0):** 10W blue diode laser
  - Engraving on metals, anodized aluminum
  - Low-power marking
  - Visible beam for alignment
- **Head 2 (T1):** 40W CO laser tube
  - Cutting wood, acrylic, leather
  - Deep engraving
  - High power for thick materials

**Advantages:**
- Best of both worlds
- One machine for multiple materials
- No tool changes required

**Challenges:**
- Different focal lengths (may need Z adjustment)
- Different power characteristics
- More complex calibration

### Dual Diodes

**Use case:** Different wavelengths or power levels

**Example setup:**
- **Head 1 (T0):** 20W 450nm blue diode
  - General cutting and engraving
  - Wood, leather, cardboard
- **Head 2 (T1):** 5W 1064nm infrared diode
  - Metal marking
  - Anodized aluminum engraving
  - Specialized materials

**Advantages:**
- Similar focal characteristics
- Easier mechanical alignment
- Compatible power supplies

### High Power + Low Power

**Use case:** Precision engraving and heavy cutting on one machine

**Example setup:**
- **Head 1 (T0):** 5W diode for fine detail
  - Intricate engraving
  - Small text
  - Delicate work
- **Head 2 (T1):** 80W CO for cutting
  - Thick material cutting
  - Fast production work
  - Heavy-duty operations

## Configuring Laser Heads

Each laser head needs individual configuration in your machine profile.

### Adding a Laser Head

1. **Open Preferences**  Machine  Laser Heads
2. Click "Add Head"
3. Configure head properties (see below)
4. Assign tool number
5. Save profile

### Laser Head Properties

<!-- SCREENSHOT
id: ui-single-head-config
type: screenshot
size: dialog
description: |
  Single laser head configuration showing:
  - Head name field
  - Tool number selector (T0, T1, T2, etc.)
  - Max power setting
  - Frame power setting
  - Spot size X/Y inputs
setup:
  - action: open_preferences
  - action: navigate_to
    section: machine
  - action: open_heads_config
  - action: select_head
    index: 1
  - action: capture
    region: dialog
filename: UI-SingleHead-Config.png
alt: "Single laser head configuration dialog"
-->

<!-- ![Single head configuration](../images/UI-Single-Head-Config.png) -->

**Head Name:**
- Descriptive label for this laser
- Examples: "Blue Diode 20W", "CO Tube", "IR Fiber Laser"
- Helps identify which laser to use for each operation

**Tool Number:**
- G-code tool identifier (T0, T1, T2, etc.)
- T0 typically the default/primary laser
- Must match your controller's tool configuration
- Sequential numbering (0, 1, 2...)

**Max Power (0-1000):**
- Maximum S value for this laser
- Defines 100% power for this head
- Typically 1000 for GRBL (S1000 = 100%)
- See [GRBL Settings](grbl-settings.md) $30/$31

**Frame Power:**
- Low-power setting for framing (outline preview)
- Should be barely visible, not cutting
- Typical: 5-20 for diodes, 5-10 for CO
- Set to 0 to disable framing for this head

**Spot Size (mm):**
- Physical diameter of focused laser beam
- X and Y dimensions (usually circular)
- Affects fill patterns and kerf calculations
- Measure by test cutting and comparing to design

### Example Configuration

**Diode + CO Setup:**

**Head 1 - Blue Diode:**
```yaml
name: "20W Blue Diode"
tool_number: 0
max_power: 1000
frame_power: 20
spot_size_mm: [0.15, 0.15]
```

**Head 2 - CO Tube:**
```yaml
name: "50W CO Tube"
tool_number: 1
max_power: 1000
frame_power: 8
spot_size_mm: [0.08, 0.08]
```

## Tool Selection in G-code

RayForge automatically generates tool change commands when switching between laser heads.

### Tool Change Command

**G-code:**
```gcode
T<number>      ; Select tool
```

**Examples:**
```gcode
T0             ; Select first laser (tool 0)
T1             ; Select second laser (tool 1)
```

### Automatic Tool Selection

When you assign operations to different laser heads, RayForge:

1. Groups operations by tool number
2. Inserts tool change commands between groups
3. Optimizes order to minimize tool changes

**Example workflow:**
```gcode
; Job with diode engraving and CO cutting

T0                    ; Select diode laser
M3 S300               ; Diode on at 30%
G1 X10 Y10 F3000      ; Engrave logo
M5                    ; Diode off

T1                    ; Select CO laser
M3 S800               ; CO on at 80%
G1 X0 Y0 F500         ; Cut outline
M5                    ; CO off
```

## Assigning Operations to Laser Heads

When creating operations, choose which laser should execute it.

### In Operation Settings

Each operation has a "Laser Head" selector:

<!-- SCREENSHOT
id: ui-operation-head-selector
type: screenshot
size: custom
region:
  x: 0
  y: 0
  width: 0
  height: 0
description: |
  Operation settings showing laser head selector with:
  - Dropdown showing available heads
  - Selected head highlighted
  - Icon indicating which tool
setup:
  - action: create_shape
    type: rectangle
  - action: add_operation
    type: contour
  - action: open_operation_settings
  - action: show_head_selector
  - action: capture
    region: settings_panel
filename: UI-OperationHead-Selector.png
alt: "Laser head selector in operation settings"
-->

<!-- ![Operation head selector](../images/UI-Operation-Head-Selector.png) -->

**Choosing the right head:**

**For engraving text/logos (low power):**
- Use diode laser if available
- Visible beam helps with alignment
- Lower power for surface marking

**For cutting thick materials:**
- Use CO if available
- Higher power for deep cutting
- Faster cutting speeds

**For metal marking:**
- Use IR/fiber laser if available
- Or diode for anodized aluminum
- Specific wavelengths for different metals

### Default Head Assignment

When adding operations:
- RayForge uses primary head (T0) by default
- Change in operation settings if needed
- Settings persist for similar operations

## Controller Requirements

Not all controllers support multiple laser heads. Check your controller's capabilities.

### GRBL Support

**Standard GRBL (v1.1):**
- Limited multi-tool support
- T0/T1 commands accepted but may not switch hardware
- Requires custom firmware or external relay

**Grbl_ESP32:**
- Better multi-tool support
- Can control multiple PWM outputs
- Configure via `config.h`

**Custom GRBL builds:**
- May support tool changing via custom macros
- Check your firmware documentation

### Hardware Switching

**Methods for physical tool selection:**

**1. PWM Output Switching:**
- Controller routes S command to different PWM pins
- T0  PWM1, T1  PWM2
- Requires firmware support

**2. External Relay:**
- G-code triggers relay (M3/M5 or custom M-code)
- Relay switches power to different lasers
- Works with standard GRBL

**3. Manual Switching:**
- Software tracks active tool
- User manually switches laser modules
- Useful for testing or simple setups

### Configuration Example (Grbl_ESP32)

In `config.h`:

```cpp
#define SPINDLE_TYPE SpindleType::PWM
#define SPINDLE_OUTPUT_PIN GPIO_NUM_25   // Tool 0
#define SPINDLE2_OUTPUT_PIN GPIO_NUM_26  // Tool 1
```

Consult your controller documentation for specific configuration.

## Calibration for Multi-Laser

### Alignment

Different lasers may not be perfectly co-located:

**Physical offset:**
- Measure distance between laser focal points
- Account for X/Y offset in designs
- Or use RayForge's future "head offset" feature

**Z-axis (focus) offset:**
- Different lasers may have different focal lengths
- Adjust Z height when changing tools
- Consider motorized Z-axis for automatic adjustment

### Testing Alignment

1. Create test pattern (crosshairs + small squares)
2. Engrave with Tool 0 (first laser)
3. Engrave same pattern with Tool 1 (second laser)
4. Measure offset between patterns
5. Adjust design placement or configure head offset

<!-- SCREENSHOT
id: ref-multi-laser-alignment-test
type: screenshot
size: full-window
description: |
  Test pattern showing alignment check for multi-laser setup:
  - Crosshairs engraved by Tool 0 (blue)
  - Crosshairs engraved by Tool 1 (red overlay)
  - Measurement annotations showing X/Y offset
filename: Ref-MultiLaser-Alignment.png
alt: "Multi-laser alignment test pattern"
-->

<!-- ![Alignment test pattern](../images/Ref-Multi-Laser-Alignment.png) -->

### Power Calibration

Each laser needs individual power/speed testing:

1. Run [Material Test Grid](../features/operations/material-test-grid.md) for Tool 0
2. Record optimal settings
3. Run separate test grid for Tool 1
4. Record settings
5. Use appropriate settings for each laser in operations

Power levels are **not** interchangeable between lasers!

**Example:**
- Diode at S300 (30%) for engraving
- CO at S800 (80%) for same visual result
- Different materials respond differently to each wavelength

## Workflow Examples

### Example 1: Logo Engraving + Part Cutting

**Scenario:** Engrave logo with diode, cut outline with CO

**Steps:**
1. Import design with logo and outline
2. Create raster operation for logo  Assign to Tool 0 (diode)
3. Create contour operation for outline  Assign to Tool 1 (CO)
4. RayForge generates:
   ```gcode
   T0                ; Select diode
   ; ... raster engraving commands
   T1                ; Select CO
   ; ... contour cutting commands
   ```

**Order:**
- Engrave first (while part secured to bed)
- Cut second (part comes free)

### Example 2: Multi-Material Job

**Scenario:** Mark anodized aluminum and cut acrylic in same job

**Steps:**
1. Design includes aluminum tags and acrylic frame
2. Engraving on aluminum  Tool 0 (diode for anodized)
3. Cutting on acrylic  Tool 1 (CO for cutting)
4. Position materials correctly on bed
5. Run job with automatic tool switching

### Example 3: Detail + Production

**Scenario:** Fine engraving with low-power laser, heavy cutting with high-power laser

**Steps:**
1. Intricate design elements  Tool 0 (5W for detail)
2. Structural cuts  Tool 1 (80W for speed)
3. Same material, different laser strengths
4. Automatic switching based on operation requirements

## Troubleshooting Multi-Laser Setups

### Wrong laser activates

**Symptoms:**
- T0 command triggers T1 laser
- Both lasers fire simultaneously

**Fix:**
- Check controller firmware tool support
- Verify PWM output pin configuration
- Test with manual T0/T1 commands
- Check relay wiring if using external switching

### Tool changes don't work

**Symptoms:**
- T1 command ignored
- No physical switching occurs

**Fix:**
- Verify controller supports multi-tool (check firmware)
- Check tool number assignments in head config
- Test tool commands manually in console
- May need firmware modification or external relay

### Power levels incorrect after tool change

**Symptoms:**
- S500 on Tool 1 gives different power than Tool 0

**Fix:**
- Verify each laser's max power setting ($30)
- Calibrate each laser separately
- Ensure power values in operations match laser characteristics
- Check $30/$31 haven't been changed

### Alignment offset between lasers

**Symptoms:**
- Tool 0 and Tool 1 engrave in different positions
- Designs don't line up

**Fix:**
- Run alignment test pattern
- Measure X/Y offset
- Account for offset in design placement
- Consider adding head offset configuration (future feature)

### Focus changes between tools

**Symptoms:**
- Tool 1 is out of focus when Tool 0 was focused

**Fix:**
- Different lasers may have different focal lengths
- Manually adjust Z-axis when changing tools
- Or install motorized Z-axis
- Or add Z-offset in tool change macro

## Advanced: Tool Change Macros

For advanced users with controller support, tool change macros can automate switching.

### Example Macro (Grbl_ESP32)

**Tool 0  Tool 1:**
```gcode
; Macro for T1
G0 Z5               ; Raise Z-axis (clearance)
M5                  ; Ensure laser off
; Custom M-code to switch relay
M62 P1              ; Example: Enable relay for Tool 1
G0 Z0               ; Return to work height (Tool 1 focus)
```

**Tool 1  Tool 0:**
```gcode
; Macro for T0
G0 Z5
M5
M63 P1              ; Disable relay (back to Tool 0)
G0 Z2.5             ; Return to work height (Tool 0 focus, different focal length)
```

Check your controller documentation for macro support and syntax.

## Future Enhancements

RayForge may add in future versions:

- **Head offset configuration**: Define X/Y offset between laser focal points
- **Auto Z-adjustment**: Automatic Z movement for different focal lengths
- **Tool-specific speed limits**: Different max speeds per laser
- **Visual tool indicators**: Show which laser is active in simulation
- **Per-head kerf settings**: Different kerf compensation for each laser

## Related Topics

- **[Machine Profiles](profiles.md)** - Configuring your machine
- **[GRBL Settings](grbl-settings.md)** - Controller-level parameters
- **[Material Test Grid](../features/operations/material-test-grid.md)** - Calibrating power/speed for each laser
- **[Simulation Mode](../features/simulation-mode.md)** - Previewing multi-tool jobs
