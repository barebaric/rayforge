# GRBL Settings

GRBL settings (also called "dollar settings" or `$$` settings) are parameters stored in your controller's EEPROM that control how your laser behaves. Understanding these settings is essential for proper machine configuration and troubleshooting.

!!! warning "Caution When Changing Settings"
    Incorrect GRBL settings can cause your machine to behave unpredictably, lose position, or even damage hardware. Always record original values before making changes, and modify one setting at a time.

## Overview

GRBL settings control:

- **Motion parameters**: Speed limits, acceleration, calibration
- **Limit switches**: Homing behavior, soft/hard limits
- **Laser control**: Power range, laser mode enable
- **Electrical configuration**: Pin inversions, pullups
- **Reporting**: Status message format and frequency

These settings are stored on your controller (not in Rayforge) and persist across power cycles.

<!-- SCREENSHOT
id: ui-grbl-settings-panel
type: screenshot
size: custom
region:
  x: 0
  y: 0
  width: 0
  height: 0
description: |
  GRBL settings panel showing:
  - Category tabs (Stepper, Limits & Homing, Laser, Calibration, etc.)
  - Settings list with current values
  - Edit/Reset buttons
  - Apply changes button
  - Refresh from device button
setup:
  - action: connect_to_device
  - action: open_preferences
  - action: navigate_to
    section: machine
  - action: open_grbl_settings
  - action: capture
    region: settings_panel
filename: UI-GRBLSettings.png
alt: "GRBL settings configuration panel"
-->

<!-- ![GRBL settings panel](../images/UI-GRBL-Settings.png) -->

## Accessing GRBL Settings

### From Rayforge

1. **Menu:** Preferences  Machine  GRBL Settings
2. Rayforge reads settings from device automatically
3. Edit values in the interface
4. Click "Apply" to write changes to controller

### From Console

You can also view/modify settings via G-code console:

**View all settings:**
```
$$
```

**View single setting:**
```
$100
```

**Modify setting:**
```
$100=80.0
```

**Restore defaults:**
```
$RST=$
```

!!! danger "Restore Defaults Erases All Settings"
    The `$RST=$` command resets all GRBL settings to factory defaults. You'll lose any calibration and tuning. Back up your settings first!

## Critical Settings for Lasers

These settings are most important for laser operation:

### $32 - Laser Mode

**Value:** 0 = Disabled, 1 = Enabled

**Purpose:** Enables laser-specific features in GRBL

**When enabled (1):**
- Laser automatically turns off during G0 (rapid) moves
- Power dynamically adjusts during acceleration/deceleration
- Prevents accidental burns during positioning

**When disabled (0):**
- Laser behaves like a spindle (CNC mode)
- Doesn't turn off during rapids
- **Dangerous for laser use!**

!!! warning "Always Enable Laser Mode"
    $32 should **always** be set to 1 for laser cutters. Disabled laser mode can cause unintended burns and fire hazards.

**Setting laser mode:**
```gcode
$32=1
```

### $30 & $31 - Laser Power Range

**$30 - Maximum Laser Power (RPM)**
**$31 - Minimum Laser Power (RPM)**

**Purpose:** Defines the power range for S commands

**Typical values:**
- $30=1000, $31=0 (S0-S1000 range, most common)
- $30=255, $31=0 (S0-S255 range, some controllers)
- $30=1000, $31=10 (prevents laser from dropping below 1%)

**How it works:**
- `S1000`  $30 value  100% laser power
- `S500`  50% of $30  50% laser power
- `S0`  $31 value  Minimum power (typically 0%)

**Example:**
```gcode
$30=1000.0    ; Maximum S value = 100% power
$31=0.0       ; Minimum S value = 0% power
```

!!! tip "Matching Rayforge Configuration"
    The "Max Power" setting in your [Machine Profile](profiles.md) should match your $30 value. If $30=1000, set max power to 1000 in Rayforge.

### $130 & $131 - Maximum Travel

**$130 - X Maximum Travel (mm)**
**$131 - Y Maximum Travel (mm)**

**Purpose:** Defines your machine's working area

**Typical values:**
- $130=300, $131=200 (300mm  200mm bed)
- $130=400, $131=400 (400mm  400mm bed)
- $130=1000, $131=600 (large format laser)

**Why it matters:**
- Soft limits ($20) use these values to prevent crashes
- Defines the coordinate system bounds
- Must match your physical machine size

**Measuring your bed:**
1. Manually jog to maximum X position
2. Note X position from status (`?` command)
3. Set $130 to that value
4. Repeat for Y axis ($131)

**Example:**
```gcode
$130=400.0    ; X-axis can travel 400mm
$131=300.0    ; Y-axis can travel 300mm
```

## Settings by Category

### Stepper Configuration ($0-$6)

Controls stepper motor electrical signals and timing.

| Setting | Description | Typical Value | Notes |
|---------|-------------|---------------|-------|
| $0 | Step pulse time ( s) | 10 | How long step signal lasts |
| $1 | Step idle delay (ms) | 25 | Delay before disabling steppers |
| $2 | Step pulse invert (mask) | 0 | Invert step signal (rare) |
| $3 | Step direction invert (mask) | 0 | Reverse motor direction |
| $4 | Invert step enable pin | 0 | Invert stepper enable signal |
| $5 | Invert limit pins | 0 | Invert limit switch logic |
| $6 | Invert probe pin | 0 | Invert probe signal (not used for lasers) |

**When to change:**
- Motors run backwards: Adjust $3
- Stepper driver compatibility: Adjust $0 or $4
- Limit switches trigger incorrectly: Adjust $5

**Example - Inverting Y direction:**
```gcode
$3=2          ; Bit 1 set (binary 10 = Y-axis inverted)
```

### Control & Reporting ($10-$13)

Controls motion planning and status report format.

| Setting | Description | Typical Value | Notes |
|---------|-------------|---------------|-------|
| $10 | Status report options (mask) | 1 | WPos=1, MPos=0 |
| $11 | Junction deviation (mm) | 0.010 | Cornering speed (lower = sharper) |
| $12 | Arc tolerance (mm) | 0.002 | Accuracy of curved paths |
| $13 | Report in inches | 0 | 0=mm, 1=inches |

**$11 - Junction Deviation:**
- **Lower (0.005-0.010)**: Sharp corners, slower on curves, better accuracy
- **Higher (0.020-0.050)**: Smoother motion, faster on curves, slightly rounded corners
- **Laser typical**: 0.010mm works well for most jobs

**$12 - Arc Tolerance:**
- Defines how finely arcs are approximated
- Lower = smoother circles, more G-code commands
- 0.002mm is good for most work

**Example:**
```gcode
$10=1         ; Report work position (WPos)
$11=0.010     ; 0.010mm junction deviation
$12=0.002     ; 0.002mm arc tolerance
$13=0         ; Report in millimeters
```

### Limits & Homing ($20-$27)

Controls limit switches and homing behavior.

| Setting | Description | Typical Value | Notes |
|---------|-------------|---------------|-------|
| $20 | Soft limits enable | 0 or 1 | Prevents moves beyond $130/$131 |
| $21 | Hard limits enable | 0 | Physical limit switch protection |
| $22 | Homing cycle enable | 0 or 1 | Enable $H homing command |
| $23 | Homing direction invert (mask) | 0 | Which direction to home |
| $24 | Homing locate feed rate (mm/min) | 25 | Slow approach speed |
| $25 | Homing search seek rate (mm/min) | 500 | Fast search speed |
| $26 | Homing debounce delay (ms) | 250 | Switch debounce time |
| $27 | Homing pull-off distance (mm) | 1.0 | Distance to back off switch |

**$20 - Soft Limits:**
- **Enabled (1)**: Controller prevents moves beyond $130/$131
- **Disabled (0)**: No software travel limits
- Requires homing to work (must know position)
- Recommended if you have home switches

**$21 - Hard Limits:**
- **Enabled (1)**: Triggers alarm if limit switch activated
- **Disabled (0)**: Limit switches ignored (except during homing)
- Can cause nuisance alarms if switches are sensitive
- Often left disabled on lasers

**$22 - Homing Enable:**
- **Enabled (1)**: `$H` command performs homing cycle
- **Disabled (0)**: Homing not available
- Only enable if you have home switches installed

**Homing cycle process:**
1. Rapid search ($25 speed) until switch triggers
2. Back off pull-off distance ($27)
3. Slow approach ($24 speed) for precise location
4. Back off to pull-off distance
5. Set position to (0,0)

**Example - Machine with home switches:**
```gcode
$20=1         ; Enable soft limits
$21=0         ; Disable hard limits (optional)
$22=1         ; Enable homing
$23=0         ; Home to min (default)
$24=50.0      ; Slow approach at 50mm/min
$25=1000.0    ; Fast search at 1000mm/min
$26=250       ; 250ms debounce
$27=2.0       ; Pull off 2mm from switch
```

### Spindle & Laser ($30-$32)

**See "Critical Settings" section above** for detailed explanation of $30, $31, and $32.

| Setting | Description | Laser Value | Notes |
|---------|-------------|-------------|-------|
| $30 | Maximum spindle speed | 1000.0 | Max S value (100% power) |
| $31 | Minimum spindle speed | 0.0 | Min S value (0% power) |
| $32 | Laser mode enable | 1 | **MUST be 1 for lasers** |

**Example:**
```gcode
$30=1000.0
$31=0.0
$32=1         ; Laser mode ON
```

### Axis Calibration ($100-$102)

Defines how many stepper motor steps equal one millimeter of movement.

| Setting | Description | Notes |
|---------|-------------|-------|
| $100 | X steps/mm | Depends on pulley/belt/gear ratio |
| $101 | Y steps/mm | Usually same as X for typical lasers |
| $102 | Z steps/mm | Not used on most lasers |

**Calculating steps/mm:**

```
steps/mm = (motor_steps_per_rev  microstepping) / (pulley_teeth  belt_pitch)
```

**Example calculation:**
- Motor: 200 steps/rev (1.8  stepper)
- Microstepping: 16 (typical)
- Pulley: 20 teeth
- Belt: GT2 (2mm pitch)

```
steps/mm = (200  16) / (20  2) = 3200 / 40 = 80
```

**Calibration procedure:**
1. Mark starting position on bed
2. Command 100mm move: `G0 X100`
3. Measure actual distance traveled
4. Calculate: `new_value = current_value  (100 / actual_distance)`
5. Set new value: `$100=new_value`

**Example:**
```gcode
$100=80.0     ; X-axis: 80 steps/mm
$101=80.0     ; Y-axis: 80 steps/mm
$102=80.0     ; Z-axis: 80 steps/mm (unused)
```

!!! tip "Calibration Test"
    Cut a 100mm  100mm square and measure the result. If it's 102mm  98mm, adjust $100 and $101 accordingly using the formula above.

### Axis Kinematics ($110-$122)

Controls maximum speed and acceleration for each axis.

| Setting | Description | Typical Value | Notes |
|---------|-------------|---------------|-------|
| $110 | X max rate (mm/min) | 5000.0 | Max X speed |
| $111 | Y max rate (mm/min) | 5000.0 | Max Y speed |
| $112 | Z max rate (mm/min) | 500.0 | Max Z speed (unused) |
| $120 | X acceleration (mm/sec ) | 500.0 | How fast X accelerates |
| $121 | Y acceleration (mm/sec ) | 500.0 | How fast Y accelerates |
| $122 | Z acceleration (mm/sec ) | 100.0 | How fast Z accelerates |

**Max rate ($110-$112):**
- Absolute speed limit for each axis
- Controller won't exceed these values
- Should match or exceed your machine profile's max speeds
- Typical: 3000-6000 mm/min for lasers

**Acceleration ($120-$122):**
- How quickly machine reaches target speed
- **Lower (100-300)**: Smooth, slow acceleration, gentler on mechanics
- **Higher (500-1000)**: Snappy, fast start/stop, may cause vibration
- Typical: 300-500 mm/sec  for lasers

**Finding safe values:**
1. Start conservative (max rate: 3000, accel: 300)
2. Gradually increase while testing
3. Listen for motor skipping, belt slipping
4. Watch for excessive vibration or ringing

**Example:**
```gcode
$110=4000.0   ; X max 4000 mm/min
$111=4000.0   ; Y max 4000 mm/min
$112=500.0    ; Z max 500 mm/min
$120=400.0    ; X accel 400 mm/sec 
$121=400.0    ; Y accel 400 mm/sec 
$122=100.0    ; Z accel 100 mm/sec 
```

### Axis Travel ($130-$132)

**See "Critical Settings" section** for detailed explanation of $130 and $131.

| Setting | Description | Notes |
|---------|-------------|-------|
| $130 | X max travel (mm) | Working area width |
| $131 | Y max travel (mm) | Working area depth |
| $132 | Z max travel (mm) | Z travel (if applicable) |

**Example:**
```gcode
$130=400.0    ; 400mm X travel
$131=400.0    ; 400mm Y travel
$132=50.0     ; 50mm Z travel (if you have Z-axis)
```

## Common Configuration Examples

### Typical Small Diode Laser (300 400mm)

```gcode
$0=10          ; Step pulse 10 s
$1=255         ; Step idle delay 255ms
$2=0           ; No step invert
$3=0           ; No direction invert
$4=0           ; No enable invert
$5=0           ; No limit invert
$10=1          ; Report WPos
$11=0.010      ; Junction deviation 0.01mm
$12=0.002      ; Arc tolerance 0.002mm
$13=0          ; Report mm
$20=1          ; Soft limits enabled
$21=0          ; Hard limits disabled
$22=1          ; Homing enabled
$23=0          ; Home to min
$24=50.0       ; Homing feed 50mm/min
$25=1000.0     ; Homing seek 1000mm/min
$26=250        ; Homing debounce 250ms
$27=2.0        ; Homing pull-off 2mm
$30=1000.0     ; Max power S1000
$31=0.0        ; Min power S0
$32=1          ; Laser mode ON
$100=80.0      ; X steps/mm
$101=80.0      ; Y steps/mm
$102=80.0      ; Z steps/mm
$110=5000.0    ; X max rate
$111=5000.0    ; Y max rate
$112=500.0     ; Z max rate
$120=500.0     ; X accel
$121=500.0     ; Y accel
$122=100.0     ; Z accel
$130=400.0     ; X max travel
$131=300.0     ; Y max travel
$132=0.0       ; Z max travel
```

### CO  Laser Without Home Switches

```gcode
$0=10
$1=25
$2=0
$3=0
$4=0
$5=0
$10=1
$11=0.010
$12=0.002
$13=0
$20=0          ; Soft limits OFF (no homing)
$21=0          ; Hard limits OFF
$22=0          ; Homing disabled
$30=1000.0
$31=0.0
$32=1          ; Laser mode ON
$100=157.48    ; Example calibration value
$101=157.48
$102=157.48
$110=3000.0
$111=3000.0
$112=500.0
$120=300.0
$121=300.0
$122=100.0
$130=600.0     ; 600 400mm bed
$131=400.0
$132=0.0
```

## Backing Up and Restoring Settings

### Backup Procedure

1. **Via Rayforge:**
   - Open GRBL Settings panel
   - Click "Export Settings"
   - Save file as `grbl-backup-YYYY-MM-DD.txt`

2. **Via console:**
   - Send `$$` command
   - Copy all output to text file
   - Save with date

### Restore Procedure

1. **From backup file:**
   - Open file
   - Send each line (`$100=80.0`, etc.) via console
   - Verify with `$$` command

2. **Via Rayforge:**
   - Open GRBL Settings panel
   - Click "Import Settings"
   - Select backup file
   - Review changes and apply

!!! tip "Regular Backups"
    Back up your GRBL settings after any calibration or tuning. Store backups in a safe location (cloud storage, USB drive). This makes recovery quick if settings are lost or corrupted.

## Troubleshooting GRBL Settings

### Can't write settings

**Error:** `error:7` (Read-only setting)

**Cause:** Setting is locked or doesn't exist

**Fix:**
- Check setting number is valid for your GRBL version
- Some settings require specific GRBL compile options

---

**Error:** `error:3` (Invalid statement)

**Cause:** Incorrect syntax

**Fix:**
- Use format: `$100=80.0` (no spaces around =)
- Ensure value is appropriate type (number vs. boolean)

### Settings won't apply

**Symptom:** Changes don't take effect

**Fix:**
- Send soft reset after changes: `Ctrl+X` or `$X`
- Power cycle controller
- Verify setting was written: Send `$$` and check value

### Machine behavior incorrect

**Symptom:** Motors run backwards

**Fix:**
- Adjust $3 (direction invert) for affected axis
- X-axis: $3=1, Y-axis: $3=2, Both: $3=3

---

**Symptom:** Jobs go out of bounds

**Fix:**
- Check $130/$131 match your machine size
- Enable soft limits ($20=1) with homing ($22=1)
- Verify origin position before starting job

---

**Symptom:** Laser doesn't turn off during rapids

**Fix:**
- **CRITICAL**: Set $32=1 (laser mode)
- Verify with `$$` that $32 shows value 1
- Power cycle controller if necessary

### Lost all settings

**Cause:** EEPROM cleared, firmware update, or `$RST=$` command

**Fix:**
1. Restore from backup file (see "Backing Up" above)
2. Or recalibrate from scratch:
   - Start with example config for your machine type
   - Adjust $100/$101 calibration
   - Set $130/$131 to bed size
   - Enable laser mode ($32=1)
   - Set power range ($30/$31)
   - Tune speeds and acceleration

## Advanced Topics

### GRBL Versions

Different GRBL versions support different settings:

- **GRBL 0.9:** Older, fewer settings
- **GRBL 1.1:** Standard, most common
- **Grbl_ESP32:** Extended settings for ESP32 features
- **GRBL-Mega:** More axes (A, B, C)

Check your version:
```gcode
$I
```

Output example:
```
[VER:1.1h.20190825:]
[OPT:V,15,128]
```

### Bitmask Settings

Some settings use bitmasks (multiple flags in one number):

**$3 - Direction Invert Mask:**
- Bit 0 (value 1) = X-axis
- Bit 1 (value 2) = Y-axis
- Bit 2 (value 4) = Z-axis

**Examples:**
- $3=0: No inversion
- $3=1: X inverted
- $3=2: Y inverted
- $3=3: X and Y inverted (1+2=3)
- $3=7: All axes inverted (1+2+4=7)

**$10 - Status Report Mask:**
- Bit 0 (value 1) = Report work position (WPos)
- Bit 1 (value 2) = Report buffer data

**Examples:**
- $10=0: MPos only
- $10=1: WPos only (recommended)
- $10=3: WPos + buffer (1+2=3)

## Related Topics

- **[Machine Profiles](profiles.md)** - Higher-level machine configuration
- **[Device Configuration](device-config.md)** - Driver and connection setup
- **[Connection Troubleshooting](../troubleshooting/connection.md)** - Fixing connection issues
- **[Common Problems](../troubleshooting/common.md)** - General troubleshooting

## External Resources

- [GRBL v1.1 Configuration](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Configuration)
- [GRBL v1.1 Commands](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Commands)
- [Grbl_ESP32 Documentation](https://github.com/bdring/Grbl_Esp32/wiki)
