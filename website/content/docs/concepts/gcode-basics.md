# G-code Basics

Understanding G-code helps you troubleshoot issues, optimize performance, and customize Rayforge output for your specific machine.

## What is G-code?

**G-code** is a standardized programming language for CNC machines (including laser cutters). It tells the machine exactly where to move, how fast to move, and what to do (laser on/off, power level, etc.).

**Key points:**

- **Text-based** - Human-readable commands
- **Line-by-line** - Machine executes one command at a time
- **Standardized** - Based on ISO 6983 and RS-274 standards
- **Dialect variations** - Different firmware (GRBL, Marlin, etc.) have slight differences

**Example G-code:**
```gcode
G21              ; Set units to millimeters
G90              ; Use absolute positioning
G0 X10 Y10       ; Rapid move to position (10, 10)
M4 S500          ; Turn laser on at 50% power
G1 X50 Y10 F3000 ; Linear move to (50, 10) at 3000 mm/min
M5               ; Turn laser off
```

---

## How Rayforge Generates G-code

Rayforge converts your designs into G-code through a multi-step process:

**Generation pipeline:**

```
Design (Workpieces + Layers)
         |
         v
Operations (Contour, Raster, etc.)
         |
         v
Toolpaths (Lines, arcs, points)
         |
         v
G-code Commands (GRBL dialect)
         |
         v
Output File (.gcode)
```

**What Rayforge does:**

1. **Analyzes your design** - Extracts geometry from workpieces
2. **Applies operations** - Determines cut/engrave paths
3. **Optimizes toolpaths** - Reorders paths, minimizes travel
4. **Generates commands** - Converts paths to G-code
5. **Injects hooks** - Adds user-defined macros at specified points
6. **Writes file** - Outputs complete G-code ready for machine

---

## G-code Command Structure

### Command Format

**Basic syntax:**
```
<Letter><Number> [<Parameter><Value> ...]
```

**Examples:**
- `G0` - G command, number 0 (rapid positioning)
- `G1 X50 Y20 F3000` - G command with parameters X, Y, F
- `M4 S500` - M command with parameter S

**Line structure:**
```gcode
G1 X50.0 Y20.5 F3000  ; Move to (50, 20.5) at 3000 mm/min
^  ^    ^     ^       ^
|  |    |     |       |
|  |    |     |       +-- Comment (optional)
|  |    |     +---------- F parameter (feed rate)
|  |    +---------------- Y parameter (Y coordinate)
|  +--------------------- X parameter (X coordinate)
+------------------------ G command (linear move)
```

### Command Types

| Letter | Type | Purpose | Examples |
|--------|------|---------|----------|
| **G** | Motion/Setup | Movement, positioning mode | G0, G1, G2, G3, G21, G90 |
| **M** | Machine | Laser on/off, air assist, coolant | M3, M4, M5, M8, M9 |
| **S** | Parameter | Spindle/laser power | S0-S1000 |
| **F** | Parameter | Feed rate (speed) | F3000 |
| **X/Y/Z** | Parameter | Coordinates | X50 Y20 |
| **I/J/K** | Parameter | Arc center offsets | I10 J5 |

---

## Essential G-code Commands

### Motion Commands (G-codes)

#### G0 - Rapid Positioning

**Purpose:** Fast movement without cutting (laser off)

**Format:** `G0 X<x> Y<y> [Z<z>] [F<speed>]`

**Example:**
```gcode
G0 X100 Y50  ; Move quickly to (100, 50)
```

**When Rayforge uses it:**
- Moving between cut paths
- Returning to origin
- Positioning before cutting

---

#### G1 - Linear Interpolation

**Purpose:** Controlled linear movement (cutting/engraving)

**Format:** `G1 X<x> Y<y> [Z<z>] F<speed>`

**Example:**
```gcode
G1 X100 Y50 F2000  ; Move to (100, 50) at 2000 mm/min
```

**When Rayforge uses it:**
- Cutting straight lines
- Engraving raster lines
- Any controlled movement with laser on

---

#### G2 / G3 - Arc Interpolation

**Purpose:** Circular/arc movements

**Format:** `G2/G3 X<x> Y<y> I<i> J<j> F<speed>`

- **G2** - Clockwise arc
- **G3** - Counter-clockwise arc
- **I, J** - Arc center offsets from start point

**Example:**
```gcode
G2 X50 Y50 I10 J0 F1500  ; Clockwise arc to (50,50), center offset (10,0)
```

**When Rayforge uses it:**
- Cutting circles and curves
- Rounded corners
- Optimized smooth curves

---

#### G4 - Dwell (Pause)

**Purpose:** Pause for specified time

**Format:** `G4 P<seconds>`

**Example:**
```gcode
G4 P0.5  ; Pause for 0.5 seconds
```

**When used:**
- Allowing laser to burn at a point (rare)
- Custom macros for material settling

---

### Setup Commands

#### G21 / G20 - Units

**Purpose:** Set measurement units

- **G21** - Millimeters (Rayforge default)
- **G20** - Inches

**Example:**
```gcode
G21  ; All coordinates in millimeters
```

---

#### G90 / G91 - Positioning Mode

**Purpose:** Set absolute or incremental positioning

- **G90** - Absolute (coordinates are absolute positions)
- **G91** - Incremental (coordinates are relative movements)

**Example:**
```gcode
G90        ; Absolute mode
G0 X10     ; Move to position X=10
G0 X20     ; Move to position X=20

G91        ; Incremental mode
G0 X10     ; Move 10mm in +X direction
G0 X10     ; Move another 10mm in +X direction (now at X=30 total)
```

**Rayforge default:** G90 (absolute mode)

---

### Machine Commands (M-codes)

#### M3 / M4 - Spindle/Laser On

**Purpose:** Turn laser on

- **M3** - Spindle mode (power varies with speed)
- **M4** - Laser mode (constant power, GRBL 1.1+)

**Format:** `M3/M4 S<power>`

**Example:**
```gcode
M4 S500  ; Laser on at power 500 (50% for 0-1000 range)
```

**Rayforge uses:** M4 for predictable laser power

---

#### M5 - Spindle/Laser Off

**Purpose:** Turn laser off

**Format:** `M5`

**Example:**
```gcode
M5  ; Laser off
```

**When Rayforge uses it:**
- End of each cut path
- Between operations
- End of job (safety)

---

#### M8 / M9 - Coolant/Air Assist

**Purpose:** Control air assist or coolant

- **M8** - Air assist on
- **M9** - Air assist off

**Example:**
```gcode
M8   ; Air assist on
; ...cutting...
M9   ; Air assist off
```

---

### Parameters

#### S - Spindle/Laser Power

**Format:** `S<value>`

**Range:** 0-1000 (GRBL), 0-255 (some firmwares)

**Example:**
```gcode
M4 S250   ; 25% power (for 0-1000 range)
M4 S750   ; 75% power
M4 S1000  ; 100% power
```

---

#### F - Feed Rate (Speed)

**Format:** `F<speed>`

**Units:** mm/min (or inches/min if G20)

**Example:**
```gcode
G1 X100 F3000  ; Move to X=100 at 3000 mm/min
```

**Modal:** Feed rate persists until changed
```gcode
G1 X10 F3000   ; Set speed to 3000
G1 X20         ; Still uses 3000 (no F needed)
G1 X30 F1500   ; Change speed to 1500
```

---

## Rayforge G-code Structure

### Typical File Structure

**Complete example:**

```gcode
; ==========================================
; Generated by Rayforge
; Job: test-project
; Date: 2025-10-03 15:30:00
; ==========================================

; === PREAMBLE ===
G21                    ; Set units to mm
G90                    ; Absolute positioning

; === JOB START HOOK (user-defined) ===
$H                     ; Home machine
M8                     ; Air assist on

; === LAYER 1: Engrave Layer ===
; Layer Start Hook (if defined)
G0 X10.000 Y10.000     ; Rapid to start position
M4 S400                ; Laser on at 40% power
G1 X50.000 F3000       ; Engrave line
G1 Y50.000             ; Engrave line
G1 X10.000             ; Engrave line
G1 Y10.000             ; Engrave line
M5                     ; Laser off
; Layer End Hook (if defined)

; === LAYER 2: Cut Layer ===
; Layer Start Hook (if defined)
G0 X20.000 Y20.000     ; Rapid to start position
M4 S900                ; Laser on at 90% power
G1 X60.000 F800        ; Cut line at slower speed
G1 Y60.000
G1 X20.000
G1 Y20.000
M5                     ; Laser off
; Layer End Hook (if defined)

; === JOB END HOOK (user-defined) ===
M5                     ; Ensure laser is off
M9                     ; Air assist off
G0 X0 Y0               ; Return to origin

; === END OF JOB ===
```

---

### Components Explained

#### 1. Header Comments

```gcode
; ==========================================
; Generated by Rayforge
; Job: test-project
; Date: 2025-10-03 15:30:00
; ==========================================
```

**Purpose:**
- Identify the file
- Track when it was generated
- Document job name

---

#### 2. Preamble

```gcode
G21  ; Set units to mm
G90  ; Absolute positioning
```

**Purpose:**
- Ensure machine is in correct mode
- Set up coordinate system
- Prevent mode confusion

---

#### 3. Job Start Hook

```gcode
$H   ; Home machine
M8   ; Air assist on
```

**Purpose:**
- User-defined setup commands
- Machine initialization
- Safety preparation

**Configured in:** Settings > Machine > Hooks

---

#### 4. Layer Operations

```gcode
; === LAYER 1: Engrave Layer ===
G0 X10.000 Y10.000     ; Rapid to start
M4 S400                ; Laser on
G1 X50.000 F3000       ; Cut
M5                     ; Laser off
```

**Purpose:**
- Execute each layer's operations
- Organized by layer order
- Comments indicate layer names

---

#### 5. Job End Hook

```gcode
M5        ; Ensure laser is off
M9        ; Air assist off
G0 X0 Y0  ; Return to origin
```

**Purpose:**
- User-defined cleanup commands
- Safety (laser off)
- Return to known position

---

## When to Edit G-code Manually

### Good Reasons to Edit

1. **Add pauses** for material changes or inspection
   ```gcode
   G4 P5  ; Pause 5 seconds
   ```

2. **Adjust power mid-job** for testing
   ```gcode
   M4 S300  ; Change to 30% power
   ```

3. **Fine-tune positions** for alignment
   ```gcode
   G0 X10.5 Y20.3  ; Shift by 0.5mm
   ```

4. **Insert custom commands** not supported by Rayforge UI

---

### Bad Reasons to Edit

1. **Complex changes** - Re-design in Rayforge instead
2. **Changing toolpaths** - Error-prone, use Rayforge operations
3. **Major restructuring** - Better to regenerate from source

---

### Editing Tips

**Before editing:**
- **Backup the file** - Keep original
- **Understand the commands** - Don't guess
- **Test on scrap** - Verify changes work

**Common edits:**
```gcode
; Add pause before cut
G4 P2

; Reduce power for testing
M4 S100  ; Was S500, testing at 10%

; Shift entire job by offset
; Find: G0 X10
; Replace: G0 X15  (shift 5mm right)
```

---

## Advanced G-code Topics

### Modal Commands

**Modal** means a command stays active until explicitly changed.

**Examples:**
- **G90** - Absolute mode (modal, stays until G91)
- **F3000** - Feed rate (modal, stays until changed)
- **M4** - Laser on (modal until M5)

**Implication:**
```gcode
G90          ; Set absolute mode (stays active)
F3000        ; Set feed rate (stays active)
G1 X10       ; Uses absolute mode and F3000
G1 X20       ; Still uses same mode and feed rate
G1 X30 F1500 ; Change feed rate, still absolute mode
```

---

### Coordinate Systems (G54-G59)

**What they are:** Multiple work coordinate systems for advanced setups.

**Commands:**
- **G54** - Work Coordinate System 1 (default)
- **G55** - Work Coordinate System 2
- **G56** - Work Coordinate System 3
- ... up to G59

**Use case:**
- Multiple fixture positions on bed
- Switch between jobs without re-homing

**Rayforge support:** Limited; primarily uses G54.

---

### Offsets (G92)

**Purpose:** Temporarily redefine coordinate zero.

**Format:** `G92 X<x> Y<y>`

**Example:**
```gcode
G92 X0 Y0  ; Set current position as (0, 0)
```

**Caution:**
- Can cause confusion
- Easy to forget offsets are active
- **Clear with:** `G92.1` (GRBL)

**Rayforge:** Avoids offsets, uses absolute positioning instead.

---

## Troubleshooting G-code Issues

### Commands Rejected by Machine

**Problem:** Machine sends error or alarm.

**Diagnosis:**

1. **Check G-code dialect** - GRBL vs Marlin vs Smoothieware
2. **Verify command syntax** - Typos or invalid parameters
3. **Check firmware version** - Some commands require newer firmware

**Solutions:**

- Use correct dialect in Rayforge settings
- Validate G-code with online checker (NCViewer, CAMotics)
- Update firmware if needed

---

### Unexpected Movement

**Problem:** Machine moves to wrong positions.

**Diagnosis:**

1. **Check positioning mode** - G90 (absolute) vs G91 (incremental)
2. **Check for offsets** - G92 may be active
3. **Check coordinates** - Out of range or incorrect

**Solutions:**

- Ensure G90 in preamble
- Clear offsets with G92.1
- Re-home machine

---

### Laser Not Turning On/Off

**Problem:** Laser stays on or doesn't activate.

**Diagnosis:**

1. **Check M commands** - M4/M5 present?
2. **Check S parameter** - Power value correct?
3. **Check firmware laser mode** - `$32=1` for GRBL

**Solutions:**

- Verify M4 S<power> before cuts
- Verify M5 after cuts
- Enable laser mode in firmware settings

---

## Best Practices

1. **Always include preamble** - G21, G90 to set modes
2. **Always end with M5** - Laser off for safety
3. **Use comments** - Document your changes
4. **Test on scrap** - Before production runs
5. **Validate G-code** - Use online simulators
6. **Keep backups** - Of working G-code files
7. **Know your firmware** - GRBL, Marlin, etc. differences

---

## Related Pages

- [G-code Dialects](../reference/gcode-dialects.md) - Firmware differences
- [Exporting G-code](../files/exporting.md) - Export settings and options
- [Firmware Compatibility](../reference/firmware.md) - Firmware versions and features
- [Macros & Hooks](../features/macros-hooks.md) - Custom G-code injection
- [GRBL Settings](../machine/grbl-settings.md) - Controller configuration
