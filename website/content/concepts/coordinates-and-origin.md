# Coordinate Systems and Origins

Understanding how RayForge handles coordinate systems is essential for positioning your work correctly and avoiding common mistakes.

## Overview

RayForge uses multiple coordinate systems and origin points to manage positioning. This can seem confusing at first, but each serves a specific purpose.

**Key concepts:**

- **Canvas coordinates** - Where you design (screen space)
- **Machine coordinates** - Physical machine position
- **Job origin** - Where the job starts on the machine
- **Workpiece coordinates** - Position within a design element

---

## Coordinate Systems Explained

### Canvas Coordinates

**What it is:** The design space where you arrange workpieces and layers.

**Characteristics:**
- **Units:** Millimeters (mm) by default
- **Origin:** Top-left corner of the canvas (0, 0)
- **X-axis:** Increases to the right
- **Y-axis:** Increases downward
- **Purpose:** Visual design and layout

**Example:**
```
(0,0) -----> X
  |
  |  Your design
  v  elements here
  Y
```

Canvas coordinates are what you see and interact with in the RayForge interface.

---

### Machine Coordinates

**What it is:** The physical coordinate system of your laser machine.

**Characteristics:**
- **Units:** Millimeters (mm)
- **Origin:** Machine home position (usually top-left or bottom-left)
- **X-axis:** Increases to the right
- **Y-axis:** Direction depends on machine configuration (up or down)
- **Purpose:** Physical movement commands

**Common machine origins:**

| Origin Position | Description | Common In |
|----------------|-------------|-----------|
| **Top-left** | (0,0) at back-left | Diode lasers |
| **Bottom-left** | (0,0) at front-left | CO2 lasers, CNC routers |
| **Center** | (0,0) at bed center | Some 3-axis machines |

**Machine coordinate example (top-left origin):**
```
(0,0) ----------------> X (max)
  |
  |  Work area
  |
  v
  Y (max)
```

---

### Job Origin

**What it is:** The reference point where your job will be positioned on the machine.

**Why it matters:**
- Determines where the laser starts cutting/engraving
- Allows you to position jobs anywhere on the bed
- Makes it easy to reposition or repeat jobs

**Setting job origin:**

1. **Design your work** in RayForge
2. **Position the job** on the canvas
3. **Set the origin mode** (absolute, current position, etc.)
4. **Frame the job** to verify position on the machine

**Job origin modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| **Absolute** | Job placed at specified machine coordinates | Repeatable positioning |
| **Current Position** | Job starts at current laser head position | Quick alignment |
| **User Origin** | Custom reference point | Advanced workflows |

---

### Workpiece Coordinates

**What it is:** The local coordinate system of each individual design element.

**Characteristics:**
- Each workpiece has its own origin (typically its bounding box corner)
- Transformations (rotate, scale, move) affect workpiece coordinates
- Operations reference workpiece coordinates

**Example:**

If you import an SVG logo:
- The logo has its own internal coordinate system
- RayForge places it on the canvas
- You can move/rotate/scale it
- The logo's local coordinates remain consistent

---

## Origin Points in Practice

### Job Origin vs Machine Origin

**Scenario:** You want to engrave a logo 50mm from the left edge and 100mm from the top of your machine bed.

**Solution:**

1. **Set machine origin** to top-left (if not already)
2. **Set job origin mode** to "Absolute"
3. **Position your job** at canvas coordinates (50, 100)
4. **Export G-code** - RayForge generates commands like:
   ```gcode
   G0 X50 Y100   ; Move to job start position
   ```

**Result:** The logo appears exactly where you specified on the machine.

---

### Current Position Origin

**Scenario:** You want to start a job wherever the laser head currently is.

**Solution:**

1. **Manually jog** the laser head to desired position
2. **Set job origin mode** to "Current Position"
3. **Run the job** - it starts from the current head position

**Use cases:**
- Quick test cuts without precise positioning
- Aligning to physical marks on material
- Iterative prototyping

---

### Repeating Jobs

**Scenario:** You need to cut 10 identical parts in a grid.

**Solution:**

1. **Design one part** in RayForge
2. **Duplicate the workpiece** 10 times
3. **Arrange in a grid** using alignment tools
4. **Set job origin** to absolute mode
5. **Export once** - all parts cut in sequence

**Benefit:** Canvas coordinates translate directly to machine coordinates, ensuring perfect spacing.

---

## Common Coordinate Mistakes

### Mistake 1: Wrong Machine Origin

**Problem:** Job appears in wrong location, possibly off the bed.

**Cause:** Machine origin configuration doesn't match actual machine.

**Example:**
- RayForge configured for top-left origin
- Machine actually has bottom-left origin
- Y-coordinates are inverted

**Solution:**

1. Check machine settings: **Settings > Machine > Profile**
2. Verify **Y-axis direction** setting
3. Test with a small frame job to confirm

---

### Mistake 2: Absolute vs Relative Positioning

**Problem:** Job moves from unexpected position.

**Cause:** Confusion between absolute (G90) and incremental (G91) modes.

**G-code modes:**
- **G90 (Absolute):** Coordinates are absolute positions
  ```gcode
  G0 X10 Y10  ; Move to position (10, 10)
  G0 X20 Y20  ; Move to position (20, 20)
  ```
- **G91 (Incremental):** Coordinates are relative movements
  ```gcode
  G0 X10 Y10  ; Move 10mm right, 10mm down from current position
  G0 X20 Y20  ; Move another 20mm right, 20mm down
  ```

**RayForge default:** G90 (absolute mode) for predictable positioning.

**Solution:** Check generated G-code preamble - should include `G90` command.

---

### Mistake 3: Job Exceeds Machine Bounds

**Problem:** Job starts but crashes into machine limits or fails with alarm.

**Cause:** Job positioned such that it extends beyond work area.

**Example:**
- Machine work area: 300mm x 200mm
- Job size: 100mm x 100mm
- Job origin: (250, 150)
- Job endpoint: (350, 250) - **exceeds bounds!**

**Prevention:**

1. Configure correct work area in **Settings > Machine > Profile**
2. RayForge will warn if job exceeds bounds
3. Use **Frame Job** feature to verify before cutting

---

### Mistake 4: Ignoring Overscan

**Problem:** Job positioned at very edge of work area, but overscan causes it to exceed bounds.

**Explanation:**

Overscan extends raster operations beyond the visible work area for quality:

```
         Visible area
    |<--------------->|
----|===============--|---- Raster line
    ^                 ^
  Overscan          Overscan
```

**Effective area:**
```
Work area: 300mm
Overscan: 5mm on each side
Usable area: 290mm (300 - 5 - 5)
```

**Solution:** Account for overscan when positioning jobs near edges.

---

## Coordinate Transformations

### Translation (Moving)

**What happens:** Workpiece coordinates shift by a fixed amount.

**Example:**
- Original position: (10, 20)
- Move by: (5, -3)
- New position: (15, 17)

**In RayForge:** Drag workpieces to move them, or use arrow keys for precise nudging.

---

### Rotation

**What happens:** Workpiece coordinates rotate around a center point.

**Example:**
- Square at (50, 50), size 20x20
- Rotate 45 degrees around center
- Corners move to new positions

**In RayForge:** Select workpiece and use rotation handles or transform tools.

**G-code impact:**
- RayForge pre-computes rotated coordinates
- G-code contains final absolute positions (no rotation commands)

---

### Scaling

**What happens:** Workpiece coordinates multiply by a scale factor.

**Example:**
- Original size: 50mm x 50mm
- Scale by 2x
- New size: 100mm x 100mm

**In RayForge:** Select workpiece and use scale handles or specify dimensions.

---

## Advanced Topics

### Work Coordinate Systems (G54-G59)

**What they are:** GRBL supports multiple coordinate systems for advanced workflows.

**G-codes:**
- **G54** - Work Coordinate System 1 (default)
- **G55** - Work Coordinate System 2
- **G56** - Work Coordinate System 3
- ... (up to G59)

**Use case:**
- Set up multiple fixture positions on the bed
- Switch between them with G-code commands
- Run same job at different locations

**RayForge support:** Limited. Primarily uses G54 (default). Advanced users can inject other coordinate systems via macros.

---

### Coordinate Offsets

**What they are:** Temporary shifts in the coordinate system.

**G-code commands:**
- **G92** - Set current position to specified coordinates (offset)
  ```gcode
  G92 X0 Y0  ; Define current position as (0,0)
  ```

**Caution:** Offsets can be confusing and lead to positioning errors. RayForge avoids them in favor of explicit absolute positioning.

---

### Homing and Machine Coordinates

**Homing ($H in GRBL):**
- Machine moves to physical limit switches
- Sets machine zero position
- Ensures repeatable positioning

**Why homing matters:**
- Without homing, machine doesn't know where it is
- Absolute positioning requires a known reference
- Homing provides that reference

**Best practice:** Home your machine before each session for consistent results.

---

## Practical Workflows

### Workflow 1: Centering a Job

**Goal:** Place a logo in the center of the work area.

**Steps:**

1. **Know your work area:** e.g., 300mm x 200mm
2. **Calculate center:** (150, 100)
3. **Know your logo size:** e.g., 50mm x 30mm
4. **Calculate logo origin:** Center - (logo size / 2)
   - X: 150 - (50/2) = 125mm
   - Y: 100 - (30/2) = 85mm
5. **Position logo** at (125, 85) on canvas
6. **Set job origin** to absolute
7. **Frame and run**

**Shortcut:** RayForge has alignment tools - select logo and use "Align Center" commands.

---

### Workflow 2: Aligning to Material

**Goal:** Align design to a pre-cut piece of material.

**Steps:**

1. **Place material** on machine bed
2. **Jog laser head** to bottom-left corner of material
3. **Set user origin** with G92 or controller command
4. **Set job origin** to current position
5. **Run job** - starts from the material corner

**Alternative (camera-based):**
1. **Use camera** to see material
2. **Align design** visually in RayForge
3. **Set job origin** based on camera reference
4. **Run job**

---

### Workflow 3: Production Grid

**Goal:** Cut multiple parts in a grid for production.

**Steps:**

1. **Design one part**
2. **Duplicate** to create grid (e.g., 3x3 grid, 10mm spacing)
3. **Position entire grid** on canvas to start at (10, 10)
4. **Set job origin** to absolute
5. **Export G-code** - all parts cut sequentially

**Coordinates in G-code:**
```gcode
; Part 1 at (10, 10)
G0 X10 Y10
; ...cut commands...

; Part 2 at (70, 10)  [10 + 50mm part + 10mm spacing]
G0 X70 Y10
; ...cut commands...
```

---

## Troubleshooting Coordinate Issues

### Job Positioned Incorrectly

**Diagnosis:**

1. **Frame the job** - does the frame match your expectation?
2. **Check machine origin** - top-left, bottom-left, or center?
3. **Check Y-axis direction** - does Y increase up or down?
4. **Check job origin mode** - absolute, current, or user?

**Solution:** Adjust machine settings to match your actual hardware.

---

### Job Runs Backwards or Mirrored

**Problem:** Design appears flipped on the machine.

**Cause:** Y-axis or X-axis direction mismatch.

**Solutions:**

- **In machine settings:** Toggle Y-axis direction
- **In design:** Manually flip the workpiece before export
- **In G-code:** Edit commands (advanced)

---

### Coordinates Off by Constant Amount

**Problem:** Job always off by same distance (e.g., 10mm to the right).

**Cause:** Job origin offset or work coordinate system offset.

**Solutions:**

1. **Check for G92 offset** - send `G92.1` to clear offsets
2. **Re-home machine** - ensure machine zero is correct
3. **Verify job origin position** - check canvas coordinates

---

### Random Position Each Time

**Problem:** Same job runs at different positions each time.

**Cause:** Using "Current Position" origin without consistent starting position.

**Solutions:**

- **Home machine** before each job
- **Use absolute positioning** instead of current position
- **Manually jog** to consistent position if needed

---

## Best Practices

1. **Always home your machine** at the start of each session
2. **Use absolute positioning (G90)** for repeatable jobs
3. **Configure machine settings** once and test thoroughly
4. **Frame jobs before running** to verify position
5. **Keep designs within work area** - account for overscan
6. **Use consistent origin modes** - don't switch mid-project
7. **Document your setup** - note origin position and Y-axis direction for future reference

---

## Related Pages

- [Machine Setup](../machine/device-config.md) - Configure machine dimensions and origin
- [GRBL Settings](../machine/grbl-settings.md) - Firmware coordinate settings
- [Exporting G-code](../files/exporting.md) - Job positioning options
- [Understanding Operations](understanding-operations.md) - How operations use coordinates
