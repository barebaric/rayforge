---
description:
  "Position workpieces precisely on the canvas. Use alignment tools, snapping, and coordinate input
  to place designs exactly where you need them."
---

# Workpiece Positioning Guide

This guide covers all the methods available in Rayforge for accurately positioning your workpiece
and aligning your designs before cutting or engraving.

## Overview

Accurate workpiece positioning is essential for:

- **Preventing waste**: Avoid cutting in the wrong location
- **Precise alignment**: Position designs on pre-printed materials
- **Repeatable results**: Run the same job multiple times consistently
- **Multi-part jobs**: Align multiple pieces on a single sheet

Rayforge provides several complementary tools for positioning:

| Method             | Purpose                 | Best For                               |
| ------------------ | ----------------------- | -------------------------------------- |
| **Focus Mode**     | See laser position      | Quick visual alignment                 |
| **Framing**        | Preview job bounds      | Verifying design fits on material      |
| **WCS Zero**       | Set coordinate origin   | Repeatable positioning                 |
| **Camera Overlay** | Visual design placement | Precise alignment on existing features |

---

## Focus Mode (Laser Pointer)

Focus mode turns on the laser at a low power level, acting as a "laser pointer" to help you see
exactly where the laser head is positioned.

### Enabling Focus Mode

1. **Connect to your machine**
2. **Click the Focus button** in the toolbar (laser icon)
3. The laser turns on at the configured focus power level
4. **Jog the laser head** to see the beam position on your material
5. **Click the Focus button again** to turn off when done

<!-- prettier-ignore-start -->
:::warning[Safety]
Even at low power, the laser can damage eyes. Never look directly into the beam or
point it at reflective surfaces. Wear appropriate eye protection.
:::
<!-- prettier-ignore-end -->

### Configuring Focus Power

The focus power determines how bright the laser dot appears:

1. Go to **Settings → Machine → Laser**
2. Find the **Focus Power** setting
3. Set a value that makes the dot visible without marking your material
   - Typical values: 1-5% for most materials
   - Set to 0 to disable the feature

<!-- prettier-ignore-start -->
:::tip[Finding the Right Power]
Start with 1% and increase gradually. The dot should be visible but
not leave any mark on your material. Darker materials may need higher power to see the dot clearly.
:::
<!-- prettier-ignore-end -->

### When to Use Focus Mode

- **Quick alignment checks**: See if the laser is roughly where you expect
- **Finding material edges**: Jog to corners to verify material placement
- **Setting WCS origin**: Position laser at desired zero point before setting WCS
- **Verifying home position**: Check that homing worked correctly

---

## Pointer Laser Offset

Some machines have a dedicated pointer laser (a small red dot laser) mounted at a fixed distance
from the cutting beam. When you align the stock using the pointer dot, the cutting beam would
actually land offset from that point — unless Rayforge compensates.

The [Pointer Offset](../machine/laser.md#pointer-offset) laser setting lets you enter that distance
and toggle the compensation on or off. With it enabled:

- **Set Work Zero at Current Position** (and Zero X / Zero Y) places the work origin at the pointer
  dot's position.
- The canvas shows a yellow pointer dot next to the red beam dot, marking where the pointer dot is
  on your material.
- A **Pointer Alignment** switch appears in the move-to popover (the compass icon next to the
  position readout).

### Pointer Alignment

While **Pointer Alignment** is on, every absolute aiming operation — Move-To, the corner shortcuts,
moving to the WCS origin, Click-to-Move, Move-Head-Here, and framing — lands the _pointer dot_ on
the aimed position, so you can position the stock entirely by the visible dot. The coordinate entry
in the popover pre-fills with the pointer dot's position. On the canvas, exactly one dot is ever
filled: the pointer dot is filled while alignment is on (the beam dot is a hollow ring then), and
hollow while it is off (the beam dot is filled).

The workflow composes cleanly with zeroing by the pointer dot: the origin sits where the pointer
marked, aiming shifts every target by the offset, and the burn is unshifted — so aligning with the
dot and cutting with the beam end up consistent.

When you press **Send** while alignment is on, a warning appears on every send (there is no "don't
ask again"). You can choose:

- **Dry-Run with Pointer**: the job runs with the pointer offset applied, so the pointer dot traces
  the toolpath while the beam runs displaced by the offset. The laser still fires at job power —
  make sure the displaced beam cannot hit anything it should not.
- **Turn Off and Burn**: alignment is switched off and the job burns normally with the beam at the
  WCS positions.
- **Cancel**.

Jog moves and jobs are never shifted by the toggle: jog is relative, and regular G-code output is
identical whether alignment is on or off. The switch is session-only — it is not saved to the
machine profile and resets when you switch machines.

The pointer offset is especially useful for stock that is larger than the laser bed (pass-through
work), where you repeatedly align the design to a reference mark on the moving stock.

---

## Framing

Framing traces the bounding rectangle of your job at low (or zero) power, showing exactly where your
design will be cut or engraved.

### How to Frame

1. **Load and position your design** in Rayforge
2. **Click Machine → Frame** or press `Ctrl+F`
3. The laser head traces the bounding box of your job
4. **Verify the outline** fits within your material

### Frame Settings

Configure framing behavior in **Settings → Machine → Laser**:

- **Frame Speed**: How fast the head moves during framing (slower = easier to see)
- **Frame Power**: Laser power during framing
  - Set to 0 for air framing (laser off, just movement)
  - Set to 1-5% for a visible trace on the material

<!-- prettier-ignore-start -->
:::tip[Air Framing vs. Low Power]
- **Air framing (0% power)**: Safe for any material, but you only see the head movement
- **Low power framing**: Leaves a faint visible mark, useful for precise alignment on dark materials
:::
<!-- prettier-ignore-end -->

### When to Frame

- **Before every job**: Quick verification that design fits
- **After positioning changes**: Confirm new placement is correct
- **Expensive materials**: Double-check before committing
- **Multi-part jobs**: Verify all parts fit on the material

See [Framing Your Job](framing-your-job) for more details.

---

## Setting WCS Zero (Work Coordinate System)

Work Coordinate Systems (WCS) let you define custom "zero points" for your jobs. This makes it easy
to align jobs to your material position.

### Quick WCS Setup

1. **Jog the laser head** to the corner of your material (or desired origin point)
2. **Open the Control Panel** (`Ctrl+L`)
3. **Select a WCS** (G54 is the default work coordinate system)
4. **Click Zero X** and **Zero Y** to set current position as origin
5. Your design's (0,0) point will now align with this position

### Understanding Coordinate Systems

Rayforge uses several coordinate systems:

| System      | Description                                    |
| ----------- | ---------------------------------------------- |
| **G53**     | Machine coordinates (fixed, cannot be changed) |
| **G54**     | Work coordinate system 1 (default)             |
| **G55-G59** | Additional work coordinate systems             |

<!-- prettier-ignore-start -->
:::tip[Multiple Work Areas]
Use different WCS slots for different fixture positions. For example:

- G54 for the left side of your bed
- G55 for the right side
- G56 for a rotary attachment
:::
<!-- prettier-ignore-end -->

### When to Set WCS Zero

- **New material placement**: Align origin to material corner
- **Fixture work**: Set origin to fixture reference point
- **Repeatable jobs**: Same job, different positions
- **Production runs**: Consistent positioning across multiple pieces

See [Work Coordinate Systems](../general-info/coordinate-systems.md) for complete documentation.

---

## Camera-Based Positioning

The camera overlay shows a live view of your material with your design superimposed, enabling
precise visual alignment.

### Setting Up the Camera

1. **Connect a USB camera** above your work area
2. Go to **Settings → Camera** and add your camera device
3. **Enable the camera** to see the overlay on your canvas
4. **Align the camera** using the alignment procedure (required for accurate positioning)

### Camera Alignment

Camera alignment maps camera pixels to real-world coordinates:

1. Open **Camera → Align Camera**
2. Place alignment markers at known positions (at least 4 points)
3. Enter the real-world X/Y coordinates for each point
4. Click **Apply** to calculate the transformation

<!-- prettier-ignore-start -->
:::tip[Alignment Accuracy]
- Use points spread across your entire work area
- Measure world coordinates carefully with a ruler
- Use machine positions (jog to known coordinates) for best accuracy
:::
<!-- prettier-ignore-end -->

### Positioning with Camera Overlay

1. **Enable the camera overlay** to see your material
2. **Import your design**
3. **Drag the design** to align with features visible in the camera
4. **Fine-tune** using arrow keys for pixel-perfect placement
5. **Frame to verify** before running the job

### When to Use Camera Positioning

- **Pre-printed materials**: Align cuts to existing prints
- **Irregular materials**: Position on non-rectangular pieces
- **Precise placement**: Sub-millimeter accuracy requirements
- **Complex layouts**: Multiple items with specific spacing

See [Camera Integration](../machine/camera.md) for complete documentation.

---

## Recommended Workflows

### Basic Positioning Workflow

For simple jobs on rectangular materials:

1. **Place material** on the laser bed
2. **Enable focus mode** and jog to verify material position
3. **Set WCS zero** at the material corner
4. **Position your design** in the canvas
5. **Frame the job** to verify placement
6. **Run the job**

### Precision Alignment Workflow

For accurate placement on pre-printed or marked materials:

1. **Set up and align camera** (one-time setup)
2. **Place material** on the laser bed
3. **Enable camera overlay** to see the material
4. **Import and position design** visually on the camera image
5. **Disable camera** and frame to verify
6. **Run the job**

### Pass-Through Workflow

For stock longer than the laser bed (fed through the machine in segments), a
[pointer laser offset](#pointer-laser-offset) removes the manual guesswork:

1. **Cut segment 1** of your design normally.
2. **Feed the stock forward** through the pass-through so the next segment is on the bed.
3. **Jog the machine** until the pointer dot marks a reference point on the stock (e.g. a corner of
   an already-cut feature).
4. **Set WCS zero** (or Zero X / Zero Y) — with the pointer offset enabled, the origin lands exactly
   where the pointer pointed.
5. **Position the next segment** of the design relative to that origin in the canvas.
6. **Frame to verify**, then **run the job**.
7. Repeat from step 2 for each remaining segment.

### Production Workflow

For running multiple identical jobs:

1. **Set up fixture** on the laser bed
2. **Set WCS zero** aligned to the fixture (e.g., G54)
3. **Load and configure** your design
4. **Frame to verify** alignment with fixture
5. **Run the job**
6. **Replace material** and repeat (WCS stays the same)

### Multi-Position Workflow

For running the same job at different locations:

1. **Set up multiple WCS positions**:
   - Jog to position 1, set G54 zero
   - Jog to position 2, set G55 zero
   - Jog to position 3, set G56 zero
2. **Load your design** (same design for all positions)
3. **Select G54**, frame, and run
4. **Select G55**, frame, and run
5. **Select G56**, frame, and run

---

## Troubleshooting

### Laser dot not visible in focus mode

- **Increase focus power** in laser settings
- **Dark materials** may require higher power (5-10%)
- **Check laser connection** and ensure machine is responding
- **Verify focus power** is not set to 0

### Camera overlay misaligned

- **Re-run camera alignment** with more reference points
- **Check camera mounting** - it may have moved
- **Verify world coordinates** were measured accurately
- **See camera troubleshooting** in Camera Integration docs

---

## Related Topics

- [Framing Your Job](framing-your-job) - Detailed framing documentation
- [Work Coordinate Systems](../general-info/coordinate-systems.md) - WCS reference
- [Camera Integration](../machine/camera.md) - Camera setup and alignment
- [Control Panel](../ui/bottom-panel.md) - Jog controls and WCS management
- [Quick Start Guide](../getting-started/quick-start.md) - Basic workflow
