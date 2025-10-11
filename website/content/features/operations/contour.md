# Contour Cutting

Contour cutting traces the outline of vector shapes to cut them free from material. It's the most common laser operation for creating parts, signs, and decorative pieces.

## Overview

Contour operations:

- Follow vector paths (lines, curves, shapes)
- Cut along the perimeter of objects
- Support single or multiple passes for thick materials
- Can use inside, outside, or on-line cutting paths
- Work with any closed or open vector shape

<!-- SCREENSHOT: feature-contour-example
description: Canvas showing a design with contour cut paths highlighted, demonstrating cutting around vector shapes
filename: Ref-Contour-Example.png
-->

<!-- ![Contour cutting example](../../images/Ref-Contour-Example.png) -->

## When to Use Contour

Use contour cutting for:

-  Cutting parts free from stock material
-  Creating outlines and borders
-  Cutting shapes from wood, acrylic, cardboard
-  Perforating or scoring (with reduced power)
-  Creating stencils and templates

**Don't use contour for:**
- L Filling areas (use [Raster](raster.md) instead)
- L Creating 3D depth effects (use [Depth Engraving](depth.md) instead)
- L Bitmap images (convert to vectors first)

## Creating a Contour Operation

### Step 1: Select Objects

1. Import or draw vector shapes on the canvas
2. Select the objects you want to cut
3. Ensure shapes are closed paths for complete cuts

### Step 2: Add Contour Operation

- **Menu:** Operations  Add Contour
- **Shortcut:** ++ctrl+shift+c++
- **Right-click:** Context menu  Add Operation  Contour

### Step 3: Configure Settings

<!-- SCREENSHOT
id: ui-contour-settings
type: screenshot
size: dialog
description: |
  Contour operation settings dialog showing all configuration options:
  - Power: 80%
  - Speed: 500 mm/min
  - Passes: 2
  - Pass depth: 1.5mm
  - Path offset: On line
  - Direction: Clockwise
setup:
  - action: open_example
    name: simple-shapes
  - action: select_all
  - action: add_operation
    type: contour
  - action: open_settings
  - action: set_parameters
    power: 80
    speed: 500
    passes: 2
    pass_depth: 1.5
  - action: capture
    region: dialog
filename: UI-Contour-Settings.png
alt: "Contour operation settings dialog"
-->

<!-- ![Contour settings](../../images/UI-Contour-Settings.png) -->

## Key Settings

### Power & Speed

**Power (%):**
- Laser intensity from 0-100%
- Higher power for thicker materials
- Lower power for scoring or marking

**Speed (mm/min):**
- How fast the laser moves
- Slower = more energy = deeper cut
- Faster = less energy = lighter cut

**Starting values (3mm plywood):**
- Power: 70-90%
- Speed: 300-600 mm/min
- Always test on scrap material first!

### Multi-Pass Cutting

For materials thicker than a single pass can cut:

**Passes:**
- Number of times to repeat the cut
- Each pass cuts deeper
- Typical: 1-3 passes for 3mm material

**Pass Depth (Z-step):**
- How much to lower Z-axis per pass (if supported)
- Requires Z-axis control on your machine
- Creates true 2.5D cutting
- Set to 0 for same-depth multiple passes

**Example for 6mm plywood:**
- Passes: 3
- Pass depth: 2mm per pass
- Result: Cuts 6mm total depth

!!! warning "Z-Axis Required"
    Pass depth only works if your machine has Z-axis control. For machines without Z-axis, use multiple passes at the same depth.

### Path Offset

Controls where the laser cuts relative to the vector path:

| Offset | Description | Use For |
|--------|-------------|---------|
| **On Line** | Cuts directly on the path | Centerline cuts, scoring |
| **Inside** | Cuts inside the shape | Parts that must fit exact size |
| **Outside** | Cuts outside the shape | Holes that parts fit into |

**Offset Distance:**
- How far inside/outside to offset (mm)
- Typically set to half your kerf width
- Kerf = width of material removed by laser
- Example: 0.15mm offset for 0.3mm kerf

<!-- SCREENSHOT: ref-contour-offset
description: Diagram showing three squares with on-line, inside, and outside offset paths illustrated
filename: Diag-Contour-PathOffset.png
-->

<!-- ![Path offset diagram](../../images/Diag-Contour-PathOffset.png) -->

### Cut Direction

**Clockwise vs Counter-Clockwise:**
- Affects which side of the cut gets more heat
- Usually clockwise for right-hand rule
- Change if one side burns more than the other

**Optimize Order:**
- Automatically sorts paths for minimum travel
- Reduces job time
- Prevents missed cuts

## Advanced Features

### Holding Tabs

Tabs keep cut pieces attached to stock material during cutting:

- Add tabs to prevent pieces from falling
- Tabs are small uncut sections
- Break tabs after job completes
- See [Holding Tabs](../holding-tabs.md) for details

<!-- SCREENSHOT: ref-contour-tabs
description: Contour path with holding tabs shown as small gaps in the cutting line
filename: Ref-Contour-HoldingTabs.png
-->

<!-- ![Contour with holding tabs](../../images/Ref-Contour-HoldingTabs.png) -->

### Kerf Compensation

Kerf is the width of material removed by the laser beam:

**Why it matters:**
- A circle cut "on line" will be slightly smaller than designed
- The laser removes ~0.2-0.4mm of material (depending on beam width)

**How to compensate:**
1. Measure your kerf on test cuts
2. Use path offset = kerf/2
3. For parts: offset **inside** by kerf/2
4. For holes: offset **outside** by kerf/2

See [Overscan & Kerf](../overscan-kerf.md) for detailed guide.

### Lead-In/Lead-Out

Lead-ins and lead-outs control where cuts start and end:

**Lead-in:**
- Gradual entry to the cut
- Prevents burn marks at start point
- Moves laser to full speed before hitting the material edge

**Lead-out:**
- Gradual exit from the cut
- Prevents damage at end point
- Common for metals and acrylics

**Configuration:**
- Length: How far the lead extends (mm)
- Angle: Direction of the lead path
- Type: Straight line, arc, or spiral

## Tips & Best Practices

### Material Testing

 **Always test first:**
1. Cut small test shapes on scrap
2. Start with conservative settings (lower power, slower speed)
3. Gradually increase power or decrease speed
4. Record successful settings

### Cutting Order

 **Best practices:**
- Engrave before cutting (keeps material secured)
- Cut inside features before outside perimeter
- Use holding tabs for parts that might move
- Cut smallest parts first (less vibration)

### Quality Cutting

 **For clean cuts:**
- Use air assist (blows debris away)
- Ensure proper focus (sharp beam = clean cut)
- Multiple faster passes often better than one slow pass
- Keep material flat and secured

  **Avoid:**
- Cutting at maximum power (creates excessive charring)
- Single slow pass (more heat = more burning)
- Skipping material tests
- Cutting warped or unsecured material

### Common Materials

| Material | Thickness | Power | Speed | Passes | Notes |
|----------|-----------|-------|-------|--------|-------|
| Paper | 0.2mm | 10-20% | 3000+ mm/min | 1 | Very fast |
| Cardboard | 3mm | 40-60% | 1500-2500 mm/min | 1 | Test for char |
| Plywood | 3mm | 70-90% | 300-600 mm/min | 1-2 | Use air assist |
| Acrylic | 3mm | 80-95% | 200-400 mm/min | 1-2 | Slow for clean edges |
| MDF | 3mm | 70-85% | 400-700 mm/min | 1-2 | Very dusty |

*Settings vary by laser type, power, and beam quality. Always test!*

## Troubleshooting

### Cuts not going through material

- **Increase:** Power setting
- **Decrease:** Speed setting
- **Add:** More passes
- **Check:** Focus is correct
- **Check:** Beam is clean (dirty lens)

### Excessive charring or burning

- **Decrease:** Power setting
- **Increase:** Speed setting
- **Use:** Air assist
- **Try:** Multiple faster passes instead of one slow
- **Check:** Material is appropriate for laser cutting

### Parts fall out during cutting

- **Add:** [Holding tabs](../holding-tabs.md)
- **Use:** Cutting order optimization
- **Cut:** Inside features before outside
- **Ensure:** Material is flat and secured

### Inconsistent cut depth

- **Check:** Material thickness is uniform
- **Check:** Material is flat (not warped)
- **Check:** Focus distance is consistent
- **Verify:** Laser power is stable

### Missed corners or curves

- **Decrease:** Speed (especially on corners)
- **Check:** Machine acceleration settings
- **Verify:** Belts are tight
- **Reduce:** Path complexity (simplify curves)

## Technical Details

### Coordinate System

Contour operations work in:
- **Units:** Millimeters (mm)
- **Origin:** Depends on machine and job setup
- **Coordinates:** X/Y plane (Z for multi-pass depth)

### Path Generation

RayForge converts vector shapes to G-code:

1. Offset path (if inside/outside cutting)
2. Optimize path order (minimize travel)
3. Insert lead-in/lead-out (if configured)
4. Add holding tabs (if configured)
5. Generate G-code commands

### G-code Commands

Typical contour G-code:
```gcode
G0 X10 Y10          ; Rapid move to start
M3 S204             ; Laser on at 80% power
G1 X50 Y10 F500     ; Cut to point at 500 mm/min
G1 X50 Y50 F500     ; Cut to next point
G1 X10 Y50 F500     ; Continue cutting
G1 X10 Y10 F500     ; Complete the square
M5                  ; Laser off
```

## Related Topics

- **[Raster Engraving](raster.md)** - Filling areas with engraving patterns
- **[Depth Engraving](depth.md)** - Creating 3D relief effects
- **[Holding Tabs](../holding-tabs.md)** - Keeping parts secured during cutting
- **[Overscan & Kerf](../overscan-kerf.md)** - Improving cut quality and accuracy
- **[Material Test Grid](material-test-grid.md)** - Finding optimal power/speed settings
