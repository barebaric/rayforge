# Raster Engraving

Raster engraving fills areas with back-and-forth scanning lines, similar to how an inkjet printer works. It's ideal for creating images, text, and filled designs on materials like wood, leather, and anodized aluminum.

## Overview

Raster operations:

- Fill closed shapes with scanning lines
- Support variable power for grayscale images
- Work with both vector shapes and bitmap images
- Use bidirectional scanning for speed
- Create permanent marks on many materials

<!-- SCREENSHOT: feature-raster-example
description: Canvas showing text and graphics filled with raster engraving pattern, demonstrating the horizontal scanning lines
filename: Ref-Raster-Example.png
-->

<!-- ![Raster engraving example](../../images/Ref-Raster-Example.png) -->

## When to Use Raster

Use raster engraving for:

-  Engraving text and logos
-  Creating images and photos on wood/leather
-  Filling solid areas with texture
-  Marking parts and products
-  Creating grayscale artwork

**Don't use raster for:**
- L Cutting through material (use [Contour](contour.md) instead)
- L Precise outlines (raster creates filled areas)
- L Fine line work (vectors are cleaner)

## Creating a Raster Operation

### Step 1: Prepare Content

Raster works with:
- **Vector shapes** - Filled with scanning lines
- **Text** - Converted to filled paths
- **Images** - Converted to grayscale and engraved

### Step 2: Add Raster Operation

- **Menu:** Operations  Add Raster
- **Shortcut:** ++ctrl+shift+r++
- **Right-click:** Context menu  Add Operation  Raster

### Step 3: Configure Settings

<!-- SCREENSHOT
id: ui-raster-settings
type: screenshot
size: dialog
description: |
  Raster operation settings dialog showing:
  - Power: 40%
  - Speed: 3000 mm/min
  - Line interval: 0.1mm
  - Scan angle: 0
  - Bidirectional: enabled
  - Overscan: 2mm
setup:
  - action: create_text
    text: "ENGRAVED"
  - action: add_operation
    type: raster
  - action: open_settings
  - action: set_parameters
    power: 40
    speed: 3000
    line_interval: 0.1
    scan_angle: 0
    bidirectional: true
    overscan: 2
  - action: capture
    region: dialog
filename: UI-Raster-Settings.png
alt: "Raster operation settings dialog"
-->

<!-- ![Raster settings](../../images/UI-Raster-Settings.png) -->

## Key Settings

### Power & Speed

**Power (%):**
- Laser intensity for the engraving
- Lower power for lighter marking
- Higher power for deeper engraving
- Typical range: 20-60% for engraving

**Speed (mm/min):**
- How fast the laser scans
- Faster = lighter, slower = darker
- Typical range: 2000-5000 mm/min

**Starting values (wood engraving):**
- Power: 30-50%
- Speed: 2500-4000 mm/min
- Always test on scrap material!

### Line Interval

**Line Interval (mm):**
- Spacing between scan lines
- Smaller = higher quality, longer job time
- Larger = faster, visible lines

| Interval | Quality | Speed | Use For |
|----------|---------|-------|---------|
| 0.05mm | Highest | Slowest | Photos, fine detail |
| 0.1mm | High | Medium | Text, logos, graphics |
| 0.2mm | Medium | Fast | Solid fills, textures |
| 0.3mm+ | Low | Fastest | Draft, testing |

**Recommended:** 0.1mm for general use

!!! tip "Resolution Match"
    For images, line interval should match or exceed image resolution. If your image is 10 pixels/mm (254 DPI), use 0.1mm line interval or smaller.

### Scan Direction

**Scan Angle (degrees):**
- Direction of scan lines
- 0 = horizontal (left to right)
- 90 = vertical (top to bottom)
- 45 = diagonal

**Why change angle?**
- Wood grain: Engrave perpendicular to grain for better results
- Pattern orientation: Match design aesthetics
- Reduce banding: Different angle can hide imperfections

**Bidirectional Scanning:**
-  **Enabled:** Laser engraves in both directions (faster)
- L **Disabled:** Laser only engraves left-to-right (slower, more consistent)

For best quality, disable bidirectional. For speed, enable it.

### Overscan

**Overscan Distance (mm):**
- How far beyond the design the laser travels before turning around
- Allows laser to reach full speed before entering the design
- Prevents burn marks at line starts/ends

**Typical values:**
- 2-5mm for most jobs
- Larger for high speeds
- See [Overscan & Kerf](../overscan-kerf.md) for details

<!-- SCREENSHOT: ref-raster-overscan
description: Diagram showing raster scan lines extending beyond the design boundary, illustrating overscan distance
filename: Diag-Raster-Overscan.png
-->

<!-- ![Raster overscan diagram](../../images/Diag-Raster-Overscan.png) -->

## Grayscale Images

Raster operations can vary laser power based on image brightness:

### Image Preparation

1. **Convert to grayscale** - Color images are converted automatically
2. **Adjust contrast** - Increase contrast for better engraving
3. **Increase brightness** - Dark images may over-engrave
4. **Resize** - Match your desired output size

### Power Mapping

- **White pixels**  Laser off (0% power)
- **Gray pixels**  Proportional power (e.g., 50% gray = 50% of max power)
- **Black pixels**  Maximum power setting

**Example:**
- Max power setting: 60%
- 50% gray pixel  30% actual power
- 100% black pixel  60% actual power

### Resolution

**Image DPI vs Line Interval:**
- 254 DPI = 10 pixels/mm  use 0.1mm line interval
- 508 DPI = 20 pixels/mm  use 0.05mm line interval

Higher resolution images require smaller line intervals for quality.

## Tips & Best Practices

### Material Selection

 **Best materials for raster:**
- Wood (natural variations create beautiful results)
- Leather (burns to dark brown/black)
- Anodized aluminum (removes coating, reveals metal)
- Coated metals (removes coating layer)
- Some plastics (test first!)

 **Challenging materials:**
- Clear acrylic (doesn't show engraving well)
- Metals without coating (requires special marking compounds)
- Glass (requires special settings/coatings)

### Quality Settings

 **For best quality:**
- Use smaller line interval (0.05-0.1mm)
- Disable bidirectional scanning
- Increase overscan (3-5mm)
- Use lower power, multiple passes
- Ensure material is flat and secured

 **For faster engraving:**
- Use larger line interval (0.15-0.2mm)
- Enable bidirectional scanning
- Minimum overscan (1-2mm)
- Single pass at higher power

### Common Issues

**Burn marks at line ends:**
- Increase overscan distance
- Check acceleration settings
- Reduce power slightly

**Visible scan lines:**
- Decrease line interval
- Reduce power (over-burning creates gaps)
- Check that material is flat

**Uneven engraving:**
- Ensure material is flat
- Check focus consistency
- Verify laser power stability
- Clean laser lens

**Banding (dark/light stripes):**
- Disable bidirectional scanning
- Check belt tension
- Reduce speed
- Try different scan angle

### Material Settings

| Material | Power | Speed | Line Interval | Notes |
|----------|-------|-------|--------------|-------|
| Birch plywood | 35-45% | 3000-4000 mm/min | 0.1mm | Nice contrast |
| Cherry wood | 30-40% | 3500-4500 mm/min | 0.1mm | Rich dark color |
| Leather | 25-35% | 3000-4000 mm/min | 0.1mm | Test on scrap |
| Anodized aluminum | 40-50% | 4000-5000 mm/min | 0.1mm | Removes coating |
| Cork | 20-30% | 3500-4500 mm/min | 0.15mm | Very forgiving |

*Settings vary by laser type and material quality. Always test!*

## Advanced Techniques

### Crosshatch Engraving

Run raster operation twice at different angles:

1. First pass: 0 (horizontal)
2. Second pass: 90 (vertical)

Creates deeper, more uniform engraving with crosshatch pattern.

### Variable Line Interval

For large areas:
- Use fine interval (0.05mm) for detailed areas
- Use coarse interval (0.2mm) for solid fills
- Combine multiple raster operations

### Dithering

For photo engraving:
- Convert images to dithered black & white
- Creates halftone-like effect
- Better detail than pure grayscale on some materials

## Troubleshooting

### Engraving too light

- **Increase:** Power setting
- **Decrease:** Speed setting
- **Check:** Focus is correct
- **Try:** Multiple passes

### Engraving too dark/burning

- **Decrease:** Power setting
- **Increase:** Speed setting
- **Increase:** Line interval
- **Check:** Material is appropriate

### Inconsistent darkness

- **Check:** Material is flat
- **Check:** Focus distance is consistent
- **Verify:** Laser beam is clean
- **Test:** Different area of material (grain varies)

### Image looks pixelated

- **Decrease:** Line interval
- **Check:** Source image resolution
- **Try:** Smaller line interval (0.05mm)
- **Verify:** Image isn't being upscaled

### Scan lines visible

- **Decrease:** Line interval
- **Reduce:** Power (over-burning creates gaps)
- **Try:** Different scan angle
- **Ensure:** Material surface is smooth

## Technical Details

### Scan Pattern Generation

RayForge converts filled shapes to scan lines:

1. Determine bounding box
2. Calculate scan lines at specified angle and interval
3. Clip lines to shape boundaries
4. Add overscan extensions
5. Optimize scan order (reduce travel time)
6. Generate G-code

### Bidirectional Scanning

**Enabled (faster):**
```
 Scan line 1 
 Scan line 2 
 Scan line 3 
```

**Disabled (more consistent):**
```
 Scan line 1 
  (return)
 Scan line 2 
  (return)
 Scan line 3 
```

### G-code Example

```gcode
G0 X-2 Y10          ; Move to overscan start
M3 S102             ; Laser on at 40% power
G1 X52 Y10 F3000    ; Scan line at 3000 mm/min
M5                  ; Laser off
G0 X52 Y10.1        ; Move to next line
M3 S102             ; Laser on
G1 X-2 Y10.1 F3000  ; Scan back (bidirectional)
M5                  ; Laser off
```

## Related Topics

- **[Contour Cutting](contour.md)** - Cutting outlines and shapes
- **[Depth Engraving](depth.md)** - Creating 3D relief effects
- **[Overscan & Kerf](../overscan-kerf.md)** - Improving engraving quality
- **[Material Test Grid](material-test-grid.md)** - Finding optimal settings
- **[Multi-Layer Workflow](../multi-layer.md)** - Combining raster with other operations
