# Depth Engraving

Depth engraving creates 3D relief effects by varying laser power based on image brightness. It's used for creating dimensional artwork, terrain maps, lithophanes, and embossed designs.

## Overview

Depth engraving operations:

- Vary laser power to create different engraving depths
- Work from grayscale images or height maps
- Create 3D relief effects on wood, acrylic, and coated materials
- Support multiple passes for deeper carving
- Produce tactile, three-dimensional results

<!-- SCREENSHOT: feature-depth-example
description: Canvas showing a grayscale image that will be depth engraved, with example of resulting 3D relief effect
filename: Ref-Depth-Example.png
-->

<!-- ![Depth engraving example](../../images/Ref-Depth-Example.png) -->

## When to Use Depth Engraving

Use depth engraving for:

-  Creating 3D portraits and artwork
-  Terrain and topographic maps
-  Lithophanes (light-transmitting 3D images)
-  Embossed logos and designs
-  Textured surfaces and patterns
-  Mold masters for casting

**Don't use depth for:**
- L Simple flat engraving (use [Raster](raster.md) instead)
- L Cutting through material (use [Contour](contour.md))
- L Text and line art (better with raster)

## How Depth Engraving Works

### Power-to-Depth Relationship

The laser engraves deeper with higher power:

- **Low power** (lighter pixels)  Shallow engraving
- **Medium power** (medium pixels)  Medium depth
- **High power** (dark pixels)  Deep engraving

**Image brightness to depth:**
```
White pixel   0% power    No engraving
Light gray    25% power   Light engraving
Medium gray   50% power   Medium depth
Dark gray     75% power   Deep engraving
Black pixel   100% power  Maximum depth
```

### Scanning Pattern

Like raster engraving, depth uses scanning lines:
- Back-and-forth horizontal passes
- Power varies along each line based on image
- Creates smooth gradients and transitions

## Creating a Depth Operation

### Step 1: Prepare Image

1. **Convert to grayscale** - Remove all color
2. **Adjust levels** - Increase contrast for more dramatic depth
3. **Invert if needed** - White=high, Black=low (or vice versa)
4. **Resize** - Match your desired output size

**Image tips:**
- Higher contrast = more dramatic depth changes
- Smooth gradients = smooth 3D surfaces
- Sharp edges = steep depth transitions

### Step 2: Import and Add Operation

1. Import your grayscale image
2. Select the image
3. **Menu:** Operations  Add Depth Engraving
4. **Shortcut:** ++ctrl+shift+d++

### Step 3: Configure Settings

<!-- SCREENSHOT
id: ui-depth-settings
type: screenshot
size: dialog
description: |
  Depth engraving operation settings dialog showing:
  - Min power: 10%
  - Max power: 60%
  - Speed: 2000 mm/min
  - Line interval: 0.08mm
  - Scan angle: 0
  - Passes: 2
  - Invert: disabled
setup:
  - action: open_image
    file: examples/portrait.png
  - action: add_operation
    type: depth
  - action: open_settings
  - action: set_parameters
    min_power: 10
    max_power: 60
    speed: 2000
    line_interval: 0.08
    passes: 2
  - action: capture
    region: dialog
filename: UI-Depth-Settings.png
alt: "Depth engraving operation settings dialog"
-->

<!-- ![Depth settings](../../images/UI-Depth-Settings.png) -->

## Key Settings

### Power Range

**Minimum Power (%):**
- Laser power for lightest areas (white pixels)
- Usually 0-20%
- Set higher to avoid very shallow areas

**Maximum Power (%):**
- Laser power for darkest areas (black pixels)
- Usually 40-80% depending on material
- Lower = subtle relief, higher = dramatic depth

**Power Range Examples:**

| Min | Max | Effect |
|-----|-----|--------|
| 0% | 40% | Subtle, light relief |
| 10% | 60% | Medium depth, safe |
| 20% | 80% | Deep, dramatic relief |

**Start conservative:**
- Test with 10-50% range
- Increase max power for more depth
- Increase min power to avoid unengraved areas

### Speed

**Speed (mm/min):**
- How fast the laser scans
- Slower = deeper engraving at same power
- Typical: 1500-3000 mm/min

**Speed vs Depth:**
- Half the speed = roughly double the depth
- Very slow speeds can char or burn
- Test to find optimal speed for material

### Line Interval

**Line Interval (mm):**
- Spacing between scan lines
- Smaller = smoother 3D surface
- Typical: 0.05-0.15mm

**For depth engraving:**
- Use finer intervals than normal raster (0.05-0.08mm)
- Smooth 3D surfaces need close line spacing
- Visible lines create ribbed texture

### Multiple Passes

**Passes:**
- Number of times to repeat the engraving
- Each pass adds more depth
- Typical: 1-3 passes

**Why multiple passes?**
- Safer (less risk of burning)
- Deeper total depth
- More consistent results
- Better for hard materials

**Single pass vs multiple:**
- 1 pass at 80% power
- OR 2 passes at 50% power  Usually better

### Invert

**Invert Image:**
-  **Enabled:** White = deep, Black = shallow
- L **Disabled:** Black = deep, White = shallow (default)

Use invert for:
- Lithophanes (light areas should be thin)
- Embossing (raised areas)
- Inverted height maps

## Material Considerations

### Best Materials

**Excellent for depth:**
- **Wood** - Natural, forgiving, good depth range
- **Acrylic** - Clean results, good for lithophanes
- **Leather** - Creates beautiful embossed effects
- **Coated materials** - Removes coating in varying depths

**Challenging:**
- **Metals** - Require marking compounds
- **Glass** - Difficult without special coatings
- **Very hard plastics** - May require high power

### Material-Specific Tips

**Wood:**
- Start with 10-50% power range
- Softwoods engrave deeper than hardwoods
- Watch for grain direction affecting depth
- Multiple light passes better than single heavy pass

**Acrylic (for lithophanes):**
- Use 3-6mm thick clear acrylic
- Engrave from back, light shines through front
- Thinner = more light, thicker = darker
- Power controls remaining thickness
- Invert image (white = thin, black = thick)

**Leather:**
- Very forgiving material
- 15-40% power range typical
- Creates beautiful embossed look
- Test on scrap (each leather type differs)

## Tips & Best Practices

### Image Preparation

 **For best results:**
- Use high-resolution images (300+ DPI)
- Increase contrast before engraving
- Blur sharp edges for smooth transitions
- Remove noise (creates unwanted texture)
- Test on small area first

 **Adjusting images:**
- **Too flat**  Increase contrast
- **Too much depth**  Decrease contrast
- **Rough surface**  Apply slight blur
- **Missing detail**  Sharpen slightly

### Testing Strategy

1. **Small test square** - 20mm  20mm area
2. **Power range test** - Try 10-40%, then adjust
3. **Speed test** - Try 2000-3000 mm/min
4. **Line interval test** - 0.05mm vs 0.1mm
5. **Scale up** - Use successful settings for full image

### Quality Settings

 **For highest quality:**
- Line interval: 0.05-0.08mm
- Multiple passes (2-3) at lower power
- Slower speed (1500-2000 mm/min)
- Disable bidirectional scanning
- Material must be perfectly flat

 **For faster results:**
- Line interval: 0.1-0.15mm
- Single pass at higher power
- Faster speed (3000-4000 mm/min)
- Enable bidirectional scanning

### Common Issues

**Uneven depth:**
- Material not flat  secure better
- Focus incorrect  adjust focus
- Power unstable  check laser

**Too shallow:**
- Increase max power
- Decrease speed
- Add more passes
- Use softer material

**Burning or charring:**
- Decrease max power
- Increase speed
- Use more passes at lower power
- Improve ventilation

**Ribbed/lined surface:**
- Decrease line interval
- Check that material is flat
- Try different scan angle

## Advanced Techniques

### Lithophanes

Lithophanes are 3D images viewed by backlighting:

**Process:**
1. Use clear acrylic (3-6mm thick)
2. Engrave from **back** side
3. **Invert** the image (white = thin)
4. Power controls remaining thickness
5. View from front with backlight

**Settings for lithophanes:**
- Power range: 20-70%
- Line interval: 0.05mm
- Multiple passes: 2-3
- Very important: material must be flat

### Terrain Maps

Create 3D topographic maps:

**Process:**
1. Get height map (white = high, black = low)
2. Increase contrast for dramatic relief
3. Use 10-60% power range
4. Fine line interval (0.05-0.08mm)

### Combining with Contour

Add depth engraving inside a contour cut:

**Workflow:**
1. Depth engrave the image
2. Contour cut around the perimeter
3. Result: 3D relief plaque that's cut free

See [Multi-Layer Workflow](../multi-layer.md) for details.

## Troubleshooting

### Image looks flat, no depth

- **Increase:** Max power setting
- **Increase:** Contrast in source image
- **Try:** More passes
- **Check:** Material is suitable for depth

### Surface is rough or textured

- **Decrease:** Line interval (try 0.05mm)
- **Apply:** Blur to source image
- **Reduce:** Power (over-burning creates rough texture)
- **Try:** Different scan angle

### Inconsistent depth across image

- **Check:** Material is flat and secured
- **Check:** Focus is consistent across area
- **Verify:** Laser power is stable
- **Try:** Smaller area (check if issue persists)

### Burning at dark areas

- **Decrease:** Max power
- **Increase:** Speed
- **Use:** Multiple passes instead of one
- **Improve:** Ventilation/air assist

### Image is backwards

- **Flip:** Source image horizontally
- **Or:** Engrave from opposite side
- **Check:** If creating lithophane, engrave from back

## Technical Details

### Power Modulation

Depth engraving uses PWM (Pulse Width Modulation):
- Laser power varies continuously along scan lines
- Each pixel gets proportional power
- Smooth gradients require many power levels

### Grayscale to Power Mapping

**Linear mapping:**
```
power = min_power + (pixel_value / 255)  (max_power - min_power)
```

**For inverted:**
```
power = min_power + ((255 - pixel_value) / 255)  (max_power - min_power)
```

### Resolution

**Image resolution affects quality:**
- 254 DPI = 10 pixels/mm  use d0.1mm line interval
- 508 DPI = 20 pixels/mm  use d0.05mm line interval

Higher resolution images can produce finer detail.

## Related Topics

- **[Raster Engraving](raster.md)** - Flat engraving with scanning lines
- **[Contour Cutting](contour.md)** - Cutting around depth-engraved pieces
- **[Multi-Layer Workflow](../multi-layer.md)** - Combining depth with other operations
- **[Material Test Grid](material-test-grid.md)** - Finding optimal power/speed settings
- **[Overscan & Kerf](../overscan-kerf.md)** - Improving engraving quality
