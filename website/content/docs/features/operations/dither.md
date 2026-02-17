# Dithered Raster Engraving

Dithered raster engraving converts grayscale images to binary patterns using
dithering algorithms, enabling high-quality photo engraving with better tonal
reproduction than simple threshold-based methods.

## Overview

Dithered raster operations:

- Convert grayscale images to binary using dithering algorithms
- Create the illusion of continuous tones using dot patterns
- Provide multiple algorithm choices for different effects
- Work best with photographs and complex grayscale images

## When to Use Dithered Raster

Use dithered raster engraving for:

- Engraving photographs on wood or leather
- Creating halftone-style artwork
- Images with smooth gradients
- When standard raster doesn't capture enough detail

**Don't use dithered raster for:**

- Simple text and logos (use standard [Raster](raster.md) instead)
- Vector artwork with solid colors
- When you need precise power control per tone (use [Depth](depth.md) instead)

## Creating a Dithered Raster Operation

### Step 1: Prepare Your Image

For best results:

1. **Use high-quality source images** - Higher resolution gives better detail
2. **Convert to grayscale** - Color images are converted automatically
3. **Adjust contrast** - Increase contrast for more dramatic results
4. **Consider material** - Wood grain adds texture to the result

### Step 2: Add Dithered Raster Operation

- **Menu:** Operations → Add Raster (Dither)
- **Right-click:** Context menu → Add Operation → Raster (Dither)

### Step 3: Configure Settings

## Key Settings

### Dithering Algorithm

Choose the algorithm that best suits your image and material:

| Algorithm       | Quality | Speed   | Best For                            |
| --------------- | ------- | ------- | ----------------------------------- |
| Floyd-Steinberg | Highest | Slowest | Photos, portraits, smooth gradients |
| Bayer 2x2       | Low     | Fastest | Coarse halftone effect              |
| Bayer 4x4       | Medium  | Fast    | Balanced halftone                   |
| Bayer 8x8       | High    | Medium  | Fine detail, subtle patterns        |

**Floyd-Steinberg** is the default and recommended for most photo engravings.
It uses error diffusion to distribute quantization errors to neighboring pixels,
creating natural-looking results.

**Bayer dithering** creates regular patterns that can produce artistic effects
resembling traditional halftone printing.

### Invert

- **Off** (default): Dark areas are engraved, light areas are not
- **On**: Light areas are engraved, dark areas are not

Use invert when:

- Your material darkens when engraved (like wood)
- You want the original image appearance to match the engraved result

### Bidirectional Scanning

- **Enabled:** Laser engraves in both directions (faster)
- **Disabled:** Laser only engraves left-to-right (slower, more consistent)

For best quality with photos, disable bidirectional scanning.

### Power & Speed

Same settings as standard raster engraving:

**Power (%):**

- Laser intensity for the engraving
- Typical range: 20-60% for engraving

**Speed (mm/min):**

- How fast the laser scans
- Typical range: 2000-5000 mm/min

**Starting values (wood engraving):**

- Power: 30-50%
- Speed: 2500-4000 mm/min

### Line Interval

**Line Interval (mm):**

- Spacing between scan lines
- Smaller = higher quality, longer job time
- Recommended: 0.1mm for most photos

| Interval | Quality | Speed   | Use For                 |
| -------- | ------- | ------- | ----------------------- |
| 0.05mm   | Highest | Slowest | Very fine detail        |
| 0.1mm    | High    | Medium  | General photo engraving |
| 0.15mm   | Medium  | Fast    | Larger images           |
| 0.2mm+   | Low     | Fastest | Test runs, draft        |

## Dithering vs Standard Raster

| Feature                | Dithered Raster      | Standard Raster          |
| ---------------------- | -------------------- | ------------------------ |
| Grayscale support      | Pattern-based        | Power modulation         |
| Detail preservation    | Excellent            | Good                     |
| Tonal range            | Perceived continuous | Limited steps            |
| Job time               | Similar              | Similar                  |
| Material compatibility | All materials        | Best with variable power |

**Choose Dithered Raster when:**

- Your machine has limited power modulation
- You want halftone-style artistic effects
- The image has subtle gradients

**Choose Standard Raster when:**

- Your machine supports smooth power modulation
- You want true grayscale engraving
- Speed is critical

## Tips & Best Practices

### Image Preparation

1. **Resolution:** Use images at least 300 DPI at target size
2. **Contrast:** Increase contrast by 10-20% before engraving
3. **Crop:** Remove unnecessary background before importing
4. **Test:** Always test on scrap material first

### Material Selection

**Best materials for dithered engraving:**

- Wood (birch, maple, cherry)
- Leather
- Anodized aluminum
- Dark slate

**Challenging materials:**

- Clear acrylic (engraving may not show well)
- Very light woods (low contrast)

### Quality Settings

**For best quality:**

- Use Floyd-Steinberg algorithm
- Use smaller line interval (0.05-0.1mm)
- Disable bidirectional scanning
- Lower power, verify focus

**For faster engraving:**

- Use Bayer 4x4 algorithm
- Larger line interval (0.15mm)
- Enable bidirectional scanning
- Single pass at higher power

## Troubleshooting

### Image looks muddy or washed out

- **Increase contrast** in your image editor
- **Check power setting** - may be too low
- **Try different algorithm** - Floyd-Steinberg usually works best

### Visible banding or streaks

- **Disable bidirectional** scanning
- **Check belt tension** on your machine
- **Reduce speed** for more consistent results

### Too dark / over-engraved

- **Reduce power** setting
- **Increase speed** setting
- **Check focus** - too close can cause darker burns

### Fine detail lost

- **Decrease line interval** (0.05mm)
- **Use higher resolution** source image
- **Use Floyd-Steinberg** algorithm

## Example Settings

### Portrait on Birch Plywood

```
Algorithm: Floyd-Steinberg
Power: 35%
Speed: 3000 mm/min
Line Interval: 0.1mm
Bidirectional: Off
Invert: On (for natural appearance)
```

### Landscape on Leather

```
Algorithm: Floyd-Steinberg
Power: 25%
Speed: 2500 mm/min
Line Interval: 0.1mm
Bidirectional: Off
Invert: Off
```

### Artistic Halftone Effect

```
Algorithm: Bayer 4x4
Power: 40%
Speed: 3500 mm/min
Line Interval: 0.15mm
Bidirectional: On
Invert: Off
```

## Related Topics

- **[Raster Engraving](raster.md)** - Standard raster with power modulation
- **[Depth Engraving](depth.md)** - 3D relief effects
- **[Overscan](../overscan.md)** - Improving engraving quality
- **[Material Test Grid](material-test-grid.md)** - Finding optimal settings
