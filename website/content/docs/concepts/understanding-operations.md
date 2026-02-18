# Understanding Operations

Operations are the core of how Rayforge converts your designs into laser instructions. This page explains what operations are, the different types available, and how to choose the right one for your project.

## What is an Operation?

An **operation** is a processing step that converts geometry (shapes, images, paths) into **toolpaths** - the actual instructions that tell the laser where to move and when to fire.

**Think of it like this:**

- Your **design** is what you want to create (a logo, a box, a photo)
- An **operation** is how you want to create it (engrave, cut, etc.)
- The **toolpath** is the specific commands that make it happen

**Example:**

```
Design: A circle
Operation: Contour Cut
Result: Laser traces the circle outline and cuts it out

Design: A circle
Operation: Raster Engrave
Result: Laser fills the circle area with back-and-forth scanning
```

---

## Operation Types

Rayforge provides several operation types, each suited for different tasks.

### Contour (Vector Cutting)

**What it does:** Follows the outline of vector paths.

**How it works:**
- Traces the edges of shapes
- Laser follows the path continuously
- Can cut through material or just mark the surface

**Best for:**
- Cutting out shapes
- Outlining designs
- Creating perimeters
- Precise vector work

**Example uses:**
- Cutting box parts
- Creating stencils
- Outlining logos
- Cutting acrylic shapes

**Settings:**
- **Power:** High for cutting, low for marking
- **Speed:** Slow for cutting, faster for marking
- **Passes:** Multiple for thick materials

### Engrave (Raster Engraving)

**What it does:** Scans back and forth to engrave images with multiple modes.

**How it works:**
- Divides area into horizontal scan lines
- Laser moves left-to-right, then right-to-left
- Supports variable power, constant power, dither, and multi-pass modes
- Like a printer printing an image

**Best for:**
- Photo engraving
- Grayscale images
- Filled areas
- Textures and patterns
- 3D relief effects (multi-pass mode)

**Example uses:**
- Engraving photos on wood
- Creating gradients
- Filling shapes with texture
- Bitmap graphics
- Lithophanes

**Settings:**
- **DPI:** Resolution (300-500 typical)
- **Speed:** Medium to fast
- **Power:** Varies with image darkness
- **Line spacing:** Controls density
- **Mode:** Variable power, constant power, dither, or multi-pass

### Shrink Wrap

**What it does:** Creates a tight boundary around selected objects.

**How it works:**
- Calculates minimum outline containing all objects
- Useful for grouping or creating boundaries
- Can add offset/margin

**Best for:**
- Creating cut boundaries around complex designs
- Grouping multiple objects
- Adding margins for cutting

**Example uses:**
- Outlining scattered elements
- Creating packing boundaries
- Adding cut lines around designs

### Material Test Grid

**What it does:** Generates a power/speed test matrix.

**How it works:**
- Creates grid of test squares
- Each square has different power/speed combination
- Helps find optimal settings

**Best for:**
- Testing new materials
- Finding cut settings
- Optimizing engraving parameters

**Example uses:**
- Testing wood power/speed combinations
- Finding acrylic cut settings
- Optimizing leather engraving

See [Material Test Grid](../features/operations/material-test-grid.md) for details.

---

## Choosing the Right Operation

### Decision Tree

```
What do you want to do?

Cut out a shape?
  ├─ Vector design → Contour
  └─ Raster design → Trace first, then Contour

Engrave an image/photo?
  ├─ Flat engraving → Engrave (variable/constant power)
  ├─ Photo quality → Engrave (dither mode)
  └─ 3D relief → Engrave (multi-pass mode)

Mark/score surface?
  → Contour (low power)

Fill an area with texture?
  → Raster

Test material settings?
  → Material Test Grid
```

### Operation Comparison

| Goal | Operation | Why? |
|------|-----------|------|
| **Cut through material** | Contour | Follows edges precisely |
| **Engrave a photo** | Engrave | Variable power or dither for shading |
| **Create 3D effect** | Engrave (multi-pass) | Controls engraving depth |
| **Mark outline only** | Contour | Low power, just surface marking |
| **Fill shape with pattern** | Engrave | Scans fill area |
| **Test settings** | Material Test Grid | Systematic testing |

---

## How Operations Process Geometry

Understanding how operations work helps you prepare files correctly.

### Contour Processing

**Input:** Vector paths (lines, curves, shapes)

**Process:**
1. Extracts path outlines
2. Determines trace direction
3. Generates G-code to follow paths
4. Adds lead-in/out if configured

**Requirements:**
- Vector geometry (not raster)
- Closed paths for cutting shapes
- Open paths for lines/curves

### Raster Processing

**Input:** Raster images (photos, bitmaps) or filled vectors

**Process:**
1. Rasterizes geometry at specified DPI
2. Divides into scan lines
3. Converts pixel darkness to laser power (or uses dithering)
4. Generates back-and-forth scanning G-code
5. Applies overscan if enabled

**Requirements:**
- Image data or shapes to fill
- Resolution (DPI) specified
- Power range configured

### Depth/3D Processing (Multi-Pass Mode)

**Input:** Grayscale image (depth map)

**Process:**
1. Reads depth map (black=deep, white=shallow)
2. Calculates multiple passes
3. Each pass engraves at different depth
4. Builds up 3D relief incrementally

**Requirements:**
- Grayscale depth map
- Maximum depth specified
- Material capable of depth engraving

---

## Multiple Operations on Same Geometry

You can apply different operations to the same object:

**Example: Logo**

1. **Raster engrave** to fill the logo shape
2. **Contour cut** to outline and cut it out

**Layer setup:**
```
Layer 1: Logo Engrave (Engrave)
  └─ Logo shape

Layer 2: Logo Cut (Contour)
  └─ Logo outline
```

**Why separate layers?**
- Different operations need different settings
- Control execution order (engrave first, cut last)
- Easier to enable/disable independently

See [Multi-Layer Workflow](../features/multi-layer.md) for details.

---

## Common Operation Mistakes

### Using Contour for Photos

**Problem:** Trying to contour a raster image.

**Why it fails:** Contour needs vector paths, not pixels.

**Solution:** Use Raster operation, or trace the image first.

### Using Raster for Cutting

**Problem:** Using raster to cut out shapes.

**Why it's inefficient:** Raster scans entire area, very slow for cutting.

**Solution:** Use Contour operation for cutting outlines.

### Wrong Power/Speed for Operation

**Problem:** Using cutting settings for engraving (or vice versa).

**Result:** Either doesn't work or damages material.

**Solution:**
- **Contour (cutting):** High power, slow speed
- **Raster (engraving):** Medium power, medium-fast speed
- **Contour (marking):** Low power, fast speed

### Forgetting Multi-Pass

**Problem:** Trying to cut thick material in one pass.

**Result:** Doesn't cut through, or excessive charring.

**Solution:** Use multiple passes at lower power each.

---

## Operation Settings Overview

### Power

**What it controls:** Laser intensity.

**Range:** 0-100% (or 0-1000 in G-code)

**Guidelines:**
- **Cutting:** 70-100%
- **Engraving:** 20-60%
- **Marking:** 10-30%

---

## Operation Execution Order

Operations execute in a specific order:

1. **Layer order** (top to bottom in layer list)
2. **Within a layer:** Defined by workflow
3. **Within operation:** Path ordering (can be optimized)

**Important:** Layers execute sequentially, not in parallel.

**Example execution:**
```
Start Job
  Layer 1: Engrave
    Operation processes all workpieces in layer
  Layer 2: Cut
    Operation processes all workpieces in layer
End Job
```

**Why order matters:**
- Engrave before cutting (so parts don't move)
- Multiple passes in sequence
- Heat management (alternate areas)

---

## Advanced: Operation Transformers

Operations can have **transformers** that modify the toolpath:

**Common transformers:**

- **Overscan:** Extends raster lines for smooth acceleration
- **Tabs:** Adds holding tabs to contour cuts
- **Optimize:** Reorders paths for efficiency
- **Kerf:** Adjusts for material removal (planned feature)

**Where to configure:** Layer workflow settings

See individual feature pages for details.

---

## Tips for Operation Selection

1. **Start with the right operation** - It's easier than trying to fix wrong choice
2. **Test on scrap** - Try operation settings on waste material first
3. **Use Material Test Grid** - Find optimal settings systematically
4. **Check examples** - Look at similar projects for guidance
5. **Understand your goal** - Cut through? Mark surface? Engrave detail?

---

## Related Pages

- [Contour Operation](../features/operations/contour.md) - Detailed contour docs
- [Engrave Operation](../features/operations/engrave.md) - Detailed engrave docs (variable power, dither, multi-pass)
- [Material Test Grid](../features/operations/material-test-grid.md) - Testing guide
- [Multi-Layer Workflow](../features/multi-layer.md) - Organizing operations
- [Power vs Speed](power-vs-speed.md) - Setting selection
