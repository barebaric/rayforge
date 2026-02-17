# Supported File Formats

This page provides detailed information about all file formats supported by Rayforge, including capabilities, limitations, and recommendations.

## Format Overview

### Quick Reference

| Format               | Type    | Import   | Export          | Recommended Use           |
| -------------------- | ------- | -------- | --------------- | ------------------------- |
| **SVG**              | Vector  | ✓ Direct | ✓ Object export | Primary design format     |
| **DXF**              | Vector  | ✓ Direct | ✓ Object export | CAD interchange           |
| **PDF**              | Mixed   | ✓ Trace  | –               | Document export (limited) |
| **PNG**              | Raster  | ✓ Trace  | –               | Photos, images            |
| **JPEG**             | Raster  | ✓ Trace  | –               | Photos                    |
| **BMP**              | Raster  | ✓ Trace  | –               | Simple graphics           |
| **RFS**              | Sketch  | ✓ Direct | ✓ Object export | Parametric sketches       |
| **G-code**           | Control | –        | ✓ Primary       | Machine output            |
| **Rayforge Project** | Project | ✓        | ✓               | Save/load projects        |

---

## Vector Formats

### SVG (Scalable Vector Graphics)

**Extension:** `.svg`
**MIME Type:** `image/svg+xml`
**Import:** Direct vector parsing or bitmap trace
**Export:** Object export (geometry only)

**What is SVG?**

SVG is an XML-based vector image format. It's the **preferred format** for importing designs into Rayforge.

**Supported Features:**

- ✓ Paths (lines, curves, arcs)
- ✓ Basic shapes (rectangles, circles, ellipses, polygons)
- ✓ Groups and transformations
- ✓ Stroke and fill colors
- ✓ Multiple layers
- ✓ Coordinate transformations (translate, rotate, scale)

**Unsupported/Limited Features:**

- ✗ Text (must be converted to paths first)
- ✗ Gradients (simplified or ignored)
- ✗ Filters and effects (ignored)
- ✗ Masks and clipping paths (may not work correctly)
- ✗ Embedded raster images (imported separately if possible)
- ✗ Complex stroke styles (dashes may be simplified)
- ✗ Symbols and use elements (instances may not update)

**Export Notes:**

When exporting a workpiece to SVG, Rayforge exports the geometry as vector paths with:

- Stroke-only rendering (no fill)
- Millimeter units
- Black stroke color

**Best Practices:**

1. **Use Plain SVG format** (not Inkscape SVG or other tool-specific variants)
2. **Convert text to paths** before exporting
3. **Simplify complex paths** to reduce node count
4. **Flatten groups** when possible
5. **Remove unused elements** (guides, grids, hidden layers)
6. **Set document units** to mm (Rayforge's native unit)

**Software Recommendations:**

- **Inkscape** (free) - Excellent SVG support, native format
- **Adobe Illustrator** - Professional tool, "Save As SVG" with simplified options
- **Affinity Designer** - Good SVG export capabilities
- **Figma** - Web-based, export as SVG

---

### DXF (Drawing Exchange Format)

**Extension:** `.dxf`
**MIME Type:** `application/dxf`, `image/vnd.dxf`
**Import:** Direct vector parsing
**Export:** Object export (geometry only)

**What is DXF?**

DXF is an AutoCAD drawing format, widely used for CAD interchange.

**Supported Versions:**

- ✓ **R12/LT2** (recommended - best compatibility)
- ✓ R13, R14
- ✓ R2000 and later (usually works, but R12 is safer)

**Supported Entities:**

- ✓ Lines (LINE)
- ✓ Polylines (LWPOLYLINE, POLYLINE)
- ✓ Arcs (ARC)
- ✓ Circles (CIRCLE)
- ✓ Splines (SPLINE) - converted to polylines
- ✓ Ellipses (ELLIPSE)
- ✓ Layers

**Unsupported/Limited Features:**

- ✗ 3D entities (use 2D projection)
- ✗ Dimensions and annotations (ignored)
- ✗ Blocks/inserts (may not instance correctly)
- ✗ Complex line types (simplified to solid)
- ✗ Text (ignored, convert to outlines first)
- ✗ Hatches (may be simplified or ignored)

**Export Notes:**

When exporting a workpiece to DXF, Rayforge exports:

- Lines as LWPOLYLINE entities
- Arcs as ARC entities
- Bezier curves as SPLINE entities
- Millimeter units (INSUNITS = 4)

**Common Issues:**

**1. Wrong scale**

- **Cause:** DXF files don't always specify units clearly
- **Solution:** Verify units before export, scale manually if needed

**2. Missing entities**

- **Cause:** Unsupported entity types or layers turned off
- **Solution:** Check layer visibility, convert complex entities to polylines

**3. Segmented curves**

- **Cause:** Splines and ellipses converted to line segments
- **Solution:** Increase segment count in export settings

**Best Practices:**

1. **Export as R12/LT2 DXF** for maximum compatibility
2. **Use 2D geometry only** (no 3D)
3. **Simplify before export:**
   - Convert splines to polylines
   - Explode blocks if needed
   - Remove dimensions and text
4. **Check units** (mm recommended)
5. **Test in viewer** before importing to Rayforge

**Software Recommendations:**

- **LibreCAD** (free) - Good DXF support
- **QCAD** (free community edition) - DXF native format
- **AutoCAD** - Industry standard
- **FreeCAD** - Free parametric CAD with DXF export
- **Fusion 360** - Free for hobbyists, excellent DXF export

---

### RFS (Rayforge Sketch)

**Extension:** `.rfs`
**MIME Type:** `application/x-rayforge-sketch`
**Import:** Direct (sketch-based workpieces)
**Export:** Object export (sketch-based workpieces)

**What is RFS?**

RFS is Rayforge's native parametric sketch format. It preserves all geometric
elements and parametric constraints, allowing you to save and share fully
editable sketches.

**Features:**

- ✓ All geometric elements (lines, arcs, circles, rectangles, etc.)
- ✓ All parametric constraints
- ✓ Dimensional values and expressions
- ✓ Fill areas

**When to Use:**

- Save reusable parametric designs
- Share editable sketches with other Rayforge users
- Archive work in progress

---

### PDF (Portable Document Format)

**Extension:** `.pdf`
**MIME Type:** `application/pdf`
**Import:** Rendered to bitmap, then traced
**Export:** Not supported

**What is PDF Import?**

Rayforge can import PDF files by rasterizing them first, then tracing to vectors.

**Process:**

1. PDF rendered to raster image (default 300 DPI)
2. Raster traced to create vector paths
3. Paths added to document

**Limitations:**

- **Not true vector import** - Even vector PDFs are rasterized
- **Quality loss** from rasterization
- **First page only** - Multi-page PDFs only import page 1
- **Slow for complex PDFs** - Rendering and tracing takes time

**When to Use:**

- Last resort when SVG/DXF not available
- Quick import of simple designs
- Documents with mixed content

**Better Alternatives:**

- **Export SVG from source** instead of PDF
- **Use vector formats** (SVG, DXF) when possible
- **For text:** Export with text converted to outlines

---

## Raster Formats

All raster formats are **imported by tracing** - converted to vector paths automatically.

### PNG (Portable Network Graphics)

**Extension:** `.png`
**MIME Type:** `image/png`
**Import:** Trace to vectors
**Export:** Not supported

**Characteristics:**

- **Lossless compression** - No quality loss
- **Transparency support** - Alpha channel preserved
- **Good for:** Logos, line art, screenshots, anything needing transparency

**Tracing Quality:**  (Excellent for high-contrast images)

**Best Practices:**

- Use PNG for logos and graphics with sharp edges
- Ensure high contrast between foreground and background
- Transparent background works better than white

---

### JPEG (Joint Photographic Experts Group)

**Extension:** `.jpg`, `.jpeg`
**MIME Type:** `image/jpeg`
**Import:** Trace to vectors
**Export:** Not supported

**Characteristics:**

- **Lossy compression** - Some quality loss
- **No transparency** - Always has background
- **Good for:** Photos, continuous-tone images

**Tracing Quality:**  (Good for photos, but complex)

**Best Practices:**

- Use high-quality JPEG (low compression)
- Increase contrast before importing
- Consider pre-processing in image editor
- Better to convert to PNG first if possible

---

### BMP (Bitmap)

**Extension:** `.bmp`
**MIME Type:** `image/bmp`
**Import:** Trace to vectors
**Export:** Not supported

**Characteristics:**

- **Uncompressed** - Large file sizes
- **Simple format** - Widely compatible
- **Good for:** Simple graphics, old software output

**Tracing Quality:**  (Good, but no better than PNG)

**Best Practices:**

- Convert to PNG for smaller file size (no quality difference)
- Only use if source software can't export PNG/SVG

---

## Output Formats

### G-code

**Extension:** `.gcode`, `.nc`, `.ngc`
**MIME Type:** `text/plain`
**Import:** Not supported
**Export:**  Primary output format

**What is G-code?**

G-code is the **machine control language** used by laser controllers.

**Rayforge G-code Features:**

- **Dialect support:** GRBL (primary)
- **Optimized toolpaths:** Efficient move ordering
- **Comments:** Human-readable annotations
- **Precision control:** Configurable decimal places
- **Macro insertion:** Custom G-code via hooks

**Export Settings:**

| Setting          | Description                    | Typical Value    |
| ---------------- | ------------------------------ | ---------------- |
| **Precision**    | Decimal places for coordinates | 3 (e.g., 12.345) |
| **Dialect**      | G-code flavor                  | GRBL             |
| **Line numbers** | Add N line numbers             | Usually off      |
| **Whitespace**   | Add spaces for readability     | Usually on       |

**File Size:**

- Simple cuts: 1-50 KB
- Complex engraving: 1-50 MB (large jobs can be >100 MB)

**Compatibility:**

-  GRBL 1.1+ (primary target)
-  grblHAL
-  GRBL 0.9 (limited testing)
-  Marlin, Smoothieware (not officially supported)

See [G-code Dialects](../reference/gcode-dialects.md) for details.

---

## Related Pages

- [Importing Files](importing.md) - How to import each format
- [Exporting G-code](exporting.md) - G-code export options
