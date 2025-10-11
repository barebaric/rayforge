# Importing Files

RayForge supports importing various file formats, both vector and raster. This page explains how to import files and optimize them for best results.

## Supported File Formats

### Vector Formats

| Format | Extension | Import Method | Best For |
|--------|-----------|---------------|----------|
| **SVG** | `.svg` | Direct vectors or trace | Vector graphics, logos, designs |
| **DXF** | `.dxf` | Direct vectors | CAD drawings, technical designs |
| **PDF** | `.pdf` | Render and trace | Documents with vector content |

### Raster Formats

| Format | Extension | Import Method | Best For |
|--------|-----------|---------------|----------|
| **PNG** | `.png` | Trace to vectors | Photos, images with transparency |
| **JPEG** | `.jpg`, `.jpeg` | Trace to vectors | Photos, continuous-tone images |
| **BMP** | `.bmp` | Trace to vectors | Simple graphics, screenshots |

!!! note "Raster Import"
    All raster images are **traced** to create vector paths that can be used for laser operations. The quality depends on the tracing configuration.

---

## Importing Files

### Method 1: File Menu

1. **File  Import** (or Ctrl+I)
2. **Select your file** from the file picker
3. **For SVG:** Choose import method (see below)
4. **For rasters:** Files are automatically traced
5. **File appears** in the canvas and document tree

### Method 2: Drag and Drop

1. **Drag file** from your file manager
2. **Drop onto** the RayForge canvas
3. **Import proceeds** as above

### Method 3: Command Line

```bash
# Open RayForge with a file
rayforge myfile.svg

# Multiple files
rayforge file1.svg file2.dxf
```

---

## SVG Import

SVG (Scalable Vector Graphics) is the **recommended format** for vector designs.

### Import Options

When importing SVG, RayForge offers two methods:

#### 1. Import Vectors Directly (Recommended)

**How it works:**
- Parses SVG and converts paths directly to RayForge geometry
- High-fidelity preservation of curves and shapes
- Maintains exact vector data

**Pros:**
- Best quality and precision
- Editable paths
- Small file size

**Cons:**
- Some advanced SVG features not supported
- Complex SVGs may have issues

**Use for:**
- Clean vector designs from Inkscape, Illustrator
- Simple to moderate complexity
- Designs without advanced SVG features

#### 2. Trace Bitmap

**How it works:**
- Renders SVG to a raster image first
- Traces the rendered image to create vectors
- More compatible but less precise

**Pros:**
- Handles complex SVG features
- Robust fallback method
- Supports effects and filters

**Cons:**
- Quality loss from rasterization
- Larger file sizes
- Not as precise

**Use for:**
- SVGs that fail direct import
- SVGs with effects, filters, gradients
- When direct import produces errors

### SVG Best Practices

**Prepare your SVG for best results:**

1. **Convert text to paths:**
   - Inkscape: `Path  Object to Path`
   - Illustrator: `Type  Create Outlines`

2. **Simplify complex paths:**
   - Inkscape: `Path  Simplify` (Ctrl+L)
   - Remove unnecessary nodes

3. **Ungroup nested groups:**
   - Flatten hierarchy where possible
   - `Object  Ungroup` (Ctrl+Shift+G)

4. **Remove hidden elements:**
   - Delete guides, grids, construction lines
   - Remove invisible/transparent objects

5. **Save as Plain SVG:**
   - Inkscape: "Plain SVG" or "Optimized SVG"
   - Not "Inkscape SVG" (has extra metadata)

6. **Check document units:**
   - Set to mm or inches as appropriate
   - RayForge uses mm internally

**Common SVG features that may not import:**

- Gradients (convert to solid fills or raster)
- Filters and effects (flatten to paths)
- Masks and clipping paths (expand/flatten)
- Embedded raster images (export separately)
- Text (convert to paths first)

---

## DXF Import

DXF (Drawing Exchange Format) is common for CAD software.

### DXF Versions

RayForge supports standard DXF formats:

- **R12/LT2** (recommended) - Best compatibility
- **R13, R14** - Good support
- **R2000+** - Generally works, but R12 is safer

**Tip:** Export as R12/LT2 DXF for maximum compatibility.

### DXF Import Tips

**Before exporting from CAD:**

1. **Simplify the drawing:**
   - Remove unnecessary layers
   - Delete dimensions and annotations
   - Remove 3D objects (use 2D projection)

2. **Check units:**
   - Verify drawing units (mm vs inches)
   - RayForge assumes mm by default

3. **Flatten layers:**
   - Consider exporting only relevant layers
   - Hide or delete construction layers

4. **Use appropriate precision:**
   - Laser precision is typically 0.1mm
   - Don't over-specify precision

**After import:**

- Check scale (DXF units may need adjustment)
- Verify all paths imported correctly
- Delete any unwanted construction elements

---

## PDF Import

PDF files can contain vector graphics, raster images, or both.

### How PDF Import Works

RayForge **renders the PDF** to an image, then **traces** it to create vectors.

**Process:**
1. PDF rendered at specified DPI (default 300)
2. Rendered image traced using vectorization
3. Resulting paths added to document

**Limitations:**
- Text is rasterized (not editable as paths)
- Vector quality depends on rendering DPI
- Multi-page PDFs: only first page imported

### PDF Import Tips

**Best results:**

1. **Use vector PDFs:**
   - PDFs created from vector software (Illustrator, Inkscape)
   - Not scanned documents or embedded images

2. **Export SVG instead if possible:**
   - Most design software can export SVG directly
   - SVG will have better quality than PDF import

3. **For documents with text:**
   - Export as SVG with fonts converted to paths
   - Or render PDF at high DPI (600+) and trace

---

## Raster Image Import (PNG, JPG, BMP)

Raster images are **automatically traced** to create vector paths.

### Tracing Process

**How it works:**

1. **Image loaded** into RayForge
2. **Tracing algorithm** detects edges and shapes
3. **Vector paths created** from the traced edges
4. **Paths added** to the document as workpieces

### Tracing Configuration

**Adjustable parameters:**

| Parameter | Description | Effect |
|-----------|-------------|--------|
| **Threshold** | Black/white cutoff | Lower = more detail, higher = simpler |
| **Despeckle** | Remove noise | Higher = cleaner, removes small details |
| **Smoothing** | Curve smoothing | Higher = smoother but less accurate |
| **Corner Threshold** | Sharp vs smooth corners | Lower = more sharp corners |

**Default settings** work well for most images.

### Preparing Images for Tracing

**For best results:**

1. **High contrast:**
   - Adjust brightness/contrast in image editor
   - Clear distinction between foreground and background

2. **Clean background:**
   - Remove noise and artifacts
   - Solid white or transparent background

3. **Appropriate resolution:**
   - 300-500 DPI for photos
   - Too high = slow tracing, too low = poor quality

4. **Crop to content:**
   - Remove unnecessary borders
   - Focus on the area to be engraved/cut

5. **Convert to black and white:**
   - For cutting: pure B&W
   - For engraving: grayscale is fine

**Image editing tools:**
- GIMP (free)
- Photoshop
- Krita (free)
- Paint.NET (free, Windows)

### Trace Quality

**Good trace candidates:**
- Logos with clear edges
- High-contrast images
- Line art and drawings
- Text (though better as vector)

**Poor trace candidates:**
- Low-resolution images
- Photos with soft edges
- Images with gradients
- Very detailed or complex photos

---

## Import Troubleshooting

### File Won't Import

**Problem:** RayForge doesn't recognize or can't open the file.

**Solutions:**

1. **Check file format** - Ensure it's a supported type
2. **Try different format** - Convert SVG  DXF or vice versa
3. **Re-export from source** - Original software may have export issues
4. **Check file corruption** - Open in another application first
5. **Simplify the file** - Remove complex features and retry

### Import is Empty

**Problem:** File imports but nothing appears in canvas.

**Diagnosis:**

1. **Extremely large coordinates** - Objects far from origin
2. **Empty file** - No actual content
3. **Unsupported features only** - All elements filtered out

**Solutions:**

- **Zoom out significantly** or `View  Zoom to Fit`
- **Check source file** in original application
- **Simplify and re-export** from design software
- **Check document tree** - objects may be there but not visible

### Imported Shapes are Wrong

**Problem:** Shapes are distorted, wrong size, or incorrect.

**Common causes:**

1. **Unit mismatch** - File in inches, interpreted as mm (or vice versa)
2. **Scale issue** - DPI or export settings wrong
3. **Transform matrix** - Complex transformations not handled
4. **Curved paths** - Arc/bezier conversion issues

**Solutions:**

- **Check source units** - Verify document units before export
- **Scale manually** - Measure and correct after import
- **Simplify paths** - Convert complex curves to simpler forms
- **Export as simpler format** - Try different export settings

### Trace Quality Poor

**Problem:** Raster images trace with jagged or incorrect paths.

**Solutions:**

1. **Increase image resolution** - Use higher quality source
2. **Adjust threshold** - Find right balance for your image
3. **Pre-process image** - Edit for contrast and cleanup
4. **Use smaller despeckle** - Preserve more detail
5. **Trace manually** - Redraw in vector software instead

### Import Takes Forever

**Problem:** Import process is extremely slow.

**Causes:**

- Very complex SVG (thousands of paths)
- High-resolution raster being traced
- Large file size

**Solutions:**

- **Simplify in design software** before importing
- **Reduce image resolution** for rasters
- **Split large files** into multiple smaller files
- **Remove unnecessary elements** (guides, hidden layers)

---

## File Organization Tips

### Naming Conventions

**Good file names:**
- `logo-engrave.svg`
- `box-cuts-3mm-ply.dxf`
- `photo-portrait-150x200.png`

**Include:**
- Project name
- Operation type (cut, engrave)
- Material/thickness if relevant
- Dimensions

### File Preparation Checklist

Before importing:

- [ ] Text converted to paths
- [ ] Complex paths simplified
- [ ] Hidden elements removed
- [ ] Correct units set
- [ ] Appropriate file format selected
- [ ] File tested in original software
- [ ] Exported with compatible settings

### Design Software Recommendations

**Vector design:**
- **Inkscape** (free, excellent SVG support)
- **Adobe Illustrator** (professional, paid)
- **Affinity Designer** (affordable alternative)
- **LibreCAD** (free, for DXF/CAD work)

**Raster editing:**
- **GIMP** (free, powerful)
- **Photoshop** (professional, paid)
- **Krita** (free, good for digital art)

---

## Related Pages

- [Supported Formats](formats.md) - Detailed format specifications
- [Exporting G-code](exporting.md) - Output options
- [Quick Start](../getting-started/quick-start.md) - First import tutorial
