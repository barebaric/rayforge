# Importing Files

Rayforge supports importing various file formats, both vector and raster. This page explains
how to import files and optimize them for best results.

## Supported File Formats

### Vector Formats

| Format    | Extension | Import Method           | Best For                        |
| --------- | --------- | ----------------------- | ------------------------------- |
| **SVG**   | `.svg`    | Direct vectors or trace | Vector graphics, logos, designs |
| **DXF**   | `.dxf`    | Direct vectors          | CAD drawings, technical designs |
| **PDF**   | `.pdf`    | Render and trace        | Documents with vector content   |
| **Ruida** | `.rd`     | Direct vectors          | Ruida controller job files      |

### Raster Formats

| Format   | Extension       | Import Method    | Best For                         |
| -------- | --------------- | ---------------- | -------------------------------- |
| **PNG**  | `.png`          | Trace to vectors | Photos, images with transparency |
| **JPEG** | `.jpg`, `.jpeg` | Trace to vectors | Photos, continuous-tone images   |
| **BMP**  | `.bmp`          | Trace to vectors | Simple graphics, screenshots     |

:::note Raster Import
:::

All raster images are **traced** to create vector paths that can be used for laser operations. The quality depends on the tracing configuration.

---

## Importing Files

### The Import Dialog

Rayforge features a unified import dialog that provides live preview and
configuration options for all supported file types. The dialog allows you to:

- **Preview your import** before adding it to the document
- **Configure tracing settings** for raster images
- **Choose import method** for SVG files (direct vectors or trace)
- **Adjust parameters** like threshold, invert, and auto-threshold

![Import Dialog](/images/import-dialog.png)

### Method 1: File Menu

1. **File Import** (or Ctrl+I)
2. **Select your file** from the file picker
3. **Configure import settings** in the import dialog
4. **Preview** the result before importing
5. **Click Import** to add to canvas and document tree

### Method 2: Drag and Drop

1. **Drag file** from your file manager
2. **Drop onto** the Rayforge canvas
3. **Configure import settings** in the import dialog
4. **Preview** the result before importing
5. **Click Import** to add to canvas and document tree

### Method 3: Command Line

```bash
# Open Rayforge with a file
rayforge myfile.svg

# Multiple files
rayforge file1.svg file2.dxf
```

### Auto-Resize on Import

When importing files that are larger than your machine's work area, Rayforge will
automatically:

1. **Scale down** the imported content to fit within the machine boundaries
2. **Preserve aspect ratio** during scaling
3. **Center** the scaled content in the workspace
4. **Show a notification** with the option to undo the resize

The resize notification appears as a toast message:

- ⚠️ "Imported item was larger than the work area and has been scaled down to fit."
- Includes a **"Reset"** button to undo the auto-resize
- The toast remains visible until dismissed or the reset action is taken

This ensures your designs always fit within your machine's capabilities while giving you
the flexibility to restore the original size if needed.

---

## SVG Import

SVG (Scalable Vector Graphics) is the **recommended format** for vector designs.

### Import Options in the Dialog

When importing SVG, the import dialog provides a toggle switch to choose
between two methods:

#### 1. Use Original Vectors (Recommended)

This option is enabled by default in the import dialog.

**How it works:**

- Parses SVG and converts paths directly to Rayforge geometry
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

Disable "Use Original Vectors" to use this method.

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

### Live Preview

The import dialog shows a live preview of how your SVG will be imported:

- Vector paths are displayed in blue overlay
- For trace mode, the original image is shown with the traced paths
- Preview updates in real-time as you change settings

### SVG Best Practices

**Prepare your SVG for best results:**

1. **Convert text to paths:**

   - Inkscape: `Path → Object to Path`
   - Illustrator: `Type → Create Outlines`

2. **Simplify complex paths:**

   - Inkscape: `Path → Simplify` (Ctrl+L)
   - Remove unnecessary nodes

3. **Ungroup nested groups:**

   - Flatten hierarchy where possible
   - `Object → Ungroup` (Ctrl+Shift+G)

4. **Remove hidden elements:**

   - Delete guides, grids, construction lines
   - Remove invisible/transparent objects

5. **Save as Plain SVG:**

   - Inkscape: "Plain SVG" or "Optimized SVG"
   - Not "Inkscape SVG" (has extra metadata)

6. **Check document units:**
   - Set to mm or inches as appropriate
   - Rayforge uses mm internally

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

Rayforge supports standard DXF formats:

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
   - Rayforge assumes mm by default

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

When importing PDF files through the import dialog, Rayforge **renders the PDF**
to an image, then **traces** it to create vectors.

**Process:**

1. PDF rendered and displayed in the import dialog preview
2. You can adjust tracing settings in real-time
3. Rendered image traced using vectorization with your settings
4. Resulting paths added to document when you click Import

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

4. **Use the import dialog preview:**
   - Adjust threshold and invert settings for best results
   - Preview shows exactly how the PDF will be traced

---

## Ruida Import

Ruida (.rd) files are proprietary binary job files used by Ruida controllers in many
laser cutting machines. These files contain both vector geometry and laser settings
organized in layers (colors).

**After import:**

- **Check scale** - Verify dimensions match expected size
- **Review layers** - Ensure all layers imported correctly
- **Validate paths** - Confirm all cutting paths are present

### Limitations

- **Read-only import** - Ruida files can only be imported, not exported
- **Binary format** - Direct editing of original .rd files not supported
- **Proprietary features** - Some advanced Ruida features may not be fully supported

---

## Raster Image Import (PNG, JPG, BMP)

Raster images are **traced** to create vector paths using the import dialog.

### Tracing Process in the Dialog

**How it works:**

1. **Image loaded** into the import dialog
2. **Live preview** shows the traced result
3. **Tracing settings** can be adjusted in real-time
4. **Vector paths created** from the traced edges
5. **Paths added** to the document as workpieces when imported

### Tracing Configuration in the Dialog

The import dialog provides these adjustable parameters:

| Parameter          | Description         | Effect                                              |
| ------------------ | ------------------- | --------------------------------------------------- |
| **Auto Threshold** | Automatic detection | When enabled, automatically finds optimal threshold |
| **Threshold**      | Black/white cutoff  | Lower = more detail, higher = simpler               |
| **Invert**         | Reverse colors      | Trace light objects on dark background              |

**Default settings** work well for most images. The dialog shows a live preview
that updates as you adjust these settings, allowing you to fine-tune the trace
before importing.

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

## Related Pages

- [Supported Formats](formats) - Detailed format specifications
- [Exporting G-code](exporting) - Output options
- [Quick Start](../getting-started/quick-start) - First import tutorial
