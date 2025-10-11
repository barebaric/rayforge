# File Formats

Rayforge supports a wide variety of file formats for importing designs and exporting G-code.

## Overview

- **[Importing Files](importing.md)**: How to import designs into Rayforge
- **[Supported Formats](formats.md)**: Complete list of supported file types
- **[Exporting G-code](exporting.md)**: Generate and save G-code files

## Supported Import Formats

### Vector Formats

Vector formats are ideal for cutting and engraving with sharp, precise paths:

| Format | Extension | Best For | Notes |
|:-------|:----------|:---------|:------|
| **SVG** | `.svg` | General use, web graphics | Most common format, fully supported |
| **DXF** | `.dxf` | CAD drawings, technical designs | AutoCAD and CAD software output |
| **PDF** | `.pdf` | Documents, print-ready designs | Vector content extracted |

### Raster Formats

Raster (bitmap) images are best for engraving and photo reproduction:

| Format | Extension | Best For | Notes |
|:-------|:----------|:---------|:------|
| **JPEG** | `.jpg`, `.jpeg` | Photos, complex images | Lossy compression |
| **PNG** | `.png` | Graphics with transparency | Lossless compression |
| **BMP** | `.bmp` | Simple graphics | Large file sizes |
| **TIFF** | `.tif`, `.tiff` | High-quality images | Support for multiple layers |

### Specialized Formats

| Format | Extension | Best For | Notes |
|:-------|:----------|:---------|:------|
| **Ruida** | `.rd` | Ruida laser controller files | Import existing Ruida jobs |

## Working with Different File Types

### Vector Files (SVG, DXF, PDF)

**Best Practices:**

- ✓ Use for contour cutting
- ✓ Scalable without quality loss
- ✓ Edit paths before import in design software
- ✓ Convert text to paths before import

**Common Issues:**

- Text not converted to paths: Convert in design software first
- Embedded raster images: Extract or trace to vectors
- Complex gradients: May not import correctly

### Raster Images (JPEG, PNG, BMP)

**Best Practices:**

- ✓ Use for engraving photos and complex images
- ✓ Higher DPI = better detail (300+ DPI recommended)
- ✓ Convert to grayscale for single-color engraving
- ✓ High contrast improves results

**Common Issues:**

- Low resolution: Results in pixelated engraving
- Wrong DPI: Specify DPI in import settings
- Color images: Converted to grayscale automatically

### Ruida Files (.rd)

Import jobs created for Ruida controllers:

- Operations converted to Rayforge operations
- Layer assignments preserved
- Power and speed settings imported
- May require adjustment for your specific machine

## File Import Workflow

1. **Prepare file**: Optimize in design software
2. **Import**: File → Open or drag-and-drop
3. **Review**: Check that all elements imported correctly
4. **Scale**: Verify dimensions are correct
5. **Assign operations**: Configure cutting/engraving operations
6. **Generate G-code**: Machine → Generate G-code

## Export Formats

### G-code

Rayforge generates standard G-code for GRBL and Smoothieware:

- **GRBL**: Standard GRBL commands (G0, G1, M3, M5, etc.)
- **Smoothieware**: Smoothieware-specific extensions
- **Custom**: Configure custom G-code dialect

See [Exporting G-code](exporting.md) for details.

### Project Files

Save your work as Rayforge project files:

- **Extension**: `.rayforge` (coming soon)
- **Contains**: All designs, operations, and settings
- **Portable**: Share with other Rayforge users

---

**Next**: [Importing Files →](importing.md) | [Supported Formats →](formats.md)
