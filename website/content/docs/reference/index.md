# Reference

Technical reference documentation for Rayforge.

## Quick Reference

- **[Keyboard Shortcuts](shortcuts.md)**: Complete list of keyboard shortcuts
- **[G-code Dialects](gcode-dialects.md)**: Supported G-code flavors and their differences
- **[Firmware Compatibility](firmware.md)**: Compatible firmware versions and features

## Keyboard Shortcuts

Master Rayforge with keyboard shortcuts for faster workflow. See the complete [Keyboard Shortcuts](shortcuts.md) reference.

**Most Common:**

- ++ctrl+o++: Open file
- ++ctrl+s++: Save project
- ++ctrl+g++: Generate G-code
- ++ctrl+3++: 3D preview
- ++ctrl+r++: Start job
- ++ctrl+p++: Pause job
- ++esc++: Stop job

## G-code Dialects

Rayforge supports multiple G-code dialects:

- **GRBL**: Standard GRBL 1.1+
- **Smoothieware**: Smoothieware v1 and v2
- **GRBL-compatible**: Generic GRBL-compatible controllers

See [G-code Dialects](gcode-dialects.md) for command differences and compatibility notes.

## Firmware Compatibility

### GRBL

- **Minimum Version**: GRBL 1.1
- **Recommended**: GRBL 1.1h or later
- **Tested Versions**: 1.1f, 1.1h, 1.1i

### Smoothieware

- **Supported**: v1 and v2-dev
- **Connection**: Telnet over network
- **Tested Versions**: v1.1, v2-dev

See [Firmware Compatibility](firmware.md) for detailed version information and features.

## Technical Specifications

### Supported File Import Formats

| Format | Extension | Type | Notes |
|:-------|:----------|:-----|:------|
| SVG | .svg | Vector | Full support |
| DXF | .dxf | Vector | AutoCAD R12-R2018 |
| PDF | .pdf | Vector | Vector content only |
| JPEG | .jpg, .jpeg | Raster | Lossy compression |
| PNG | .png | Raster | Lossless, transparency |
| BMP | .bmp | Raster | Uncompressed |
| TIFF | .tif, .tiff | Raster | High quality |
| Ruida | .rd | Proprietary | Import from Ruida controllers |

### G-code Output

- **Format**: Plain text G-code
- **Encoding**: UTF-8 or ASCII
- **Line endings**: LF (Linux/Mac) or CRLF (Windows)
- **Commands**: Standard GRBL/Smoothieware command set

### Coordinate System

- **Units**: Millimeters (mm) or inches
- **Origin**: Configurable (bottom-left, top-left, etc.)
- **Coordinate Mode**: Absolute (G90) or relative (G91)

### Performance Limits

| Metric | Recommended | Maximum |
|:-------|:------------|:--------|
| Canvas Objects | < 1000 | 10000 |
| Path Points | < 100K | 1M |
| Raster Resolution | 300 DPI | 600 DPI |
| G-code Lines | < 1M | 10M |

## System Requirements

### Minimum Requirements

- **OS**: Ubuntu 24.04+, Windows 10+
- **CPU**: Dual-core 2.0 GHz
- **RAM**: 4 GB
- **GPU**: OpenGL 3.3 compatible
- **Disk**: 500 MB free space

### Recommended Requirements

- **OS**: Ubuntu 24.04 LTS, Windows 11
- **CPU**: Quad-core 3.0 GHz or better
- **RAM**: 8 GB or more
- **GPU**: Dedicated graphics with OpenGL 4.5+
- **Disk**: 1 GB free space (SSD recommended)

---

**Explore:** [Keyboard Shortcuts →](shortcuts.md) | [G-code Dialects →](gcode-dialects.md) | [Firmware →](firmware.md)
