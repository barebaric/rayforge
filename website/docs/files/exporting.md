# Exporting from Rayforge

Rayforge supports several export options for different purposes:

- **G-code** - Machine control output for running jobs
- **Object Export** - Export individual workpieces to vector formats
- **Document Export** - Export all workpieces as a single file

---

## Exporting Objects

You can export any workpiece to vector formats for use in design software, CAD
applications, or for archiving.

### How to Export

1. **Select a workpiece** on the canvas
2. **Choose Object → Export Object...** (or right-click → Export Object...)
3. **Select format** and save location

### Available Formats

| Format  | Extension | Description                                                                                                    |
| ------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| **RFS** | `.rfs`    | Rayforge's native parametric sketch format. Preserves all constraints and can be re-imported for editing.      |
| **SVG** | `.svg`    | Scalable Vector Graphics. Widely compatible with design software like Inkscape, Illustrator, and web browsers. |
| **DXF** | `.dxf`    | Drawing Exchange Format. Compatible with most CAD applications like AutoCAD, FreeCAD, and LibreCAD.            |

### Export Notes

- **SVG and DXF** export the resolved geometry (not parametric constraints)
- Exports use **millimeter units**
- Geometry is scaled to actual dimensions (world space)
- Multiple subpaths (disconnected shapes) are preserved as separate elements

### Use Cases

**Sharing designs:**

- Export to SVG for sharing with Inkscape users
- Export to DXF for CAD software users

**Further editing:**

- Export to SVG/DXF, edit in external software, re-import

**Archiving:**

- Use RFS for sketch-based designs to preserve editability
- Use SVG/DXF for long-term storage or non-Rayforge users

---

## Exporting Documents

You can export all workpieces in a document to a single vector file. This is
useful for sharing complete projects or creating backups in standard formats.

### How to Export

1. **Choose File → Export Document...**
2. **Select format** (SVG or DXF)
3. **Choose save location**

### Available Formats

| Format  | Extension | Description                                                                                                    |
| ------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| **SVG** | `.svg`    | Scalable Vector Graphics. Widely compatible with design software like Inkscape, Illustrator, and web browsers. |
| **DXF** | `.dxf`    | Drawing Exchange Format. Compatible with most CAD applications like AutoCAD, FreeCAD, and LibreCAD.            |

### Export Notes

- All workpieces from all layers are combined into a single file
- Workpiece positions are preserved
- Empty workpieces are skipped
- The bounding box encompasses all geometry

### Use Cases

- **Project sharing**: Export entire project for collaboration
- **Backup**: Create a visual archive of your work
- **Further editing**: Take the whole design into Inkscape or CAD software

---

## Exporting G-code

Generated G-code contains everything exactly as it would be sent to the machine.
The exact format, commands, numeric precision, etc. depends on the settings of
the currently selected machine and its G-code dialect.

---

### Export Methods

### Method 1: File Menu

**File Export G-code** (Ctrl+E)

- Opens file save dialog
- Choose location and filename
- G-code generated and saved

### Method 2: Command Line

```bash
# Export from command line (if supported)
rayforge --export output.gcode input.svg
```

---

### G-code Output

Generated G-code contains everything exactly as it would be sent to the machine.
The exact format, commands, numeric precision, etc. depends on the settings of
the currently selected machine and its G-code dialect.

---

## Related Pages

- [Importing Files](importing) - Getting designs into Rayforge
- [Supported Formats](formats) - File format details
- [G-code Dialects](../reference/gcode-dialects) - Dialect differences
- [Hooks & Macros](../machine/hooks-macros) - Customizing output
- [Simulation Mode](../features/simulation-mode) - Preview before export
