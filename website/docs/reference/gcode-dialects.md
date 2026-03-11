# G-code Dialect Support

Rayforge supports multiple G-code dialects to work with different controller firmware.

## Supported Dialects

Rayforge currently supports these G-code dialects:

| Dialect                        | Firmware     | Common Use                  |
| ------------------------------ | ------------ | --------------------------- |
| **Grbl (Compat)**              | GRBL 1.1+    | Diode lasers, hobby CNC     |
| **Grbl (Compat, no Z axis)**   | GRBL 1.1+    | 2D laser cutters without Z  |
| **GRBL Dynamic (Depth-Aware)** | GRBL 1.1+    | Depth-Aware laser engraving |
| **GRBL Dynamic (no Z axis)**   | GRBL 1.1+    | Depth-Aware laser engraving |
| **Mach4 (M67 Analog)**         | Mach4        | High-speed raster engraving |
| **Smoothieware**               | Smoothieware | Laser cutters, CNC          |
| **Marlin**                     | Marlin 2.0+  | 3D printers with laser      |

:::note Recommended Dialects
:::

**Grbl (Compat)** is the most tested and recommended dialect for standard laser applications.

**GRBL Dynamic (Depth-Aware)** is recommended for Depth-Aware laser engraving where power varies during cuts (e.g., variable depth engraving).

---

## Mach4 (M67 Analog)

The **Mach4 (M67 Analog)** dialect is designed for high-speed raster engraving with Mach4 controllers. It uses the M67 command with analog output for precise laser power control.

### Key Features

- **M67 Analog Output**: Uses `M67 E0 Q<0-255>` for laser power instead of inline S commands
- **Reduced Buffer Pressure**: By separating power commands from motion commands, the controller buffer is less stressed during high-speed operations
- **High-Speed Raster**: Optimized for fast raster engraving operations

### When to Use

Use this dialect when:

- You have a Mach4 controller with analog output capability
- You need high-speed raster engraving
- Your controller experiences buffer overflow with standard inline S commands

### Command Format

The dialect generates G-code like:

```gcode
M67 E0 Q127  ; Set laser power to 50% (127/255)
G1 X100 Y200 F1000  ; Move to position
M67 E0 Q0    ; Turn laser off
```

---

## Creating a Custom Dialect

To create a custom G-code dialect based on a built-in dialect:

1. Open **Machine Settings** → **G-code Dialect**
2. Click the **Copy** icon on a built-in dialect to create a new custom dialect
3. Edit the dialect settings as needed
4. Save your custom dialect

Custom dialects are stored in your configuration directory and can be shared.

---

## Related Pages

- [Exporting G-code](../files/exporting) - Export settings
- [Firmware Compatibility](firmware) - Firmware versions
- [Device Settings](../machine/device) - GRBL configuration
- [Macros & Hooks](../machine/hooks-macros) - Custom G-code injection
