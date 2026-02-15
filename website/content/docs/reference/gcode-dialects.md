# G-code Dialect Support

Rayforge supports multiple G-code dialects to work with different controller firmware.

## Supported Dialects

Rayforge currently supports these G-code dialects:

| Dialect                        | Firmware     | Common Use                  | Status                          |
| ------------------------------ | ------------ | --------------------------- | ------------------------------- |
| **GRBL (universal)**           | GRBL 1.1+    | Diode lasers, hobby CNC     |  Primary, fully supported      |
| **GRBL (no Z axis)**           | GRBL 1.1+    | 2D laser cutters without Z  |  Optimized variant             |
| **GRBL Dynamic (Depth-Aware)** | GRBL 1.1+    | Depth-Aware laser engraving |  Recommended for dynamic power |
| **GRBL Dynamic (no Z axis)**   | GRBL 1.1+    | Depth-Aware laser engraving |  Optimized variant             |
| **Smoothieware**               | Smoothieware | Laser cutters, CNC          |  Experimental                  |
| **Marlin**                     | Marlin 2.0+  | 3D printers with laser      |  Experimental                  |

!!! note "Recommended Dialects"
**GRBL (universal)** is the most tested and recommended dialect for standard laser applications.

    **GRBL Dynamic (Depth-Aware)** is recommended for Depth-Aware laser engraving where power varies during cuts (e.g., variable depth engraving).
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

- [Exporting G-code](../files/exporting.md) - Export settings
- [Firmware Compatibility](firmware.md) - Firmware versions
- [GRBL Settings](../machine/grbl-settings.md) - GRBL configuration
- [Macros & Hooks](../features/macros-hooks.md) - Custom G-code injection
