# G-code Dialect Support

RayForge supports multiple G-code dialects to work with different controller firmware.

## Supported Dialects

RayForge currently supports these G-code dialects:

| Dialect | Firmware | Common Use | Status |
|---------|----------|------------|--------|
| **GRBL (universal)** | GRBL 1.1+ | Diode lasers, hobby CNC |  Primary, fully supported |
| **GRBL (no Z axis)** | GRBL 1.1+ | 2D laser cutters without Z |  Optimized variant |
| **Smoothieware** | Smoothieware | Laser cutters, CNC |  Experimental |
| **Marlin** | Marlin 2.0+ | 3D printers with laser |  Experimental |

!!! note "Recommended Dialect"
    **GRBL (universal)** is the most tested and recommended dialect for laser applications.

---

## Related Pages

- [Exporting G-code](../files/exporting.md) - Export settings
- [Firmware Compatibility](firmware.md) - Firmware versions
- [GRBL Settings](../machine/grbl-settings.md) - GRBL configuration
- [Macros & Hooks](../features/macros-hooks.md) - Custom G-code injection
