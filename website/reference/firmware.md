# Firmware Compatibility

This page documents firmware compatibility for laser controllers used with RayForge.

## Overview

RayForge is designed primarily for **GRBL-based controllers** but has experimental support for other firmware types.

### Compatibility Matrix

| Firmware | Version | Status | Driver | Notes |
|----------|---------|--------|--------|-------|
| **GRBL** | 1.1+ |  Fully supported | GRBL Serial | Recommended |
| **grblHAL** | 2023+ |  Compatible | GRBL Serial | Modern GRBL fork |
| **GRBL** | 0.9 |  Limited | GRBL Serial | Older, may have issues |
| **Smoothieware** | All |  Experimental | None (use GRBL driver) | Untested |
| **Marlin** | 2.0+ |  Experimental | None (use GRBL driver) | Laser mode required |
| **Other** | - |  Not supported | - | Request support |

---

## GRBL Firmware

**Status:**  Fully Supported
**Versions:** 1.1+
**Driver:** GRBL Serial

### GRBL 1.1 (Recommended)

**What is GRBL 1.1?**

GRBL 1.1 is the most common firmware for hobby CNC and laser machines. Released in 2017, it's stable, well-documented, and widely supported.

**Features supported by RayForge:**

-  Serial communication (USB)
-  Real-time status reporting
-  Laser mode (M4 constant power)
-  Settings read/write ($$, $X=value)
-  Homing cycles ($H)
-  Work coordinate systems (G54)
-  Jogging commands ($J=)
-  Feed rate override
-  Soft limits
-  Hard limits (endstops)

**Known limitations:**

- Power range: 0-1000 (S parameter)
- No network connectivity (USB only)
- Limited onboard memory (small G-code buffer)

### Checking GRBL Version

**Query version:**

Connect to your controller and send:
```
$I
```

**Response examples:**
```
[VER:1.1h.20190825:]
[OPT:V,15,128]
```

- `1.1h` = GRBL version 1.1h
- Date indicates build

### GRBL 0.9 (Older)

**Status:**  Limited Support

GRBL 0.9 is an older version with some compatibility issues:

**Differences:**

- Different status report format
- No laser mode (M4) - uses M3 only
- Fewer settings
- Different jogging syntax

**If you have GRBL 0.9:**

1. **Upgrade to GRBL 1.1** if possible (recommended)
2. **Use M3 instead of M4** (less predictable power)
3. **Test thoroughly** - some features may not work

**Upgrade instructions:** See [GRBL Wiki](https://github.com/gnea/grbl/wiki)

---

## grblHAL

**Status:**  Compatible
**Versions:** 2023+
**Driver:** GRBL Serial

### What is grblHAL?

grblHAL is a modern fork of GRBL with enhanced features:

- Multiple controller hardware support (STM32, ESP32, etc.)
- Ethernet/WiFi networking
- SD card support
- More I/O pins
- Enhanced laser support

**Compatibility with RayForge:**

-  **Fully compatible** - grblHAL maintains GRBL 1.1 protocol
-  All GRBL features work
-  Additional features (networking, SD) not yet supported by RayForge
-  Status reporting identical to GRBL

**Using grblHAL:**

1. Select "GRBL Serial" driver in RayForge
2. Connect via USB serial (just like GRBL)
3. All features work as documented for GRBL

**Future:** RayForge may add support for grblHAL-specific features (networking, etc.)

---

## Smoothieware

**Status:**  Experimental
**Versions:** All
**Driver:** GRBL Serial (compatibility mode)

### Compatibility Notes

Smoothieware uses different G-code syntax:

**Key differences:**

| Feature | GRBL | Smoothieware |
|---------|------|--------------|
| **Laser On** | M4 S{power} | M3 S{power} |
| **Power Range** | 0-1000 | 0.0-1.0 (float) |
| **Status** | `<...>` format | Different format |

**Using Smoothieware with RayForge:**

1. **Select Smoothieware dialect** in machine settings
2. **Test with low power** first
3. **Verify power range** matches your config
4. **No real-time status** - limited feedback

**Limitations:**

- Status reporting not fully compatible
- Power scaling may differ
- Settings ($$ commands) not supported
- Untested on real hardware

**Recommendation:** If possible, use GRBL-compatible firmware instead.

---

## Marlin

**Status:**  Experimental
**Versions:** 2.0+ with laser support
**Driver:** GRBL Serial (limited compatibility)

### Marlin for Laser

Marlin 2.0+ can control lasers when properly configured.

**Requirements:**

1. **Marlin 2.0 or later** firmware
2. **Laser features enabled:**
   ```cpp
   #define LASER_FEATURE
   #define LASER_POWER_INLINE
   ```
3. **Correct power range** configured:
   ```cpp
   #define SPEED_POWER_MAX 1000
   ```

**Compatibility:**

-  M4 laser mode supported
-  Basic G-code (G0, G1, G2, G3)
-  Status reporting differs
-  Settings commands different
-  Air assist (M8/M9) may not work

**Using Marlin with RayForge:**

1. **Select Marlin dialect** in machine settings
2. **Configure Marlin** for laser use
3. **Test power range** matches (0-1000 or 0-255)
4. **Limited testing** - use with caution

**Better alternative:** Use GRBL firmware on laser machines.

---

## Firmware Upgrade Guide

### Upgrading to GRBL 1.1

**Why upgrade?**

- Laser mode (M4) for constant power
- Better status reporting
- More reliable
- Better RayForge support

**How to upgrade:**

1. **Identify your controller board:**
   - Arduino Nano/Uno (ATmega328P)
   - Arduino Mega (ATmega2560)
   - Custom board

2. **Download GRBL 1.1:**
   - [GRBL Releases](https://github.com/gnea/grbl/releases)
   - Get latest 1.1 version (1.1h recommended)

3. **Flash firmware:**

   **Using Arduino IDE:**
   ```
   1. Install Arduino IDE
   2. Open GRBL sketch (grbl.ino)
   3. Select correct board and port
   4. Upload
   ```

   **Using avrdude:**
   ```bash
   avrdude -c arduino -p m328p -P /dev/ttyUSB0 \
           -U flash:w:grbl.hex:i
   ```

4. **Configure GRBL:**
   - Connect via serial
   - Send `$$` to view settings
   - Configure for your machine

### Backup Before Upgrade

**Save your settings:**

1. Connect to controller
2. Send `$$` command
3. Copy all settings output
4. Save to file

**After upgrade:**

- Restore settings one-by-one: `$0=10`, `$1=25`, etc.
- Or use defaults and reconfigure

---

## Controller Hardware

### Common Controllers

| Board | Typical Firmware | RayForge Support |
|-------|------------------|------------------|
| **Arduino CNC Shield** | GRBL 1.1 |  Excellent |
| **MKS DLC32** | grblHAL |  Excellent |
| **Cohesion3D** | Smoothieware |  Limited |
| **SKR boards** | Marlin/grblHAL |  Varies |
| **Ruida** | Proprietary |  Not supported |
| **Trocen** | Proprietary |  Not supported |
| **TopWisdom** | Proprietary |  Not supported |

### Recommended Controllers

For best RayForge compatibility:

1. **Arduino Nano + CNC Shield** (GRBL 1.1)
   - Cheap (~$10-20)
   - Easy to flash
   - Well documented

2. **MKS DLC32** (grblHAL)
   - Modern (ESP32-based)
   - WiFi capable
   - Active development

3. **Custom GRBL boards**
   - Many available on marketplaces
   - Check for GRBL 1.1+ support

---

## Firmware Configuration

### GRBL Settings for Laser

**Essential settings:**

```
$30=1000    ; Max spindle/laser power (1000 = 100%)
$31=0       ; Min spindle/laser power
$32=1       ; Laser mode enabled (1 = on)
```

**Machine settings:**

```
$100=80     ; X steps/mm (calibrate for your machine)
$101=80     ; Y steps/mm
$110=3000   ; X max rate (mm/min)
$111=3000   ; Y max rate
$120=100    ; X acceleration (mm/sec)
$121=100    ; Y acceleration
$130=300    ; X max travel (mm)
$131=200    ; Y max travel (mm)
```

**Safety settings:**

```
$20=1       ; Soft limits enabled
$21=1       ; Hard limits enabled (if you have endstops)
$22=1       ; Homing enabled
```

### Testing Firmware

**Basic test sequence:**

1. **Connection test:**
   ```
   Send: ?
   Expect: <Idle|...>
   ```

2. **Version check:**
   ```
   Send: $I
   Expect: [VER:1.1...]
   ```

3. **Settings check:**
   ```
   Send: $$
   Expect: $0=..., $1=..., etc.
   ```

4. **Movement test:**
   ```
   Send: G91 G0 X10
   Expect: Machine moves 10mm in X
   ```

5. **Laser test (very low power):**
   ```
   Send: M4 S10
   Expect: Laser turns on (dim)
   Send: M5
   Expect: Laser turns off
   ```

---

## Troubleshooting Firmware Issues

### Firmware Not Responding

**Symptoms:**
- No response to commands
- Connection fails
- Status not reported

**Diagnosis:**

1. **Check baud rate:**
   - GRBL 1.1 default: 115200
   - GRBL 0.9: 9600
   - Try both

2. **Check USB cable:**
   - Data cable, not charge-only
   - Replace with known-good cable

3. **Check port:**
   - Linux: `/dev/ttyUSB0` or `/dev/ttyACM0`
   - Windows: COM3, COM4, etc.
   - Correct port selected in RayForge

4. **Test with terminal:**
   - Use screen, minicom, or PuTTY
   - Send `?` and see if you get response

### Firmware Crashes

**Symptoms:**
- Controller locks up during job
- Random disconnections
- Inconsistent behavior

**Possible causes:**

1. **Buffer overflow** - G-code file too complex
2. **Electrical noise** - Poor grounding or EMI
3. **Firmware bug** - Upgrade to latest version
4. **Hardware issue** - Faulty controller

**Solutions:**

- Upgrade firmware to latest stable version
- Simplify G-code (reduce precision, fewer segments)
- Add ferrite beads to USB cable
- Improve grounding and cable routing

### Wrong Firmware

**Symptoms:**
- Commands rejected
- Unexpected behavior
- Error messages

**Solution:**

1. Query firmware version: `$I`
2. Compare with RayForge expectations
3. Upgrade or select correct dialect

---

## Future Firmware Support

### Requested Features

Users have requested support for:

- **Ruida controllers** - Chinese laser controllers
- **Trocen/AWC** - Commercial laser controllers
- **ESP32 WiFi** - Network connectivity for grblHAL
- **Laser API** - Direct machine API (no G-code)

**Status:** Not currently supported. Feature requests welcome on GitHub.

### Contributing

To add firmware support:

1. Implement driver in `rayforge/machine/driver/`
2. Define G-code dialect in `rayforge/machine/models/dialect.py`
3. Test thoroughly on real hardware
4. Submit pull request with documentation

---

## Related Pages

- [G-code Dialects](gcode-dialects.md) - Dialect details
- [GRBL Settings](../machine/grbl-settings.md) - GRBL configuration
- [Connection Issues](../troubleshooting/connection.md) - Connection troubleshooting
- [Device Configuration](../machine/device-config.md) - Machine setup
