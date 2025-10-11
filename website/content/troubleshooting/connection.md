# Connection Issues

This page helps you diagnose and resolve problems connecting RayForge to your laser machine via serial connection.

## Quick Diagnosis

### Symptoms

Common connection problems include:

- "Port must be configured" error when trying to connect
- Connection repeatedly failing and reconnecting
- Serial port not appearing in the port list
- "Permission denied" errors when trying to open serial port
- Device appears to connect but doesn't respond to commands

---

## Common Problems and Solutions

### No Serial Ports Detected

**Problem:** The serial port dropdown is empty or doesn't show your device.

**Diagnosis:**

1. Check if your device is powered on and connected via USB
2. Try unplugging and re-plugging the USB cable
3. Test the USB cable with another device (cables can fail)
4. Try a different USB port on your computer

**Solutions:**

=== "Linux"
    If you're using the Snap version, you need to grant serial port permissions:

    ```bash
    sudo snap connect rayforge:serial-port
    ```

    See [Snap Permissions](snap-permissions.md) for detailed Linux setup.

    For non-Snap installations, add your user to the `dialout` group:

    ```bash
    sudo usermod -a -G dialout $USER
    ```

    Then log out and log back in for the change to take effect.

=== "Windows"
    1. Open Device Manager (Win+X, then select Device Manager)
    2. Look under "Ports (COM & LPT)" for your device
    3. If you see a yellow warning icon, update or reinstall the driver
    4. Note the COM port number (e.g., COM3)
    5. If the device isn't listed at all, the USB cable or driver may be faulty

=== "macOS"
    1. Check System Information  USB to verify the device is recognized
    2. Install CH340/CH341 drivers if your controller uses this chipset
    3. Check for `/dev/tty.usbserial*` or `/dev/cu.usbserial*` devices

### Permission Denied Errors

**Problem:** You get "Permission denied" or similar errors when trying to connect.

**On Linux (non-Snap):**

Your user needs to be in the `dialout` group (or `uucp` on some distributions):

```bash
# Add yourself to the dialout group
sudo usermod -a -G dialout $USER

# Verify you're in the group (after logging out/in)
groups | grep dialout
```

**Important:** You must log out and log back in (or reboot) for group changes to take effect.

**On Linux (Snap):**

Grant serial port access to the snap:

```bash
sudo snap connect rayforge:serial-port
```

See the [Snap Permissions](snap-permissions.md) guide for more details.

**On Windows:**

Close any other applications that might be using the serial port (including:
- Previous instances of RayForge
- Serial monitor tools
- Other laser software
- Arduino IDE or similar tools

### Wrong Serial Port Selected

**Problem:** RayForge connects but the machine doesn't respond.

**Diagnosis:**

You might have selected the wrong port, especially if you have multiple USB devices connected.

**Solution:**

1. Disconnect all other USB serial devices
2. Note which ports are available in RayForge
3. Plug in your laser controller
4. Refresh the port list - the new port is your laser
5. On Linux, laser controllers typically appear as:
   - `/dev/ttyUSB0` (common for CH340 chipsets)
   - `/dev/ttyACM0` (common for native USB controllers)
6. On Windows, note the COM port from Device Manager
7. Avoid ports named `/dev/ttyS*` on Linux - these are hardware serial ports, not USB

!!! warning "Hardware Serial Ports"
    RayForge will warn you if you select `/dev/ttyS*` ports on Linux, as these are typically not USB-based GRBL devices. USB serial ports use `/dev/ttyUSB*` or `/dev/ttyACM*`.

### Incorrect Baud Rate

**Problem:** Connection establishes but commands don't work or produce garbled responses.

**Solution:**

GRBL controllers typically use one of these baud rates:

- **115200** (most common, GRBL 1.1+)
- **9600** (older GRBL versions)
- **250000** (less common, some custom firmware)

Try different baud rates in RayForge's device settings. The most common is **115200**.

### Connection Keeps Dropping

**Problem:** RayForge connects successfully but keeps disconnecting and reconnecting.

**Possible Causes:**

1. **Flaky USB cable** - Replace with a known-good cable (preferably short, <2m)
2. **USB power issues** - Try a different USB port, preferably on the computer itself rather than a hub
3. **EMI/Interference** - Keep USB cables away from motor wires and high-voltage power supplies
4. **Firmware issues** - Update your GRBL firmware if possible
5. **USB port conflicts** - On Windows, try different USB ports

**Troubleshooting Steps:**

```bash
# On Linux, monitor system logs while connecting:
sudo dmesg -w
```

Look for messages like:
- "USB disconnect" - indicates physical/cable issues
- "device descriptor read error" - often a power or cable problem

### Device Not Responding After Connection

**Problem:** Connection status shows "Connected" but the machine doesn't respond to commands.

**Diagnosis:**

1. Check that the correct firmware type is selected (GRBL vs other)
2. Verify the machine is powered on (controller and power supply)
3. Check if the machine is in an alarm state (requires homing or alarm clear)

**Solution:**

Try sending a manual command in the debug console (if available):

- `?` - Request status report
- `$X` - Clear alarm
- `$H` - Home the machine

If there's no response, double-check baud rate and port selection.

---

## Connection Status Messages

RayForge shows different connection states:

| Status | Meaning | Action |
|--------|---------|--------|
| **Disconnected** | Not connected to any device | Configure port and connect |
| **Connecting** | Attempting to establish connection | Wait, or check configuration if stuck |
| **Connected** | Successfully connected and receiving status | Ready to use |
| **Error** | Connection failed with an error | Check error message for details |
| **Sleeping** | Waiting before reconnection attempt | Previous connection failed, retrying in 5s |

---

## Testing Your Connection

### Step-by-Step Connection Test

1. **Configure the machine:**
   - Open Settings  Machine
   - Select or create a machine profile
   - Choose the correct driver (GRBL Serial)
   - Select the serial port
   - Set baud rate (typically 115200)

2. **Attempt connection:**
   - Click "Connect" in the machine control panel
   - Watch the connection status indicator

3. **Verify communication:**
   - If connected, try sending a status query
   - The machine should report its position and state

4. **Test basic commands:**
   - Try homing (`$H`) if your machine has limit switches
   - Or clear alarms (`$X`) if needed

### Using Debug Logs

RayForge includes detailed debug logging for connection issues.

To enable verbose logging:

```bash
# Run RayForge from terminal with debug logging
RAYFORGE_LOG_LEVEL=DEBUG rayforge
```

Check the logs for:
- Connection attempts and failures
- Serial data transmitted (TX) and received (RX)
- Error messages with stack traces

---

## Advanced Troubleshooting

### Checking Port Availability Manually

=== "Linux"
    ```bash
    # List all USB serial devices
    ls -l /dev/ttyUSB* /dev/ttyACM*

    # Check permissions
    ls -l /dev/ttyUSB0  # Replace with your port

    # Should show: crw-rw---- 1 root dialout
    # You need to be in the 'dialout' group

    # Test port manually
    sudo minicom -D /dev/ttyUSB0 -b 115200
    ```

=== "Windows"
    ```powershell
    # List COM ports in PowerShell
    [System.IO.Ports.SerialPort]::getportnames()

    # Or use Device Manager:
    # Win + X  Device Manager  Ports (COM & LPT)
    ```

### Firmware Compatibility

RayForge is designed for GRBL-compatible firmware. Ensure your controller runs:

- **GRBL 1.1** (most common, recommended)
- **GRBL 0.9** (older, may have limited features)
- **grblHAL** (modern GRBL fork, supported)

Other firmware types (Marlin, Smoothieware) are not currently supported via the GRBL driver.

### USB-to-Serial Chipsets

Common chipsets and their drivers:

| Chipset | Linux | Windows | macOS |
|---------|-------|---------|-------|
| **CH340/CH341** | Built-in kernel driver | [CH341SER driver](http://www.wch.cn/downloads/) | Requires driver |
| **FTDI FT232** | Built-in kernel driver | Built-in (Windows 10+) | Built-in |
| **CP2102 (SiLabs)** | Built-in kernel driver | Built-in (Windows 10+) | Built-in |

---

## Still Having Issues?

If you've tried everything above and still can't connect:

1. **Check the GitHub issues** - Someone may have reported the same problem
2. **Create a detailed issue report** with:
   - Operating system and version
   - RayForge version (Snap/Flatpak/AppImage/source)
   - Controller board model and firmware version
   - USB chipset (check Device Manager on Windows or `lsusb` on Linux)
   - Full error messages and debug logs
3. **Test with another application** - Try connecting with a serial terminal (minicom, PuTTY, Arduino Serial Monitor) to verify the hardware works

---

## Related Pages

- [Snap Permissions](snap-permissions.md) - Detailed Linux Snap permission setup
- [Common Problems](common.md) - Other common issues and solutions
- [Device Configuration](../machine/device-config.md) - Machine setup guide
- [GRBL Settings](../machine/grbl-settings.md) - GRBL configuration reference
