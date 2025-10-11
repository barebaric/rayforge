# Device Configuration

Device configuration determines how RayForge communicates with your laser cutter hardware. Choosing the correct driver and connection settings is essential for reliable operation.

## Overview

Device configuration includes:

- **Driver selection**: Communication protocol (Serial, Network, WiFi)
- **Connection parameters**: Port, baud rate, IP address, hostname
- **G-code dialect**: Command language variant (GRBL, Smoothie, Marlin)
- **Connection behavior**: Auto-homing, alarm clearing, retry logic

<!-- SCREENSHOT
id: ui-device-config-dialog
type: screenshot
size: dialog
description: |
  Device configuration dialog showing:
  - Driver type selector
  - Connection parameters (varies by driver)
  - Test connection button
  - Advanced settings section
setup:
  - action: open_preferences
  - action: navigate_to
    section: machine
  - action: open_device_config
  - action: capture
    region: dialog
filename: UI-DeviceConfig.png
alt: "Device configuration dialog"
-->

<!-- ![Device configuration dialog](../images/UI-Device-Config.png) -->

## Available Drivers

RayForge supports multiple communication methods to connect to different controller types.

### GRBL (Serial)

**Best for:**
- USB-connected laser cutters
- GRBL-based controllers (most common)
- Arduino/AVR-based systems
- Desktop and small format lasers

**Connection method:**
- Direct USB serial connection
- Virtual COM port (Windows) or /dev/ttyUSB* (Linux)
- Standard baud rates: 115200 (most common), 230400, 57600

**Configuration parameters:**

<!-- SCREENSHOT
id: ui-grbl-serial-settings
type: screenshot
size: custom
region:
  x: 0
  y: 0
  width: 0
  height: 0
description: |
  GRBL Serial driver configuration showing:
  - Port selection dropdown (/dev/ttyUSB0, /dev/ttyACM0, etc.)
  - Baud rate selector (115200 selected)
setup:
  - action: open_preferences
  - action: navigate_to
    section: machine
  - action: select_driver
    driver: GrblSerialDriver
  - action: capture
    region: driver_settings
filename: UI-GRBLSerial-Settings.png
alt: "GRBL Serial driver settings"
-->

<!-- ![GRBL Serial settings](../images/UI-GRBL-Serial-Settings.png) -->

**Port:**
- Serial port device path
- **Linux:** `/dev/ttyUSB0`, `/dev/ttyACM0`, etc.
- **Windows:** `COM3`, `COM4`, etc.
- **macOS:** `/dev/tty.usbserial-*`

!!! warning "Permission Issues on Linux"
    On Linux, you may need permissions to access serial ports. See [Snap Permissions](../troubleshooting/snap-permissions.md) if running the snap package, or add your user to the `dialout` group for traditional installs:
    ```bash
    sudo usermod -a -G dialout $USER
    ```
    Log out and back in for changes to take effect.

**Baud Rate:**
- Communication speed in bits per second
- **115200**: Default for most GRBL controllers (recommended)
- **230400**: Faster, if controller supports it
- **57600**: Older boards or slow microcontrollers
- **9600**: Very old or specialized hardware

**Finding your port:**

=== "Linux"
    ```bash
    # List USB serial devices
    ls /dev/ttyUSB* /dev/ttyACM*

    # Or use dmesg to see what was detected
    dmesg | grep tty
    ```

=== "Windows"
    1. Open Device Manager
    2. Expand "Ports (COM & LPT)"
    3. Look for "USB Serial Port (COMx)"
    4. Note the COM number

=== "macOS"
    ```bash
    # List USB serial devices
    ls /dev/tty.usbserial-*
    ls /dev/cu.usbserial-*
    ```

### GRBL (Network)

**Best for:**
- WiFi-enabled GRBL controllers
- ESP32-based laser controllers
- Remote or wireless operation
- Controllers with WebUI (like Grbl_ESP32)

**Connection method:**
- HTTP and WebSocket protocols
- Direct network connection to controller
- No USB cable required

**Configuration parameters:**

**Hostname/IP Address:**
- IP address of the controller (e.g., `192.168.1.100`)
- Or hostname (e.g., `laser.local`)
- Must be reachable on your network

**Finding your controller's IP:**

```bash
# If controller supports mDNS
ping laser.local

# Or check your router's DHCP client list
# Or check controller's display/WebUI
```

**Network requirements:**
- Controller and computer on same network
- No firewall blocking ports 80, 81, or 8080
- Controller must support HTTP API (Grbl_ESP32, etc.)

**Advantages:**
- No USB cable required
- Can control from any computer on network
- Multiple clients can monitor status
- WebUI and RayForge can run simultaneously

**Disadvantages:**
- Network latency may affect responsiveness
- Requires WiFi-capable controller
- More complex troubleshooting if connectivity issues

### No Device (Simulator)

**Best for:**
- Testing RayForge without hardware
- Designing jobs offline
- Training and demonstrations
- Developing G-code workflows

**Behavior:**
- Simulates a connected laser
- Accepts all commands without error
- Reports simulated position and status
- Doesn't actually control any hardware

**Use cases:**
- Preview G-code output
- Test operation sequences
- Learn RayForge interface
- Design projects away from machine

!!! tip "Testing Without Risk"
    Use "No Device" driver to safely test complex jobs before running on real hardware. You can generate G-code and review it, then switch to your real driver when ready to cut.

## G-code Dialects

Different controllers speak slightly different "dialects" of G-code. Select the one matching your hardware.

### GRBL

**Most common dialect** for hobby laser cutters.

**Characteristics:**
- Uses `M3`/`M5` for laser on/off
- Power set with `S` parameter (S0-S1000)
- Supports `$` system commands
- Real-time status reporting (`?` command)

**Typical controllers:**
- Arduino + GRBL firmware
- Grbl_ESP32
- Most Chinese laser boards
- DIY laser builds

**Example G-code:**
```gcode
G21                 ; Millimeters
G90                 ; Absolute positioning
M3 S800             ; Laser on at 80% power
G1 X10 Y10 F500     ; Move and cut
M5                  ; Laser off
```

### Smoothie

**Less common**, used on more advanced controllers.

**Characteristics:**
- Similar to GRBL but different configuration system
- Uses `/dev/ttyACM*` typically
- Web-based configuration
- More sophisticated motion planning

**Typical controllers:**
- Smoothieboard
- Cohesion3D boards (some)
- Higher-end laser retrofits

**Example G-code:**
```gcode
G21                 ; Millimeters
G90                 ; Absolute positioning
M3                  ; Laser on
S0.8                ; Power (0.0-1.0 range)
G1 X10 Y10 F500     ; Move and cut
M5                  ; Laser off
```

### Marlin

**Primarily for 3D printers**, occasionally used for lasers.

**Characteristics:**
- 3D printer origins
- Laser support added via M3/M4 commands
- Complex configuration through firmware
- May require recompilation for laser mode

**Typical controllers:**
- RepRap boards (RAMPS, SKR, etc.) with laser module
- 3D printer + laser upgrade
- CNC routers with laser attachment

!!! info "Dialect Mismatch Symptoms"
    If commands are rejected or laser doesn't respond correctly, verify your dialect selection matches your controller's firmware. Check controller documentation or manufacturer website.

## Connection Settings

### Home on Start

**Enabled:**
- Machine runs homing cycle immediately after connection
- Finds physical home switches (X, Y origins)
- Establishes absolute coordinate system
- Required for accurate positioning

**Disabled:**
- No automatic homing
- Uses current position as origin
- Must home manually if needed

**When to enable:**
-  Your machine has home/limit switches
-  You want absolute positioning
-  Production workflows requiring repeatability

**When to disable:**
- L Machine lacks home switches
- L Home switches are broken/disconnected
- L You prefer manual homing control

!!! warning "Homing Without Switches"
    Enabling "Home on Start" on a machine without home switches will cause the machine to drive into physical limits, potentially damaging belts or losing position. Only enable if hardware supports it.

### Clear Alarm on Connect

**Enabled:**
- Automatically sends `$X` (unlock) command on connection
- Clears GRBL alarm state
- Allows movement immediately

**Disabled:**
- Alarm state must be cleared manually
- User must diagnose alarm cause
- Safer, prevents ignoring real issues

**When to enable:**
- Machine frequently enters alarm state on connection
- You understand the alarm cause (e.g., power cycle alarm)
- Convenience over safety checking

**When to disable:**
-  **Recommended default**
- You want to diagnose alarm causes
- Safety-critical applications
- Unknown alarm conditions

**GRBL alarms:**
- Hard limit triggered
- Soft limit exceeded
- Abort during cycle
- Probe fail
- Homing fail

See [GRBL documentation](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Alarms) for alarm details.

## Testing Your Configuration

### Connection Test

After configuring driver settings:

1. Click "Test Connection" button
2. RayForge attempts to connect to device
3. Success: Green indicator, "Connected" status
4. Failure: Error message with diagnostic info

<!-- SCREENSHOT
id: ui-connection-test-success
type: screenshot
size: dialog
description: |
  Connection test dialog showing successful connection with:
  - Green checkmark
  - "Connected successfully" message
  - Device info (GRBL version, etc.)
setup:
  - action: open_device_config
  - action: configure_driver
    driver: GrblSerialDriver
    port: /dev/ttyUSB0
    baudrate: 115200
  - action: click
    button: test_connection
  - action: wait_for_connection
  - action: capture
    region: dialog
filename: UI-ConnectionTest-Success.png
alt: "Successful connection test dialog"
-->

<!-- ![Connection test success](../images/UI-Connection-Test-Success.png) -->

### Verifying Communication

Once connected:

**Check console log:**
- Should show connection handshake
- GRBL welcome message
- Status reports

**Test movement:**
- Use jog controls to move laser head
- Verify movement direction matches commands
- Check homing if enabled

**Test framing:**
- Load a simple design
- Run frame operation (low power outline)
- Verify dimensions and position

## Common Connection Issues

### Port not found

**Symptoms:**
- Port doesn't appear in dropdown
- "Port not found" error

**Solutions:**
- Check USB cable is connected
- Try different USB port
- Check device power
- Linux: Verify permissions (see [Snap Permissions](../troubleshooting/snap-permissions.md))
- Windows: Install CH340/CP2102 drivers if needed

### Connection timeout

**Symptoms:**
- "Connection timeout" error
- Port opens but no response

**Solutions:**
- Verify baud rate (try 115200 first)
- Check if another program is using the port
- Restart controller (power cycle)
- Try different USB cable
- Check for hardware issues

### Commands rejected

**Symptoms:**
- "Unknown command" errors
- G-code not executing

**Solutions:**
- Verify G-code dialect matches controller
- Check for firmware compatibility
- Update GRBL firmware if outdated
- Review controller documentation

### Intermittent connection

**Symptoms:**
- Random disconnections
- Dropped commands
- "Connection lost" errors

**Solutions:**
- Replace USB cable (most common cause)
- Check for electrical interference
- Reduce baud rate (115200  57600)
- Verify adequate power supply to controller
- Check for loose connections

See [Connection Troubleshooting](../troubleshooting/connection.md) for detailed diagnostics.

## Advanced Configuration

### Multiple Controller Support

If you have multiple lasers:

1. Create separate machine profiles
2. Each profile has its own driver configuration
3. Switch between machines using machine selector

See [Machine Profiles](profiles.md) for details.

### Custom Baud Rates

Some controllers use non-standard baud rates:

- 256000 bps
- 921600 bps (high-speed controllers)
- Custom firmware rates

Check controller documentation or try standard rates first.

### Network Driver Advanced

**Custom ports:**
- Default: HTTP on 80, WebSocket on 81
- Some controllers use 8080 for HTTP
- Check controller's WebUI documentation

**Static IP vs. DHCP:**
- Static IP: Reliable, consistent connection
- DHCP: Convenient but IP may change
- Consider setting static IP on controller

**Firewall issues:**
- Ensure ports 80 and 81 are open
- Disable firewall temporarily to test
- Add firewall exception for RayForge

## G-code Precision

**Setting:** Number of decimal places for coordinates

**Values:**
- **1**: `X12.3` - Very small files, lower precision
- **2**: `X12.34` - Compact, adequate for most work
- **3**: `X12.345` - **Recommended default** - Good balance
- **4**: `X12.3456` - High precision, larger files
- **5**: `X12.34567` - Extreme precision, rarely needed

**When to adjust:**
- **Lower (1-2)**: Large engraving files, storage-limited controllers
- **Higher (4-5)**: Precision mechanical parts, fine detail work

**Impact:**
- Precision: More decimals = finer positioning resolution
- File size: More decimals = larger G-code files
- Performance: Larger files = longer transfer time

Most machines have 0.01mm resolution, making precision=3 optimal.

## Driver Development

RayForge's driver system is extensible. If your controller isn't supported:

**Options:**
1. Use generic GRBL driver (often works for GRBL-based controllers)
2. Request driver support (GitHub issues)
3. Contribute a driver (see developer docs)

**Driver interface:**
- Python-based
- Async I/O for responsiveness
- Signal-based event system
- VarSet for configuration UI

## Related Topics

- **[Machine Profiles](profiles.md)** - Complete machine configuration
- **[GRBL Settings](grbl-settings.md)** - Controller parameters ($$ settings)
- **[Connection Troubleshooting](../troubleshooting/connection.md)** - Fixing connection problems
- **[Snap Permissions](../troubleshooting/snap-permissions.md)** - Linux serial port access
