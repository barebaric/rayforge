---
description:
  "Configure general machine settings in Rayforge — set the machine name, select a driver, and
  configure speeds and acceleration."
---

# General Settings

The General page in Machine Settings contains the machine name, driver selection and connection
settings, and speed parameters.

![General Settings](/screenshots/machine-settings-general.webp)

## Machine Name

Give your machine a descriptive name. This helps identify the machine in the machine selector
dropdown when you have multiple machines configured.

## Driver

Select the driver that matches your machine's controller. The driver handles communication between
Rayforge and the hardware.

GRBL devices can connect over USB/serial or over the network:

- **GRBL (USB/Serial)** — Buffer-counting driver with deadlock detection and stall recovery. Its
  GRBL protocol stack runs in Rust. Recommended for most GRBL devices
- **GRBL (USB/Serial — compatibility mode)** — Ping-pong protocol driver. Sends one line, waits for
  "ok", sends the next. No buffer management, no deadlock detection. Only needed when the standard
  driver causes false alarms or communication errors
- **GRBL (Bluetooth)** — Bluetooth Classic SPP/RFCOMM connection for controllers paired in the
  operating system
- **grblHAL (Telnet)** — Raw TCP/telnet connection for networked grblHAL controllers and for ESP3D's
  telnet bridge
- **FluidNC / ESP3D (Web API)** — HTTP/WebSocket driver for FluidNC and ESP3D v3 controllers, with
  file upload and device settings support

xTool machines on Wi-Fi use their own experimental drivers; see [xTool Wi-Fi](xtool) for the D1/D1
Pro and the M2.

Ruida-based controllers are supported by the **Ruida RPA** driver, which connects over USB or UDP
directly, or via TUI RPC through the Ruida Protocol Analyzer. Its USB field offers a dropdown of
connected devices with VID:PID matching, and it honors each step's Power Mode setting — Dynamic
enables the controller's power scaling, Constant disables it. Power behavior is tuned with the
driver's **Power Scaling**, **VECTOR power floor** and **IMAGE power bias** options: power scaling
raises the emitted minimum power as the layer's cut speed decreases, the vector power floor
compensates over-burn at the ends of lines, and the image bias compensates for CO2 tubes that do not
fire at very low power.

### Epilog

Epilog lasers with an Ethernet port (Zing 16, Zing 24/6030) use the **Epilog** driver. Jobs are
encoded as Epilog print jobs and uploaded to the machine's print queue; start them from the laser's
own job list. The driver offers the machine's IP address, the print port (515 by default), the print
queue name (`Laser`, as in Epilog's setup instructions), the raster resolution (100–1000 DPI) and
the vector frequency (10–5000 Hz).

The print protocol offers no remote control, so Rayforge cannot home, jog, query the status or move
the head on these machines, and focus and air assist are set at the laser. The job language always
addresses a top-left page, so the coordinate origin is fixed to **Top Left**. **Send** uploads the
job and **Frame** uploads a beam-off outline job you start the same way. Speed is converted to the
machine's percentage scale: Rayforge uses 6000 mm/min as the 100% reference, so 600 mm/min is 10%.
Dither and threshold engravings are sent as native 1-bit raster rows; power-modulated (grayscale)
engravings use the machine's per-dot power mode. Rotary jobs are not supported.

### Serial Port Binding

Instead of a device path (e.g. `/dev/ttyUSB0` or `COM3`), the serial port field also accepts a USB
`VID:PID` identifier such as `0403:6001`. When a machine is bound by VID:PID, auto-reconnect follows
it to its new port after the OS re-enumerates USB devices — for example after a reboot or when the
machine is unplugged and reconnected. You can find a device's VID:PID in the output of `lsusb`
(Linux) or Device Manager → Hardware IDs (Windows).

After selecting a driver, connection-specific settings appear below the selector (e.g. serial port,
baud rate). These vary depending on the chosen driver.

### Custom Baud Rates

The baud rate selector lists common rates, but any value between 300 and 4000000 bits per second is
accepted. Choose **Custom...** at the end of the list and enter the rate your firmware uses — for
example `500000`. A configured custom rate stays visible in the selector.

<!-- prettier-ignore-start -->
:::tip
An error banner at the top of the page warns you if the driver is not configured or
encounters a problem.
:::
<!-- prettier-ignore-end -->

## Speeds & Acceleration

These settings control the maximum speeds and acceleration. They are used for job time estimation
and path optimization.

### Max Travel Speed

The maximum speed for rapid (non-cutting) movements when the laser is off and the head is moving to
a new position.

- **Typical range**: 2000-5000 mm/min
- **Note**: Actual speed is also limited by your firmware settings. This field is disabled if the
  selected G-code dialect does not support specifying a travel speed.

### Max Cut Speed

The maximum speed allowed during cutting or engraving operations.

- **Typical range**: 500-2000 mm/min
- **Note**: Individual operations may use lower speeds

### Acceleration

The rate at which the machine accelerates and decelerates, used for time estimations and calculating
the default overscan distance.

- **Typical range**: 500-2000 mm/s²
- **Note**: Must match or be lower than firmware acceleration settings

<!-- prettier-ignore-start -->
:::tip
Start with conservative speed values and increase gradually. Observe your machine for belt
skipping, motor stalling, or loss of positioning accuracy.
:::
<!-- prettier-ignore-end -->

## Exporting a Machine Profile

Click the share icon in the header bar of the settings dialog to export the current machine
configuration. Choose a folder to save to. A zip file is created containing the machine settings and
its G-code dialect, which can be shared with other users or imported on another system.

## See Also

- [First Time Setup](../getting-started/first-time-setup.md) - Create a machine step by step with
  the configuration wizard
- [Hardware Settings](hardware) - Work area dimensions and axis configuration
- [Device Settings](device) - Read and write firmware settings on the controller
