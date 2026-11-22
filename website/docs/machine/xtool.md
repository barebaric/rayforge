---
description:
  "Connect xTool D1 and D1 Pro laser cutters and the xTool M2 over Wi-Fi with Rayforge's
  experimental xTool drivers."
---

# xTool Wi-Fi

Rayforge includes experimental drivers for xTool machines connected over Wi-Fi. They talk to xTool's
proprietary network API instead of the GRBL Telnet or FluidNC / ESP3D interfaces, so the machine
must be set up with its own xTool driver.

## Supported Machines

| Profile          | Driver           | What It Can Do                                                           |
| ---------------- | ---------------- | ------------------------------------------------------------------------ |
| **xTool D1 Pro** | xTool (Wi-Fi)    | Upload and start jobs, home, jog, pause, resume and cancel               |
| **xTool M2**     | xTool M2 (Wi-Fi) | Monitor the machine, home, jog, and pause, resume or cancel a staged job |

Both drivers are marked **experimental**. Test them with a scrap piece before relying on them for
production work, and report problems with a debug log attached.

## Setting Up an xTool Machine

1. Connect the machine to the same network as your computer and find its IP address (for example in
   the xTool app or in your router's device list).
2. Add a machine and pick the **xTool D1 Pro** or **xTool M2** device profile. The profile
   pre-configures the driver and the machine dimensions.
3. Enter the machine's **Hostname or IP address**.
4. Leave the ports at their defaults unless your network setup requires something else:
   - D1 Pro: HTTP port `8080`, WebSocket port `8081`
   - M2: port `28900`
5. Connect.

If Rayforge reports that no xTool machine answered, check the IP address, make sure the machine is
powered on, and confirm that your computer and the machine are on the same network.

## D1 and D1 Pro

The **xTool (Wi-Fi)** driver uploads the job to the machine, starts it, and follows its progress
through the machine's WebSocket job events. Homing, jogging and absolute moves, pause, resume and
cancel work as usual.

The machine's network API does not report its position, so Rayforge drives the position display from
the job state. Homing is available for all axes at once; per-axis homing is disabled. Writing
firmware settings, work coordinate offsets and probing are not supported by this driver.

## xTool M2

The **xTool M2 (Wi-Fi)** driver connects to the M2's TLS service at port `28900`. It monitors the
machine state and supports homing, jogging, and pause, resume or cancel of a job that is already
staged on the machine.

The M2's network API cannot receive G-code yet, so Rayforge cannot send its own jobs to it. Stage
the job in xTool Studio, start it there, and use Rayforge to monitor and control it. Z moves, tool
changes, remote laser power control, firmware settings and work coordinate offsets are not
supported.

## TLS Certificates

The M2 driver validates the machine's TLS certificate. If the machine uses a self-signed
certificate, enable **Allow invalid TLS certificates** in the driver settings. Only do this on a
network you trust.

## See Also

- [General Settings](general) - Driver selection, hostname and port fields
- [Firmware Reference](../reference/firmware.md) - Supported controllers and drivers
- [Remote Devices](remote) - Connecting to a machine through a raydriver daemon
