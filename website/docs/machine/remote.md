---
description:
  "Run your machine from a small computer next to the hardware while Rayforge runs elsewhere. Jobs
  survive disconnects and reconnects resync progress exactly."
---

# Remote Devices

Run your machine from a small computer next to the hardware (for example a Raspberry Pi in the
workshop) while Rayforge runs on your laptop or desktop. Jobs run on the device host and keep going
even if Rayforge disconnects; reconnecting resyncs progress exactly.

## How it works

- The **raydriver daemon** hosts the machine connection next to the hardware.
- Rayforge keeps your machine configuration and submits jobs over the LAN.
- Every G-code driver (GRBL serial, GRBL telnet, GRBL simple, Smoothie, Marlin) works locally or
  remotely — remote is a connection setting, not a separate driver.

## Set up the device host

1. Install raydriver on the device host: `pip install raydriver` (add `[daemon]` for network
   discovery).
2. Run `raydriver daemon --setup`. This writes `~/.config/raydriver/daemon.json` with a generated
   token and prints the certificate command.
3. For LAN access, create a certificate with the printed `openssl` one-liner and set `tls_cert` and
   `tls_key` in the config. A daemon on a non-loopback address refuses to start without both a token
   and TLS.
4. Run `raydriver daemon install` to install and start the background service (systemd on Linux,
   launchd on macOS, a logon task on Windows), or run `raydriver daemon` in the foreground.

## Connect Rayforge

- The machine wizard discovers daemons on the network and lists their devices.
- Alternatively, enable **Remote device** in the driver's connection settings and enter the daemon
  hostname, port, token and TLS certificate.
- Probing works over the remote connection, so the wizard can identify hardware that is not attached
  to your computer.

## Jobs and reconnects

- A submitted job runs to completion on the device host even if Rayforge closes or the network
  drops.
- Reconnect to see exact progress. One client at a time holds control; others can watch.

## Cameras on the device host

Video does not travel over the control connection. Use Rayforge's [network cameras](camera.md) with
a stream reachable on the device host:

- **IP camera**: add its stream URL directly.
- **USB camera on the device host**: run a standard streamer such as `ustreamer` and add its MJPEG
  URL as an HTTP stream camera.

The daemon advertises configured camera endpoints so the wizard can prefill the URL.

## Security notes

- LAN (or tunnel) only — no cloud, no accounts.
- Token and TLS are mandatory for any non-loopback daemon.
- Keep camera streams LAN-only or firewalled; they are separate from the TLS control channel.
