# Ruida (Raydriver) Driver

The **Ruida (Raydriver)** driver is an experimental alternative to the
default Ruida RPA driver. It connects Rayforge to Ruida-based laser
controllers over USB, UDP or TCP. The protocol stack, job compiler and
Ops encoder run in Rust via `raydriver.ruida`; no external Ruida library
is required. The existing Ruida RPA driver, its device profiles and its
TUI RPC mode remain available and are the default.

## Machine Settings Configuration

Add a Ruida machine manually and choose the **Ruida (Raydriver)**
driver. The driver is marked as untested.

Configuring both the **Hostname** and the **USB device** enables
automatic swap between connections when cables are connected and
disconnected. USB is preferred when both cables are connected because
of slightly better performance (no ACK handshake).

Enabling **TCP protocol** switches the network connection from UDP to
TCP; leave it disabled for the default UDP transport. The setting only
applies to network connections.

The **magic number** setting is used for other controllers. The default
of `0x88` will work with many common controllers. The value is entered
in hex.

The power settings control the encoder: **Power scaling** enables
effective-min power scaling, **Enable VECTOR power floor** and the
**VECTOR power floor** value control vector over-burn compensation, and
**IMAGE power bias** lifts raster scan-line power above the tube's
non-firing band.

## Golden Files

The `golden/` directory contains test fixture files that can be used to
compare results. The committed fixtures are the GlueScript transcript
(`transcript.cglu`) and the compiled wire stream (`job.hex`), produced
by the representative job in `regen_golden.py`.

## Limitations

- This driver has no TUI RPC mode; use the default Ruida RPA driver for
  that. Raydriver's `raydriver analyzer tui` is a live protocol monitor,
  not an RPC host; use a direct connection, or the raydriver daemon for
  remote setups.
- This version includes rudimentary acceleration and deceleration power
  compensation. It is a machine setting, so it applies per job.
- **WARNING**: The depthmap engrave mode has not been tested. Because it
  can use the Z axis for raising and lowering the bed, use with caution.
  Machines having a Z axis may not have a hard limit switch, so it is
  possible to crash the laser head into the bed. Machines equipped with a
  probe are still at risk if there is no flat surface for the probe to
  contact.
- Rotary is not supported because typical use requires swapping the Y
  axis connection with the rotary connection.
