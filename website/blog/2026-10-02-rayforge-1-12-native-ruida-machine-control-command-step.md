---
slug: rayforge-1-12-native-ruida-machine-control-command-step
title: "Rayforge 1.12 - Native Ruida Support, Machine Control, Command Step"
authors: rayforge_team
tags: [release, 1.12, ruida, machine-control, camera, command-step]
description:
  "Rayforge 1.12 brings native Ruida support, arbitrary laser head positioning with pointer
  alignment, machine notes, network cameras, per-step power modes, and a new Command step for custom
  machine code."
---

Rayforge 1.12 is here, and it is a big one: 277 commits across 736 files. The headline is native
Ruida support, but this cycle also reworked how you move and aim the laser head, added machine
notes, network cameras, per-step power modes, and a new way to inject custom machine code.

<!-- truncate -->

## Ruida Machines, Natively Supported

Rayforge can now talk to Ruida-based controllers directly. The new **Ruida RPA** driver connects
over USB or UDP, or via TUI RPC through the Ruida Protocol Analyzer, and ships with all installs. It
honors the per-step Power Mode and the controller's power scaling, and framing became a driver
capability, so Ruida machines can trace the job outline with the beam off. New **Generic Ruida RPA**
and **Monport MP-570 60W CO2** device profiles are included, and the old prototype Ruida (UDP)
driver was removed.

This feature is almost entirely the work of @StevenIsaacs, who spent months reverse-engineering the
protocol and testing it against real hardware. Thank you!

## Move the Head Wherever You Want

Machine moves are no longer limited to predefined targets. A new **Move to Position** popover in the
Current Position area sends the head to any X/Y coordinate (and Z, when your machine has one) at
your configured jog speed. It also hosts the workarea corner and WCS origin shortcuts.

Two conveniences round it out: **Click Canvas to Move Head** works just like Click to Zero, the
canvas context menu gains a **Move Head Here** action, and **Ctrl+M** arms click-to-move from the
keyboard.

If your machine has a red-dot alignment laser, **Pointer Alignment** turns its dot into a
first-class aiming reference: with a per-head pointer offset set, every aiming move shifts so the
_pointer dot_ lands exactly on the aimed position. The canvas draws the pointer dot next to the beam
dot, and starting a job asks whether to turn alignment off first, since jobs always burn with the
unshifted beam.

## Machine Notes

A new **Notes** category in Machine Settings combines the guidance shipped with the device profile
with your own notes for the machine. Device notes are read-only; My Notes are edited with a built-in
Markdown editor (headings, emphasis, lists, links, code blocks, and expandable details sections).
Profile setup guidance also appears on the setup wizard's review page, and personal notes survive
profile updates (thanks to @atkaper, #471).

## Command Step

The new **Command** step injects custom machine code at any position in a layer's workflow:
pre-positioning the head, toggling air assist between operations, or sending controller-specific
codes. Each line is emitted verbatim exactly where the step sits, supports the same path variables
as macros (`machine.*`, `layer.*`, `job.*`), and travels unexpanded with your project. A step
warning appears when the active machine's driver does not consume G-code, such as Ruida (#449).

## Camera Upgrades

- **Network cameras** work as stream sources alongside USB cameras, with automatic reconnection
  after read failures (thanks to @atkaper, #438)
- Lens calibration now supports **ArUco/AprilTag marker grids** and printable **dot grids**
  alongside ChArUco boards. Marker grids tolerate partial views, and their dictionary, ID offset,
  origin corner and numbering order are all editable, so factory-printed patterns can be calibrated
  against directly instead of printing a new card (thanks to @TOverbye, #466)

## Sketcher

The sketcher keeps improving: array tools now also lay out text boxes (#399), helper geometry moves
together with the array members it belongs to, and dragging elements no longer resizes the main
window (#385).

## Power Modes and Framing

Each step now has a **Power Mode**: Dynamic (M4) or Constant (M3). Constant power avoids power sags
at corners during vector cuts, while raster engraving keeps dynamic power (#437). The **Frame**
operation gains a **Round Corners** toggle and a corner radius setting, and framing at **0% power**
traces the outline with the beam off -- ideal for machines with an auxiliary alignment laser.

## An Experimental Rust GRBL Driver

The complete GRBL serial protocol stack -- character-counting flow control, job streaming, stall
detection, deadlock recovery, cancel, and probing -- now runs in Rust through the new `raydriver`
package. The **GRBL (Rust)** driver is experimental and can be selected per machine as a drop-in
alternative to GRBL (Serial). Dialects and settings remain plain Rayforge data.

## New Device Profiles

- **Creality Falcon A1 Pro** (thanks to @atkaper, #461)
- **Creality Falcon 2 Pro 22W**, shipping camera lens calibration and image settings as a starting
  default; the 40W profile's work area is corrected to the official 400 x 415 mm spec (#463)

## Fixes and Minor Improvements

- Selecting a driver that requires connection details no longer crashes the app with a GTK assertion
  and a crash loop at startup (#415)
- Importing DXF or LBRN2 files on macOS no longer produces a garbled, wrongly scaled result caused
  by the generic binary MIME type routing files to the Ruida importer (#384)
- Material textures are included again in wheel-based installs, repairing blank material thumbnails
  on flatpak and deb installs (#419)
- The Machine Settings dialog is now a single instance per main window instead of opening duplicates
  (#416 -- thank you @StevenIsaacs)
- Print and cut: the wizard now launches on machines whose position reports include an extra rotary
  axis (#394), and capturing an alignment point accounts for Pointer Alignment (#478)
- GRBL: reading device settings no longer aborts on grblHAL bitmask values (#401), and the material
  test grid is repaired for fractional row and column counts (#405 -- thank you @atkaper)
- macOS: the full-screen main window stays visible when a dialog closes on top of it (#453)
- The ChArUco calibration card is detected again on blurry, unevenly lit frames by falling back
  through progressively more tolerant detection passes (#443, #465)
- G-code no longer contains zero-length travel moves (raygeo)
- Serial ports can be bound by USB VID:PID instead of a device path, so auto-reconnect follows the
  machine after the OS re-enumerates USB devices (#459)
- Configs, machine profiles, and recipes are persisted atomically with backups, so an interrupted
  save can no longer destroy them (#455)
- Cancelling a GRBL job now stops the machine immediately instead of draining the buffer, and a
  firmware that stays silent after a cancel (e.g. the Sculpfun iCube over Bluetooth) no longer
  leaves the machine controls dead (#428)
- SVGs import again when they contain a DOCTYPE declaration (such as Affinity Studio exports), and
  viewBox-only files import at the correct scale (#484, #434 -- thank you @MausRundung)
- The time estimate in the machine dropdown counts down again while a job runs, Z jogs are clamped
  to the configured extents, and the hardware settings page gains Z Min/Z Max rows
- Material test speeds are capped at the machine's live speed limit, and the raster default
  threshold is now 254, so only pure white stays unengraved
- raygeo and raydriver now ship aarch64 (ARM64) Linux wheels, so ARM64 installs use prebuilt
  binaries instead of compiling from source
- Hindi is now available
- Usage tracking (strictly opt-in) now reports the host OS and anonymous machine types (driver,
  laser type, optical power, bed size, rotary), so the project knows what hardware to prioritize.
  Machine names and identifiers are never sent
- GitHub Discussions are now linked in the About dialog and on the website

This release was made possible with the support of our Patreon supporters: DaveSys, froqstar,
old-man-and-the-seam, pghpete, Derek McTavish, starlynx.dev, and six anonymous supporters.

## Download Rayforge 1.12

- [Website](https://rayforge.org/)
- [GitHub Releases](https://github.com/barebaric/rayforge/releases)

## Join the Community

- [Discord](https://discord.gg/sTHNdTtpQJ)
- [GitHub Discussions](https://github.com/barebaric/rayforge/discussions)
- [Patreon](https://www.patreon.com/c/knipknap)
