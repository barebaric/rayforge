---
slug: rayforge-1-11-beta1-sketcher-physical-burn-auto-config
title: 'Rayforge 1.11.0-beta1 - Sketcher Tools, Physical Burn Model, Auto-Configuration'
authors: rayforge_team
tags: [release, 1.11, beta, sketcher, 3d-preview, discovery]
description: "Rayforge 1.11.0-beta1 brings new sketcher mirror/duplicate/circular-array tools, a physical burn model in the 3D preview, and automatic machine configuration with device discovery."
---

A while back I ran a vote asking where I should focus next. You spoke
clearly: sketcher improvements came out on top, CNC second, and a
better 3D preview third.

This cycle delivered the sketcher work you asked for -- and I went
ahead and built the better preview too, even though it wasn't the
runner-up. (CNC folks: you're on deck.)

<!-- truncate -->

## Sketcher: Mirror, Duplicate, and Circular Arrays

The sketcher picked up several new tools this cycle.

- You can now **mirror** the selection vertically or horizontally across
  its center.
- **Ctrl+D** duplicates the selection in-place, and the **arrow keys**
  nudge selected entities around.
- The headline tool is the new **circular array** (polar pattern). Drop a
  guide circle, set the count and angle, and copies are generated
  parametrically. The dialog is non-modal with a live preview, the guide
  circle's radius resizes the whole array, and double-clicking it
  reopens the editor.
- Rectangles got a small upgrade: they now auto-create a center point,
  and **Shift**-click draws the rectangle symmetrically around the start
  point -- matching the ellipse tool.

A bunch of undo/redo and text-editing bugs were fixed, and Ctrl+S now
saves the document instead of accidentally triggering the symmetry
tool.

## 3D Preview: A Physical Burn Model

The 3D preview no longer floats engraving as a flat overlay on top of
the stock. Laser raster and vector ops now **char the stock itself** --
the preview actually shows burning, using a physically motivated model
that accounts for your laser's wavelength, optical wattage, and spot
size, and the material's absorption at that wavelength.

- It works for rotary too -- engraving bakes into the rotary stock.
- Stock also got smarter: the material manager now guarantees a default
  stock material, and new stock assets and layers pick it up
  automatically.

A caveat: the burn model is physically motivated, but it's not yet well
calibrated. To get the charring to match real materials across
different lasers, I need your help. If you can run a material test grid
and **send me a photo of the result along with your machine data**
(laser wavelength, optical wattage, spot size, speed, and the
powers/depths you tested), that data lets me tune the model to
reality. More samples = a more accurate preview for everyone.

## Auto-Configuration: Plug It In and Go

I've been chipping away at the setup friction. The wizard now does a
lot more of the work for you.

- On first launch, the machine configuration wizard opens automatically
  and **discovers nearby devices** -- over USB and over the network.
  OctoPrint servers and ESP3D boards now appear in the wizard alongside
  USB serial devices.
- Discovered devices are **matched to built-in profiles** when possible,
  so you often just pick your machine from the list instead of filling
  everything in by hand. GRBL dialects and OctoPrint/Smoothieware
  settings are detected automatically.
- The wizard also checks **serial-port and camera permissions** before
  discovery starts. If something is missing, a new first page explains
  how to fix it on your platform, with commands you can copy with one
  click.

One more detail: USB serial ports now sort ahead of hardware ports in
the port selector, which gained a two-line row showing the device path
plus a short description. A configured but unplugged port stays pinned
to the top so it doesn't fall off the list.

## Fixes and Minor Improvements

- The laser and aux outputs are now turned off when a job is stopped or
  aborted, so a stuck "laser on" command no longer lingers after an
  interrupted cut
- Slow GRBL serial handshakes with grblHAL fixed
- New `--script` CLI flag runs Python at startup so scripts can register
  plugins and template functions; built-in text-box template functions
  (date, time, uuid, etc.) are available out of the box
- `.rfs` sketch files now record a schema version for forward
  compatibility
- New right panel display mode setting
- Refined app icon
- Updated translations (de, es, fr, pt, uk, zh_CN)
- raygeo upgraded to 1.47.0

This beta was 70 commits, 460 files, and around 20k net lines of code.

## Download Rayforge 1.11.0-beta1

- [GitHub Releases](/docs/getting-started/installation)
- [Website](https://rayforge.org/)

## Join the Community

- [Discord](https://discord.gg/sTHNdTtpQJ)
- [Patreon](https://www.patreon.com/c/knipknap)
