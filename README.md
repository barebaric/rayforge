[![GitHub Release](https://img.shields.io/github/release/barebaric/rayforge.svg?style=flat)](https://github.com/barebaric/rayforge/releases/)
[![PyPI version](https://img.shields.io/pypi/v/rayforge)](https://pypi.org/project/rayforge/)
[![Snap Release](https://snapcraft.io/rayforge/badge.svg)](https://snapcraft.io/rayforge)
[![Launchpad PPA](https://img.shields.io/badge/PPA-blue)](https://launchpad.net/~knipknap/+archive/ubuntu/rayforge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Get it from the Snap Store](https://snapcraft.io/en/light/install.svg)](https://snapcraft.io/rayforge)
<a href="https://flathub.org/apps/org.rayforge.rayforge"><img alt="Get it from Flathub" src="website/content/docs/images/flathub-badge.svg" height="55"/></a>

# Rayforge

Rayforge is a modern, cross-platform 2D CAD, G-code sender and control software for GRBL-based laser cutters and engravers.
Built with Gtk4 and Libadwaita, it provides a clean, native interface for Linux and Windows, offering a full suite of tools
for both hobbyists and professionals.

<p align="center">
  <img src="website/content/assets/icon.svg" />
</p>

![Screenshot](website/content/docs/images/ss-main.png)

You can also check the [official Rayforge homepage](https://rayforge.org).
We also have a [Discord](https://discord.gg/sTHNdTtpQJ).


## Key Features

| Feature                      | Description                                                                                                      |
| :--------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| **Modern UI**                | Polished and modern UI built with Gtk4 and Libadwaita. Supports system, light, and dark themes.                  |
| **Parametric Sketch Editor** | Create precise, constraint-based 2D designs with geometric and dimensional constraints.                         |
| **Multi-Layer Operations**   | Assign different operations (e.g., engrave then cut) to layers in your design.                                   |
| **Versatile Operations**     | Supports Contour, Raster Engraving (with cross-hatch fill), Shrink Wrap, and Depth Engraving.                    |
| **Overscan & Kerf Comp.**    | Improve engraving quality with overscan and ensure dimensional accuracy with kerf compensation.                  |
| **2.5D Cutting**             | Perform multi-pass cuts with a configurable step-down between each pass for thick materials.                     |
| **3D G-code Preview**        | Visualize G-code toolpaths in 3D to verify the job before sending it to the machine.                             |
| **Multi-Machine Profiles**   | Configure and instantly switch between multiple machine profiles.                                                |
| **GRBL Firmware Settings**   | Read and write firmware parameters (`$$`) directly from the UI.                                                  |
| **Comprehensive 2D Canvas**  | Full suite of tools: alignment, transformation, measurement, zoom, pan, and more.                                |
| **Advanced Path Generation** | High-quality image tracing, travel time optimization, path smoothing, and spot size interpolation.               |
| **Holding Tabs**             | Add tabs to contour cuts to hold pieces in place. Supports manual and automatic placement.                       |
| **G-code Macros & Hooks**    | Run custom G-code snippets before/after jobs. Supports variable substitution.                                    |
| **Broad File Support**       | Import from SVG, DXF, PDF, JPEG, PNG, BMP, and even Ruida files (`.rd`).                                         |
| **Multi-Laser Operations**   | Choose different lasers for each operation in a job                                                              |
| **Camera Integration**       | Use a USB camera for workpiece alignment, positioning, and background tracing.                                   |
| **Cross-Platform**           | Native builds for Linux and Windows.                                                                             |
| **Extensible**               | Open development model makes it easy to [add support for new devices](website/content/docs/developer/driver.md). |
| **Multi-Language**           | Available in English, Portuguese, Spanish, and German.                                                           |
| **G-code Dialects**          | Supports GRBL, Smoothieware, and other GRBL-compatible firmwares.                                                |

## Device Support

| Device Type      | Connection Method       | Notes                                                          |
| :--------------- | :---------------------- | :------------------------------------------------------------- |
| **GRBL**         | Serial Port             | Supported since version 0.13. The most common connection type. |
| **GRBL**         | Network (WiFi/Ethernet) | Connect to any GRBL device on your network.                    |
| **Smoothieware** | Telnet                  | Supported since version 0.15.                                  |

## Installation

For installation instructions [refer to our homepage](https://rayforge.org/docs/0.22/getting-started/installation.html).

## Development

For detailed information about developing for Rayforge, including setup instructions,
testing, and contribution guidelines, please see the
[Developer Documentation](https://rayforge.org/docs/latest/developer/getting-started/).

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
