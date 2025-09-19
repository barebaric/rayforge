[![GitHub Release](https://img.shields.io/github/release/barebaric/rayforge.svg?style=flat)](https://github.com/barebaric/rayforge/releases/)
[![PyPI version](https://img.shields.io/pypi/v/rayforge)](https://pypi.org/project/rayforge/)
[![Snap Release](https://snapcraft.io/rayforge/badge.svg)](https://snapcraft.io/rayforge)
[![Launchpad PPA](https://img.shields.io/badge/PPA-blue)](https://launchpad.net/~knipknap/+archive/ubuntu/rayforge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Rayforge

Rayforge is a modern, cross-platform G-code sender and control software for GRBL-based laser cutters and engravers. Built with Gtk4 and Libadwaita, it provides a clean, native interface for Linux and Windows, offering a full suite of tools for both hobbyists and professionals.

![Screenshot](docs/ss-main.png)

## Key Features

| Feature                      | Description                                                                                          |
| :--------------------------- | :--------------------------------------------------------------------------------------------------- |
| **Modern UI**                | Polished and modern UI built with Gtk4 and Libadwaita. Supports system, light, and dark themes.      |
| **Multi-Layer Operations**   | Assign different operations (e.g., engrave then cut) to layers in your design.                       |
| **Versatile Operations**     | Supports Contour, External Outline, and Raster Engraving with detailed settings.                     |
| **2.5D Cutting**             | Perform multi-pass cuts with a configurable step-down between each pass for thick materials.         |
| **3D G-code Preview**        | Visualize G-code toolpaths in 3D to verify the job before sending it to the machine.                 |
| **Multi-Machine Profiles**   | Configure and instantly switch between multiple machine profiles.                                    |
| **GRBL Firmware Settings**   | Read and write firmware parameters (`$$`) directly from the UI.                                      |
| **Comprehensive 2D Canvas**  | Full suite of tools: alignment, transformation, measurement, zoom, pan, and more.                    |
| **Advanced Path Generation** | Intelligent path interpolation based on spot size, with travel time optimization and path smoothing. |
| **Holding Tabs**             | Add tabs to contour cuts to hold pieces in place. Supports manual and automatic placement.           |
| **G-code Macros & Hooks**    | Run custom G-code snippets before/after jobs. Supports variable substitution.                        |
| **Broad File Support**       | Import from SVG, DXF, PDF, PNG, and even Ruida files (`.rd`).                                        |
| **Camera Integration**       | Use a USB camera for workpiece alignment, positioning, and background tracing.                       |
| **Cross-Platform**           | Native builds for Linux and Windows.                                                                 |
| **Extensible**               | Open development model makes it easy to [add support for new devices](docs/driver.md).               |
| **Multi-Language**           | Available in English, Portuguese, Spanish, and German.                                               |
| **G-code Dialects**          | Supports GRBL, Smoothieware, and other GRBL-compatible firmwares.                                    |

## Device Support

| Device Type      | Connection Method       | Notes                                                          |
| :--------------- | :---------------------- | :------------------------------------------------------------- |
| **GRBL**         | Serial Port             | Supported since version 0.13. The most common connection type. |
| **GRBL**         | Network (WiFi/Ethernet) | Connect to any GRBL device on your network.                    |
| **Smoothieware** | Telnet                  | Supported since version 0.15.                                  |

## Installation

### Windows

Download the latest installer from the **[Releases Page](https://github.com/barebaric/rayforge/releases/)**.

### Linux

We offer several installation methods for Linux.

#### Ubuntu & Derivatives (via PPA)

For users on Ubuntu and its derivatives (like Linux Mint, Pop!\_OS), the recommended
method is our official PPA. This integrates directly with your system's package
manager and provides automatic updates.

> [!NOTE]
> The PPA supports **Ubuntu 24.04 LTS and newer**.

Open a terminal and run the following commands:

```bash
sudo add-apt-repository ppa:knipknap/rayforge
sudo apt update
sudo apt install rayforge
```

#### Cross-Distro (Snap)

The Snap package includes all dependencies and runs in a sandbox. It is the recommended
method for most other Linux distributions.

[![Get it from the Snap Store](https://snapcraft.io/en/light/install.svg)](https://snapcraft.io/rayforge)

> [!IMPORTANT]
> The Snap is sandboxed and requires you to manually grant permissions for hardware
> access. Run these commands after installation.

For camera access:

```bash
sudo snap connect rayforge:camera
```

For USB serial port access:

```bash
# First, enable experimental hotplug support
sudo snap set system experimental.hotplug=true

# Connect your laser via USB, then run this command
sudo snap connect rayforge:serial-port
```

#### From Source (pip)

This method is for developers and advanced users. You must install system dependencies manually.

On Debian/Ubuntu-based systems:

```bash
sudo apt update
sudo apt install python3-pip python3-gi gir1.2-gtk-3.0 gir1.2-adw-1 gir1.2-gdkpixbuf-2.0 libgirepository-1.0-dev libgirepository-2.0-0 libvips42t64 libpotrace-dev libagg-dev libadwaita-1-0 libopencv-dev

pip3 install rayforge
```

_(Package names may differ on other distributions.)_

### Other Operating Systems (e.g., macOS)

There are no official builds for other platforms, but Rayforge may run from source via
the `pip` method. Contributions for packaging on other platforms are welcome.

## Screenshots

<details>
<summary><b>Click to expand screenshots</b></summary>

**Operation Settings**
![Contour Settings](docs/contour-settings.png)

**Machine & Device Configuration**
![General Settings](docs/machine-settings.png)
![Device Settings](docs/machine-device.png)
![Advanced Settings](docs/machine-advanced.png)
![Laser Settings](docs/machine-laser.png)

**Integrated Camera Support**
![Camera Settings](docs/machine-camera.png)
![Camera Alignment](docs/camera-alignment.png)
![Camera Image](docs/camera-image.png)
![Camera Overlay on Worksurface](docs/camera-overlay.png)

</details>

## Contributing

We welcome contributions of all kinds! Whether you're fixing a bug, adding
a feature, or improving documentation, your help is appreciated.

- **Report a Bug:** Open an [issue](https://github.com/barebaric/rayforge/issues)
  with a clear description and steps to reproduce.
- **Suggest a Feature:** Start a discussion or open a feature request issue.
- **Submit a Pull Request:** Please discuss major changes in an issue first.
- **Add a Driver:** See the [driver development guide](docs/driver.md) to add
  support for your hardware.

### Packaging Help Wanted

We are actively seeking help to create and maintain a **[Flatpak](flatpak/)** package.
If you have experience with Flatpak, your contribution would be highly valuable!

## Development

This project uses [**Pixi**](https://pixi.sh/) to manage dependencies and
development environments for a reproducible setup.

### Prerequisites

First, install Pixi by following the instructions on the
[official website](https://pixi.sh/latest/installation/).

### 1. Setup Environment

Clone the repository and install dependencies.

```bash
# Install Gtk/Adwaita system dependency (Debian/Ubuntu)
sudo apt install gir1.2-adw-1

# Let Pixi handle the rest
pixi install
```

This command reads `pixi.toml` and installs all conda and pip dependencies into a local
`.pixi` virtual environment.

### 2. Run the Application

```bash
pixi run rayforge
```

### 3. Run Tests

```bash
pixi run test
```

### 4. Activate the Development Shell

For an interactive workflow, you can activate a shell within the project's environment.

```bash
pixi shell
```

Inside this shell, you can run commands directly without the `pixi run` prefix
(e.g., `rayforge`, `pytest`). Type `exit` to leave the shell.

### 5. Manage Dependencies

Always use Pixi to manage dependencies to ensure the `pixi.lock` file is updated.

```bash
# Add a conda package
pixi add numpy

# Add a PyPI package
pixi add --pypi requests
```

### 6. Translation Workflow

The following tasks are available for managing language translations:

#### Update `.po` files from source code

```bash
pixi run update-translations
```

#### Compile `.po` files to binary `.mo` files

```bash
pixi run compile-translations
```

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
