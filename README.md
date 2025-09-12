[![GitHub Release](https://img.shields.io/github/release/barebaric/rayforge.svg?style=flat)](https://github.com/barebaric/rayforge/releases/)
[![PyPI version](https://img.shields.io/pypi/v/rayforge)](https://pypi.org/project/rayforge/)
[![Snap Release](https://snapcraft.io/rayforge/badge.svg)](https://snapcraft.io/rayforge)
[![Launchpad PPA](https://img.shields.io/badge/PPA-blue)](https://launchpad.net/~knipknap/+archive/ubuntu/rayforge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Rayforge

Rayforge is a powerful, open-source, and cross-platform software for controlling your
laser cutter and engraver. Designed with a modern and intuitive interface, it provides
all the tools you need to bring your creations to life on Linux and Windows.

![Screenshot](docs/ss-main.png)

## Why Rayforge?

Rayforge combines a user-friendly experience with advanced features, making it suitable
for both hobbyists and professionals. Whether you're engraving complex raster images or
cutting precise vector shapes, Rayforge offers a streamlined workflow from design to final
product.

## Features

| Feature                       | Description                                                                                          |
| :---------------------------- | :--------------------------------------------------------------------------------------------------- |
| **Intuitive User Interface**  | Polished and modern UI built with Gtk4 and Libadwaita.                                               |
| **Multi-Layer & Multi-Step**  | Assign different operations (e.g., engrave then cut) to layers in your design.                       |
| **Versatile Operations**      | Supports Contour, External Outline, and Raster Engraving.                                            |
| **2.5D Cutting**              | Perform multi-pass cuts with a step-down between each pass for thicker materials.                    |
| **3D G-code Preview**         | Visualize your G-code paths in a 3D view to verify the final job before you cut.                     |
| **Multi-Machine Support**     | Configure and instantly switch between multiple machine profiles.                                    |
| **Firmware Settings**         | Read and write firmware parameters directly on GRBL-based devices.                                   |
| **Powerful Canvas**           | Full suite of tools: alignment, transformation, measurement, zoom, pan, and more.                    |
| **Optimized Path Generation** | Intelligent path interpolation based on spot size, with travel time optimization and path smoothing. |
| **Broad File Support**        | Import from popular formats including SVG, DXF, PDF, PNG, and even Ruida files.                      |
| **Camera Support**            | Use a camera for live positioning, de-distortion, and aligning workpieces on the bed.                |
| **Cross-Platform**            | Native support for both Linux and Windows.                                                           |
| **Extensible**                | Open development model makes it easy to [add support for your own laser](docs/driver.md).            |
| **Multi-Language**            | Available in English, Portuguese, Spanish, and German.                                               |
| **G-code Dialects**           | Support for GRBL, Smoothieware, and other GRBL-compatible firmwares.                                 |
| **Customization**             | Switch between system, light, and dark themes.                                                       |
| **And much more...**          | Job framing, air assist control, console access, control buttons, etc.                               |

## Device Support

| Device Type      | Connection Method       | Notes                                                          |
| :--------------- | :---------------------- | :------------------------------------------------------------- |
| **GRBL**         | Serial Port             | Supported since version 0.13. The most common connection type. |
| **GRBL**         | Network (WiFi/Ethernet) | Connect to any GRBL device on your network.                    |
| **Smoothieware** | Telnet                  | Supported since version 0.15.                                  |

## Installation

### Windows

The easiest way to get started on Windows is to download the latest installer from our
**[releases page](https://github.com/barebaric/rayforge/releases/)**.

### Linux

We offer several installation methods for Linux.

#### Ubuntu & Derivatives (via PPA)

For users on Ubuntu and its derivatives (like Linux Mint, Pop!_OS), the recommended method is our official PPA. This integrates directly with your system's package manager and provides automatic updates.

> [!NOTE]
> The PPA supports **Ubuntu 24.04 LTS and newer**.

Open a terminal and run the following commands:
```bash
sudo add-apt-repository ppa:knipknap/rayforge
sudo apt update
sudo apt install rayforge
```

#### Cross-Distro (Recommended via Snap)

For other Linux distributions, or if you prefer a sandboxed application, the recommended method is installing via the Snap Store. It includes all necessary dependencies in a single package.

[![Get it from the Snap Store](https://snapcraft.io/en/light/install.svg)](https://snapcraft.io/rayforge)

> [!IMPORTANT]
> To grant the application access to your camera and serial port, run the
> following commands once after installation!

```bash
sudo snap connect rayforge:camera
```

If you want to be able to use USB Serial connections, you need to to the following:

1. Update to the latest Rayforge edge release.
2. Plug your laser into a USB port.
3. Open a terminal
4. Execute `sudo snap set system experimental.hotplug=true`
5. Execute `sudo snap connect rayforge:serial-port`. This will fail if your laser is not connected via USB!
6. Start Rayforge

#### Advanced: From Source (via PIP)

For advanced users who prefer not to use Snap or a PPA, Rayforge can be installed via PIP.
You will need to manually install system dependencies first. On Debian/Ubuntu-based
systems, this can be done with:

```bash
sudo apt update
sudo apt install python3-pip python3-gi gir1.2-gtk-3.0 gir1.2-adw-1 gir1.2-gdkpixbuf-2.0 libgirepository-1.0-dev libgirepository-2.0-0 libvips42t64 libpotrace-dev libagg-dev libadwaita-1-0 libopencv-dev

pip3 install rayforge
```

_(Note: Package names may differ slightly on other distributions.)_

### Other Operating Systems (e.g., macOS)

While there are no official installers for other operating systems at this time,
Rayforge is built with cross-platform code. It may be possible to install it from
source using the PIP instructions above. Contributions for packaging on other
platforms are welcome!

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

Contributions are what make the open-source community such an amazing place.
We welcome any contributions, from bug reports to new features!

- **Report a Bug:** Find a problem? Let us know by [opening an issue](https://github.com/barebaric/rayforge/issues).
- **Suggest a Feature:** Have a great idea? We'd love to hear it.
- **Write Code:** Check out our open issues. If you want to add a major feature, please discuss it with us first.
- **Add a Driver:** To add support for your machine, please see the [driver development guide](docs/driver.md).
- **Help with Packaging:** We are actively looking for help with packaging, especially for **[Flatpak](flatpak/)**. If you have experience with this, your help would be invaluable!

## Development

This project uses [**Pixi**](https://pixi.sh/) to manage dependencies and development environments.
It provides a single, cross-platform tool for a reproducible setup.

### Prerequisites

Ensure you have Pixi installed on your system. You can find the installation instructions
on the [official website](https://pixi.sh/latest/installation/).

### 1. Initial Setup

After cloning the repository, setting up the development environment is a single command:

```bash
pixi install
```

This will read the `pixi.toml` file, create a local `.pixi` environment,
and install all specified conda and pypi dependencies in it.

### 2. Running the Application

To run the main application, use the `run` task defined in `pixi.toml`:

```bash
pixi run rayforge
```

### 3. Running Tests

To run the test suite, use the `test` task:

```bash
pixi run test
```

### 4. Working in the Activated Environment

For a more interactive workflow (similar to activating a virtual environment), you can start a shell within the project's environment:

```bash
pixi shell
```

Now, your shell is configured with all the project's dependencies. You can run commands directly without the `pixi run` prefix:

```bash
# Inside the pixi shell
rayforge
pytest -v
```

Exit the shell by typing `exit`.

### 5. Managing Dependencies

The `pixi.toml` file is the single source of truth for all dependencies. Do not use `pip` or `conda` directly to add packages.

**To add a new Conda package:**

```bash
# Example: Add numpy from conda-forge
pixi add numpy
```

**To add a new PyPI package:**

```bash
# Example: Add requests from PyPI
pixi add --pypi requests
```

Pixi will automatically resolve the dependencies and update the `pixi.toml` and `pixi.lock` files.

### 6. Translation Workflow

The following tasks are available for managing language translations:

*   **To extract new strings from the code and update `.po` files:**
    ```bash
    pixi run update-translations
    ```

*   **To compile `.po` files into `.mo` files for the application to use:**
    ```bash
    pixi run compile-translations
    ```

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
