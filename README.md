# Rayforge

Rayforge is a software for laser cutters and engravers. It runs on Linux and
maybe on Windows, though the latter is pretty much untested.

It supports direct communication with GRBL based machines (network or serial).

![Screenshot](docs/ss-main.png)

## Features

| Feature                          | Description                                                |
| -------------------------------- | ---------------------------------------------------------- |
| Intuitive user interface         | Drag & drop reordering, focus on essentials                |
| Multi step operations            | For example, first engrave, then cut                       |
| Multiple operation types         | Contour, External Outline, Raster Engraving                |
| High quality path generation     | Interpolation based on spot size, path optimization        |
| Multiple input formats           | SVG, DXF, PDF, and PNG import are supported                |
| Open development                 | Easily [add support for your own laser](docs/driver.md)    |
| Cross-platform                   | Support for Linux and (experimental) support for Windows   |
| Camera support                   | Live camera feed, de-distortion, alignment                 |
| Much more                        | Framing, support for air assist, control buttons, ...      |


## Device support

| Device Type                      | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| GRBL (network based)             | Connect any GRBL based laser through WiFi or Ethernet        |
| GRBL (serial port based)         | Since version 0.13, serial GRBL based machines are supported |
| Smoothieware (Telnet based)      | Starting with version 0.15                                   |


### Screenshots

![Camera Alignment](docs/camera-alignment.png)
![Camera Image](docs/camera-image.png)
![Camera Overlay](docs/camera-overlay.png)
![Camera Settings](docs/camera-settings.png)


## Installation

### Linux

On Linux the only currently supported method is Snap:

[![Get it from the Snap Store](https://snapcraft.io/en/light/install.svg)](https://snapcraft.io/rayforge)

You can also install it through PIP if you know what you are doing. Something like this:

```
sudo apt install python3-pip-whl python3-gi gir1.2-gtk-3.0 gir1.2-adw-1 gir1.2-gdkpixbuf-2.0 libgirepository-1.0-dev libgirepository-2.0-0 libvips42t64

pip3 install rayforge
```

### Windows

Head over to the [releases page](https://github.com/barebaric/rayforge/releases/).

### Other operating systems

There is currently no installer for other operating systems - contributions are
welcome, in the form of Github workflow actions or build instructions.

If you know what you are doing, you may be able to install manually using
PIP on Windows or Mac - the source code should be fully cross-platform.


## Development

Setup:
```
sudo apt install python3-pip-whl python3-gi gir1.2-gtk-3.0 gir1.2-adw-1 libgirepository-1.0-dev libgirepository-2.0-0 libvips42t64
git clone git@github.com:barebaric/rayforge.git
cd rayforge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Driver development

If you want to develop a driver to support your machine with Rayforge,
please check the [driver development guide](docs/driver.md).
