# Rayforge

Rayforge is a software for laser cutters and engravers.

![Screenshot](docs/ss-main.png)


## Installation

Currently the only supported installation method is through PIP.
I could not figure out how to deploy a Flatpak to Flathub yet.

```
pip3 install rayforge
```

## Features

| Feature                          | Description                                             |
| -------------------------------- | ------------------------------------------------------- |
| Intuitive user interface         | Drag & drop reordering, focus on essentials             |
| Multi step operations            | For example, first engrave, then cut                    |
| Mutltiple operation types        | Countour, External Outline, Raster Engraving            |
| High quality path generation     | Interpolation based on spot size, path optimization     |
| Multiple input formats           | SVG, DXF, and PNG import are supported                  |
| Direct device support            | Easily [add support for your own laser](docs/driver.md) |
| Much more                        | Air assist, device drivers                              |


## Development

Setup:
```
git clone git@github.com:barebaric/rayforge.git
cd rayforge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Driver development

If you want to develop a driver to support your machine with Rayforge,
please check the [driver development guide](docs/driver.md).
