# Ruida RPA Driver

The Ruida RPA driver connects Rayforge to Ruida-based laser controllers. It supports both a direct connection (USB or UDP) and a TUI RPC connection via the Ruida Protocol Analyzer. A machine settings configuration is available as the "Ruida RPA (Connect via Ruida Protocol Analyzer)" device.

## Installation

Installation is similar to the Rayforge instructions at: `https://rayforge.org/docs/getting-started/installation#linux-pixi`. However, there are two differences:

- Clone from: `https://github.com/StevenIsaacs/rayforge`
- Switch to the correct beta branch (currently `beta3`).

ruida-pa is installed from PyPI as part of the default pixi environment. To diagnose problems against a local ruida-pa source checkout, point pixi at it via a `dependency-override` or install it into the environment with `pixi run pip install --no-deps --force-reinstall <path-to-ruida-pa>`.

## Machine Settings Configuration

To configure Rayforge to use the ruidarpa driver, select the **Ruida RPA (Connect via Ruida Protocol Analyzer)** device in the machine settings.

Configuring both the **UDP Hostname** and the **USB device** enables automatic swap between connections when cables are connected and disconnected. USB is preferred when both cables are connected because of slightly better performance (no ACK handshake).

The **magic number** setting is used for other controllers. The default of `0x88` will work with many common controllers. The value is entered in hex.

Enabling and disabling **TUI RPC** automatically switches between using the TUI versus a direct connection.

## Starting the TUI

The TUI is started using the command:

```
pixi run rpa-script --tui
```

This requires the pixi environment, which is available when running Rayforge from a source checkout. When running Rayforge from an install package (e.g. snap, deb, flatpak), the pixi environment is not available. In that case, install ruida-pa from PyPI in a Python virtual environment and run the TUI directly:

```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install ruida-pa
rpa-script --tui
```

`pip install ruida-pa` installs all dependencies (bokeh, pyserial, textual, rpyc) automatically, so the TUI works out of the box.

The ruida-pa guides are located at: `https://github.com/StevenIsaacs/ruida-pa/tree/main/docs/guides`

## Using the TUI and RPC to Capture Jobs

The TUI and RPC can be used to capture and display Rayforge jobs. To capture a job, start the TUI and then send a job from Rayforge with TUI RPC enabled in the machine settings. The TUI will display the job data as it is received.

Use the `/autosave` command within the TUI to automatically save captured job data. This writes gluescript (`.cglu`), rpascript (`.rds`), RDWorks compatible (`.rd`), and HTML output files to the TUI's working directory.

## Loading and Editing Script Files

The TUI can load existing gluescript (`.cglu`), rpascript (`.rds`) and RDWorks (`.rd`) files for review and editing. To edit the loaded script files, a simple internal editor is provided. However, for large files it is most convenient to use and external editor and reload the edited script. This workflow is most useful when characterizing the behavior of a Ruida controller with specific rpascript commands -- load a captured or known script, inspect the commands, and edit them to experiment with the controller's response to different command sequences and parameter.

## Viewing and Sharing HTML Output

The TUI generates HTML files as part of the `/autosave` and `/plot` commands. These HTML files provide a visual representation of the captured job data and can be viewed in any web browser. Share these HTML files with others to review and discuss captured job output without needing access to the TUI or the raw script files.

## Golden Files

The `golden/` directory contains test fixture files that can be used to verify correct installation and to compare results. Each job is represented by a set of files sharing the same basename across the following extensions:

- **`.cglu`** -- Gluescript source files describing the job (e.g. `declare_job`, `declare_layer`).
- **`.ryp`** -- Binary Rayforge project files.
- **`.rd`** -- Binary raw Ruida data files.
- **`.rds`** -- Assembled Ruida command listings (rpascript) generated from the gluescript, such as `REF_POINT_MACHINE` and `SET_ABSOLUTE`.
- **`.html`** -- HTML output files providing a visual representation of the job.

The fixture basenames are: `rect`, `rect-fill`, `ellipse-fill`, `dithered`, `greyscale`, `test-grid-cut`, `test-grid-engrave`, and `text`. Each job has a complete set of files across all five extensions. These golden files serve as a baseline -- if your installation produces output that differs from the golden files, something may be misconfigured.

## Debug Using VSCode

A VSCode launch file is provided for debugging if necessary. Be sure to select the correct `pixi` environment (e.g. `default`) before running Rayforge using the debugger.

## Syntax Highlighting

VSCode compatible extensions for syntax highlighting both `gluescript` (`.cglu`) and `rpascript` (`.rds`) files are available at: `https://github.com/StevenIsaacs/ruida-pa/tree/main/.vscode/extensions`

## Limitations

- The TUI is not recommended for jobs containing large images because of the delay introduced by RPC. Currently there is no progress indication while large files are being transferred which can require a minute or two. Allow time for the transfer to complete. The TUI will display the beginning of a transfer and when it completed. If there is a comms failure during transfer, error messages will be displayed and recovery is automatic. In other words, no news is good news in this case.
- This version includes rudimentary acceleration and deceleration power compensation. This is used to reduce excessive burn (over-burn) at the ends of lines. Rayforge currently does not support a per workpiece settings for compensation. For now compensation is a machine setting which means it can only be applied on a per job basis. A more comprehensive solution to this problem is coming. There are two settings available:
  - **VECTOR power floor** Use this setting to set the minimum power when accelerating or decelerating. This is the over-burn compensation. Setting this to 100% effectively disables the setting.
  - **IMAGE power bias** A CO2 laser will not fire at extremely low power settings (typically below 8%). This settings adds a power bias to the per pixel power settings for an image. For example: When this is set to 8% and image pixel power value of 20% will be increased to 28%. NOTE: Power values of 0% are not affected by this setting. Instead, 0% pixels are converted to moves instead of cuts.
- Tested only with a RDC6442S controller on a Monport MP570 CO2 laser while running on Fedora Linux.
- **WARNING**: The depthmap engrave mode has not been tested. Because it can use the Z axis for raising and lowering the bed, use with caution. Machines having a Z axis may not have a hard limit switch so it is possible to crash the laser head into the bed. Machines equipped with a probe are still at risk if there is no flat surface for the probe to contact.
- Rotary is not yet supported because typical use requires swapping the Y axis connection with the rotary connection. More command characterization is required to instead use the actual U axis connection available on some controllers.
- There are many yet to be characterized Ruida commands. This implementation uses only the currently essential and well understood commands. Help is needed to add more commands to the well understood list.
